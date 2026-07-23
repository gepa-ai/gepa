"""AutoResearch engine: black-box optimization via a Claude Code subprocess.

The api creates an :class:`EvalServer` and starts its HTTP endpoint. This
engine launches one ``claude --print`` session inside a work_dir laid out
with ``program.md``, ``candidate.txt``, ``best_candidate.txt``, and an
``eval.sh`` script that POSTs to the eval server. Budget enforcement happens
server-side (HTTP 429 on exhaustion); LLM cost is bounded via
``--max-budget-usd``.

Train/val handling: the agent sees one combined set. When the task has both
train and val splits, both are materialized into ``train/<id>.json`` and
``eval.sh`` scores against the combined pool. The agent never knows val
existed; the test set is sealed at the eval server's HTTP layer and is
reachable only via the in-process Python API (used by the runner for
held-out reporting).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.oa.budget import BudgetTracker
from gepa.oa.engine import Result
from gepa.oa.engines.claude_utils import copy_session_transcript
from gepa.oa.sandbox import DENY_WEB_TOOLS, bwrap_prefix, claude_permission_args
from gepa.oa.task import seed_as_text
from gepa.oa.utils import example_to_json

if TYPE_CHECKING:
    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.eval_server import EvalServer
    from gepa.oa.task import Task


@dataclass
class AutoResearchConfig:
    """Engine-specific options for :class:`AutoResearchEngine`.

    Populated from ``OptimizeAnythingConfig.engine_config``.

    Attributes:
        model: Claude model id. Versioned default — pin for reproducibility;
            pass ``"sonnet"`` / ``"opus"`` to track Anthropic's current default.
        ralph: Continue one Claude Code session via repeated
            ``claude --resume`` while budget remains. Stops early on a zero-cost
            iteration (no progress) or a claude error.
        max_no_eval_seconds: Terminate the subprocess after this many seconds
            without an eval call. ``None`` disables the guard.
        handoffs: Prior-stage artifacts for sequential compositions — each item
            describes one stage (paths materialized under ``handoff/`` and
            referenced from ``program.md``).
        effort: ``claude --effort`` value (CLI flag).
        max_thinking_tokens: Fixed thinking-token budget (``MAX_THINKING_TOKENS``).
    """

    model: str = "claude-sonnet-4-6"
    ralph: bool = True
    max_no_eval_seconds: float | None = None
    handoffs: list[dict[str, Any]] | None = None
    effort: str | None = None
    max_thinking_tokens: int | None = None

    def __post_init__(self) -> None:
        # yaml/CLI may pass ralph as a string ("false") and max_no_eval_seconds
        # as a string/int; coerce to the declared types.
        self.ralph = _config_bool(self.ralph)
        if self.max_no_eval_seconds is not None:
            self.max_no_eval_seconds = float(self.max_no_eval_seconds)


# Nudge fed to claude --resume on each Ralph iteration.
RALPH_CONTINUE_PROMPT = (
    "Continue iterating on the candidate. Re-read program.md if needed. "
    "Run ./eval.sh as appropriate. "
    "Keep refining best_candidate.txt until you exhaust the budget "
    "or genuinely cannot find another improvement."
)

# Stop spawning new Ralph iterations when the LLM budget has less than this
# much headroom left — not worth spawning another CLI for pennies.
_RALPH_MIN_HEADROOM_USD = 0.05

# Hard safety cap on Ralph iterations — prevents pathological infinite
# loops if budget tracking gets confused. Real runs hit budget exhaustion
# long before this.
_RALPH_SAFETY_ITERATION_CAP = 1000

# How long to wait after the eval-count budget reports exhausted before we
# SIGTERM the subprocess. Lets in-flight 429 responses drain cleanly.
_BUDGET_EXHAUSTION_GRACE_SECONDS = 120.0

_BUDGET_EXHAUSTED_MARKER = "BUDGET_EXHAUSTED"


EVAL_SCRIPT_SINGLE = """\
#!/usr/bin/env bash
# Usage: ./eval.sh <candidate_file>
set -euo pipefail
CANDIDATE_FILE="$1"
SERVER_URL="{server_url}"
CANDIDATE=$(cat "$CANDIDATE_FILE")
BODY=$(jq -n --arg c "$CANDIDATE" '{{candidate: $c}}')
RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/evaluate" \\
    -H "Content-Type: application/json" -d "$BODY")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')
echo "$BODY"
if [ "$HTTP_CODE" = "429" ]; then echo "BUDGET_EXHAUSTED" >&2; exit 1; fi
"""

EVAL_SCRIPT_DATASET = """\
#!/usr/bin/env bash
# Usage: ./eval.sh <candidate_file>                   # all training examples
#        ./eval.sh <candidate_file> --ids id1,id2,id3 # specific examples
set -euo pipefail
CANDIDATE_FILE="$1"; shift
SERVER_URL="{server_url}"
CANDIDATE=$(cat "$CANDIDATE_FILE")
IDS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ids) IDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -n "$IDS" ]; then
    IDS_JSON=$(echo "$IDS" | jq -R 'split(",")')
    BODY=$(jq -n --arg c "$CANDIDATE" --argjson ids "$IDS_JSON" '{{candidate: $c, example_ids: $ids}}')
else
    BODY=$(jq -n --arg c "$CANDIDATE" '{{candidate: $c}}')
fi
RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/evaluate_examples" \\
    -H "Content-Type: application/json" -d "$BODY")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')
echo "$BODY"
if [ "$HTTP_CODE" = "429" ]; then echo "BUDGET_EXHAUSTED" >&2; exit 1; fi
"""


_PROGRAM_MD = """\
# Task: {name}

{optional_sections}\
## Candidate
Iteratively improve the candidate in `candidate.txt`.
Initial candidate: {candidate_len} chars.

{handoff_section}\
{eval_section}

## Budget
{budget_section}

## Strategy
{strategy_section}

## Rules
{rules_section}
"""

_EVAL_GENERALIZATION = """\
## Evaluation
This is a **generalization** task with {train_size} training examples.
Training examples are in `train/` as individual JSON files.

### Train evaluation
```bash
# Evaluate on all training examples
./eval.sh candidate.txt

# Evaluate on specific examples
./eval.sh candidate.txt --ids example_0,example_1,example_2
```{cost_line}"""

_EVAL_SINGLE = """\
## Evaluation
This is a **single-task** optimization.
```bash
./eval.sh candidate.txt
```{cost_line}"""


def _budget_section(budget: BudgetTracker, max_token_cost: float | None) -> str:
    lines: list[str] = []
    if budget.max_evals is not None:
        lines.append(f"- **Evaluation cap:** {budget.max_evals} calls (each eval = 1 unit).")
    if max_token_cost is not None:
        lines.append(
            f"- **LLM token budget:** ${max_token_cost:.2f} "
            "(enforced by the Claude CLI; you'll stop automatically when exhausted)."
        )
    if not lines:
        lines.append("- No hard budget limit — iterate freely.")
    return "\n".join(lines)


def _strategy_section(task: Task) -> str:
    """Explicit experiment loop, in the spirit of karpathy/autoresearch.

    Two modes:
      - dataset task  → ``./eval.sh`` on the full training pool is the signal.
      - single task   → ``./eval.sh`` runs the single-task ``eval_fn``.
    """
    has_train_dir = task.has_dataset and bool(task.train_set)

    if has_train_dir:
        edit_step = (
            "2. **Hypothesize and edit.** Read failures in `train/` "
            "(spot-check with `./eval.sh candidate.txt --ids id1,id2` if useful), "
            "form a hypothesis, edit `candidate.txt`."
        )
    else:
        edit_step = (
            "2. **Hypothesize and edit.** Use the judge feedback from the prior eval, "
            "form a hypothesis, edit `candidate.txt`."
        )

    return (
        "Run this as an explicit experiment loop forever, until you hit the max score (if there is one).\n"
        "1. **Baseline.** Run `./eval.sh candidate.txt` on the seed; note the score and any judge feedback.\n"
        f"{edit_step}\n"
        "3. **Score.** Run `./eval.sh candidate.txt`.\n"
        "4. **Keep or discard.** If the score improved over the best so far, "
        "`cp candidate.txt best_candidate.txt`. "
        "Otherwise `cp best_candidate.txt candidate.txt` to revert.\n"
        "5. Repeat from step 2"
    )


def _perfect_score_section(perfect_score: float | None) -> str:
    if perfect_score is None:
        return ""
    return (
        f"\n## Perfect Score\n"
        f"The maximum achievable score is **{perfect_score}**. "
        f"Once you reach this score, stop iterating — further improvements are impossible.\n"
    )


def _rules_section(task: Task, budget: BudgetTracker) -> str:
    del task
    exhaust_rule = (
        "\n- When the budget is exhausted, scripts return BUDGET_EXHAUSTED." if budget.max_evals is not None else ""
    )
    return (
        f"- You cannot modify eval.sh or the server.\n- Focus on meaningful improvements each iteration.{exhaust_rule}"
    )


def _handoff_section(handoffs: list[dict[str, Any]] | None) -> str:
    if not handoffs:
        return ""
    return (
        "## Prior Optimizer Handoff\n"
        "Prior sequential-stage artifacts are available in `handoff/`. "
        "Each stage has a `summary.json`, optional `best_candidate.txt`, and optional eval JSON records. "
        "Use them as context for what has already been tried and scored.\n\n"
    )


def _build_program_md(
    task: Task,
    budget: BudgetTracker,
    *,
    max_token_cost: float | None,
    perfect_score: float | None,
    handoffs: list[dict[str, Any]] | None,
) -> str:
    optional = ""
    if task.objective:
        optional += f"## Objective\n{task.objective}\n\n"
    if task.background:
        optional += f"## Background\n{task.background}\n\n"

    has_eval_budget = budget.max_evals is not None

    if task.has_dataset and task.train_set:
        # Combined visible pool = train + val. Agent never knows val existed.
        train_size = len(task.train_set) + (len(task.val_set) if task.val_set else 0)
        cost_line = (
            f"\nEach example costs 1 budget unit. A full train eval costs {train_size} units."
            if has_eval_budget
            else ""
        )
        eval_section = _EVAL_GENERALIZATION.format(
            train_size=train_size,
            cost_line=cost_line,
        )
    else:
        cost_line = "\nEach eval costs 1 budget unit." if has_eval_budget else ""
        eval_section = _EVAL_SINGLE.format(cost_line=cost_line)

    return _PROGRAM_MD.format(
        name=task.name,
        optional_sections=optional,
        candidate_len=len(task.seed_candidate or ""),
        handoff_section=_handoff_section(handoffs),
        eval_section=eval_section,
        budget_section=_budget_section(budget, max_token_cost) + _perfect_score_section(perfect_score),
        strategy_section=_strategy_section(task),
        rules_section=_rules_section(task, budget),
    )


def _materialize_sandbox(
    work_dir: Path,
    task: Task,
    server: EvalServer,
    budget: BudgetTracker,
    *,
    max_token_cost: float | None = None,
    perfect_score: float | None = None,
    handoffs: list[dict[str, Any]] | None = None,
) -> None:
    server_url = server.url
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "program.md").write_text(
        _build_program_md(task, budget, max_token_cost=max_token_cost, perfect_score=perfect_score, handoffs=handoffs)
    )
    (work_dir / "candidate.txt").write_text(seed_as_text(task.seed_candidate))
    (work_dir / "best_candidate.txt").write_text(seed_as_text(task.seed_candidate))

    eval_template = EVAL_SCRIPT_DATASET if task.has_dataset else EVAL_SCRIPT_SINGLE
    eval_script = work_dir / "eval.sh"
    eval_script.write_text(eval_template.format(server_url=server_url))
    eval_script.chmod(0o755)

    if task.train_set:
        train_dir = work_dir / "train"
        train_dir.mkdir(exist_ok=True)
        # Materialize train + val together as a single visible pool. The agent
        # sees one combined "training" set and never knows val existed; test
        # is sealed at the eval-server HTTP layer.
        for split in ("train", "val"):
            for eid, ex in server.iter_split(split):
                (train_dir / f"{eid}.json").write_text(json.dumps(example_to_json(eid, ex), indent=2, default=str))

    _materialize_handoff(work_dir, handoffs)


def _materialize_handoff(work_dir: Path, handoffs: list[dict[str, Any]] | None) -> None:
    if not isinstance(handoffs, list) or not handoffs:
        return
    handoff_dir = work_dir / "handoff"
    handoff_dir.mkdir(exist_ok=True)
    index: list[dict[str, object]] = []
    for item in handoffs:
        if not isinstance(item, dict):
            continue
        stage_idx = int(item.get("stage_idx", len(index)))
        engine = str(item.get("engine", "engine"))
        stage_dir = handoff_dir / f"stage_{stage_idx:02d}_{_safe_name(engine)}"
        stage_dir.mkdir(exist_ok=True)
        copied: dict[str, str] = {}
        for key, filename in (
            ("summary_path", "summary.json"),
            ("best_candidate_path", "best_candidate.txt"),
        ):
            source = item.get(key)
            if isinstance(source, str) and source:
                path = Path(source)
                if path.exists() and path.is_file():
                    shutil.copy2(path, stage_dir / filename)
                    copied[key] = str(stage_dir / filename)
        trace_source = item.get("eval_trace_dir")
        if isinstance(trace_source, str) and trace_source:
            trace_dir = Path(trace_source)
            if trace_dir.exists() and trace_dir.is_dir():
                shutil.copytree(trace_dir, stage_dir / "evals", dirs_exist_ok=True)
                copied["eval_trace_dir"] = str(stage_dir / "evals")
        index.append(
            {
                "stage_idx": stage_idx,
                "engine": engine,
                "best_score": item.get("best_score"),
                "num_evals": item.get("num_evals"),
                "copied": copied,
            }
        )
    (handoff_dir / "index.json").write_text(json.dumps(index, indent=2, default=str) + "\n")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value) or "engine"


class AutoResearchEngine:
    """Black-box research optimizer backed by Claude Code.

    Engine-specific options come from ``OptimizeAnythingConfig.engine_config``,
    parsed into :class:`AutoResearchConfig`.
    """

    name = "autoresearch"

    def __init__(self, config: OptimizeAnythingConfig) -> None:
        engine_config = AutoResearchConfig(**config.engine_config)
        self.model = engine_config.model
        self.ralph = engine_config.ralph
        self.max_no_eval_seconds = engine_config.max_no_eval_seconds
        self.handoffs = engine_config.handoffs
        self.effort = engine_config.effort
        self.max_thinking_tokens = engine_config.max_thinking_tokens
        self.run_dir = config.run_dir
        self.stop_at_score = config.stop_at_score
        self.sandbox = config.sandbox
        # Proposer-cost cap: USD the claude subprocess(es) may spend. Enforced
        # via --max-budget-usd; the eval server never sees proposer spend.
        self.max_token_cost = config.max_token_cost
        self._pending_tempdir: tempfile.TemporaryDirectory[str] | None = None

    def run(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget

        # When sandbox=True, force tempdir work_dir even if run_dir is set.
        # macOS Seatbelt bug: ``denyRead: ["~/"]`` no-ops if ``allowRead``
        # contains any path under ``~/``. A TMPDIR-rooted work_dir keeps
        # allowRead outside home, so the deny rule actually blocks
        # ~/.claude/projects, ~/.ssh, the project source tree, etc.
        # ``process_result`` mirrors artifacts back to self.run_dir so the
        # user doesn't lose them.
        if self.sandbox:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="optimize_anything_cc_")
            work_dir = Path(self._pending_tempdir.name)
        elif self.run_dir:
            work_dir = Path(self.run_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="optimize_anything_cc_")
            work_dir = Path(self._pending_tempdir.name)

        _materialize_sandbox(
            work_dir,
            task,
            server,
            budget,
            max_token_cost=self.max_token_cost,
            perfect_score=self.stop_at_score,
            handoffs=self.handoffs,
        )

        candidate_file = work_dir / "candidate.txt"
        best_file = work_dir / "best_candidate.txt"
        eval_script = work_dir / "eval.sh"

        prompt = (
            f"Read program.md for full task instructions. "
            f"The candidate to improve is in {candidate_file}. "
            f"Write your best candidate to {best_file}. "
            f"Use {eval_script} to evaluate."
        )

        session_id = str(uuid.uuid4())
        env = {**os.environ, "GEPA_OMNI_WORK_DIR": str(work_dir)}
        env.pop("CLAUDECODE", None)
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        if self.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        adapter_cost = 0.0
        ralph_iterations = 1
        invocations: list[dict[str, float | int | None]] = []

        proc = self._run_claude(
            work_dir=work_dir,
            session_id=session_id,
            prompt=prompt,
            budget=budget,
            adapter_cost=adapter_cost,
            resume=False,
            env=env,
        )
        if (
            proc.returncode != 0
            and not budget.exhausted
            and "NO_EVAL_PROGRESS" not in proc.stderr
            and not _saw_budget_exhausted(proc)
        ):
            raise RuntimeError(
                "Claude Code subprocess failed "
                f"(exit {proc.returncode}). "
                f"stdout_tail={_tail_text(proc.stdout)!r} "
                f"stderr_tail={_tail_text(proc.stderr)!r}"
            )
        iter_cost = _extract_claude_cost(proc.stdout)
        adapter_cost += iter_cost
        invocations.append({"cost": iter_cost, "score": server.best_score, "returncode": proc.returncode})

        if self.ralph:
            while ralph_iterations < _RALPH_SAFETY_ITERATION_CAP:
                if _saw_budget_exhausted(proc):
                    break
                if not self._has_budget_headroom(server, adapter_cost):
                    break
                if (
                    self.stop_at_score is not None
                    and server.best_score is not None
                    and server.best_score >= self.stop_at_score
                ):
                    break
                if proc.returncode != 0:
                    break
                proc = self._run_claude(
                    work_dir=work_dir,
                    session_id=session_id,
                    prompt=RALPH_CONTINUE_PROMPT,
                    budget=budget,
                    adapter_cost=adapter_cost,
                    resume=True,
                    env=env,
                )
                iter_cost = _extract_claude_cost(proc.stdout)
                adapter_cost += iter_cost
                invocations.append({"cost": iter_cost, "score": server.best_score, "returncode": proc.returncode})
                if _saw_budget_exhausted(proc):
                    break
                if proc.returncode != 0:
                    break
                ralph_iterations += 1
                # Iteration produced no measurable cost — agent likely
                # has no more progress to make. Stop spending.
                if iter_cost < 0.001:
                    break

        best_candidate = best_file.read_text() if best_file.exists() else seed_as_text(task.seed_candidate)
        best_score = server.best_score
        if task.has_dataset:
            aggregate_best = _best_aggregate_candidate(server)
            if aggregate_best is not None:
                best_candidate, best_score = aggregate_best
            else:
                best_score = float("-inf")

        return Result(
            best_candidate=best_candidate,
            best_score=best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={
                "adapter_cost": adapter_cost,
                "session_id": session_id,
                "work_dir": str(work_dir),
                "ralph_iterations": ralph_iterations,
                "invocations": invocations,
            },
        )

    def _run_claude(
        self,
        *,
        work_dir: Path,
        session_id: str,
        prompt: str,
        budget: BudgetTracker,
        adapter_cost: float,
        resume: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Invoke ``claude --print`` once.

        ``resume=True`` continues the pinned session via ``--resume
        <session_id>``. ``--max-budget-usd`` is set to the *remaining* LLM
        budget so successive Ralph iterations don't stack the cap.

        Monitors the subprocess while it runs and terminates it when:
          * ``max_no_eval_seconds`` has elapsed with no eval activity
            (signals an agent wedged on internal reasoning), or
          * the eval-count budget reported exhausted ``_BUDGET_EXHAUSTION_GRACE_SECONDS``
            ago (giving in-flight 429s time to drain).
        """
        cmd: list[str] = bwrap_prefix(work_dir) if self.sandbox else []
        cmd += [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--model",
            self.model,
        ]
        if resume:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--session-id", session_id])
        cmd.append(DENY_WEB_TOOLS)
        # Single source of the permission posture (see claude_permission_args):
        # bypassPermissions in the bwrap jail / unsandboxed, or the macOS
        # Seatbelt settings whose --permission-mode default enforces the whitelist.
        cmd.extend(claude_permission_args(work_dir, sandboxed=self.sandbox))
        if self.max_thinking_tokens is None and self.effort is not None:
            cmd.extend(["--effort", self.effort])
        if self.max_token_cost is not None:
            remaining = max(0.0, self.max_token_cost - adapter_cost)
            cmd.extend(["--max-budget-usd", f"{remaining:.6f}"])
        cmd.append(prompt)

        proc = subprocess.Popen(
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        exhausted_since: float | None = None
        last_eval_used = budget.used
        last_eval_time = time.monotonic()
        while proc.poll() is None:
            if budget.used != last_eval_used:
                last_eval_used = budget.used
                last_eval_time = time.monotonic()
            if self.max_no_eval_seconds is not None and time.monotonic() - last_eval_time >= self.max_no_eval_seconds:
                proc.terminate()
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                stderr = (
                    (stderr or "")
                    + f"\nNO_EVAL_PROGRESS: terminated Claude Code after {self.max_no_eval_seconds:.1f}s without eval progress.\n"
                )
                return subprocess.CompletedProcess(cmd, proc.returncode, stdout or "", stderr)
            if budget.exhausted:
                if exhausted_since is None:
                    exhausted_since = time.monotonic()
                elif time.monotonic() - exhausted_since >= _BUDGET_EXHAUSTION_GRACE_SECONDS:
                    proc.terminate()
                    try:
                        stdout, stderr = proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate()
                    stderr = (
                        stderr or ""
                    ) + "\nBUDGET_EXHAUSTED: terminated Claude Code after eval budget was exhausted.\n"
                    return subprocess.CompletedProcess(cmd, proc.returncode, stdout or "", stderr)
            time.sleep(1.0)

        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout or "", stderr or "")

    def _has_budget_headroom(self, server: EvalServer, adapter_cost: float) -> bool:
        """Whether starting another Ralph iteration is worthwhile.

        Stops if the eval-count budget is exhausted, or if the LLM budget
        has less than ``_RALPH_MIN_HEADROOM_USD`` left — not worth
        spawning another CLI for pennies.
        """
        budget = server.budget
        if budget.exhausted:
            return False
        if self.max_token_cost is not None:
            if adapter_cost >= self.max_token_cost - _RALPH_MIN_HEADROOM_USD:
                return False
        return True

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        # Prefer ``self.run_dir`` when the caller's server has no output_dir:
        # callers like terrarium's optimize_anything adapter set ``output_dir=None`` on
        # inner servers (to avoid double per-eval-JSON mirroring) but still
        # want engine artifacts (work tree, session transcripts) preserved
        # under the configured ``run_dir`` rather than lost when the tempdir
        # is cleaned up.
        dest = output_dir if output_dir is not None else (Path(self.run_dir) if self.run_dir else None)
        if dest is None:
            if self._pending_tempdir is not None:
                self._pending_tempdir.cleanup()
                self._pending_tempdir = None
            return
        dest.mkdir(parents=True, exist_ok=True)
        work_dir = Path(result.metadata["work_dir"])
        session_id = result.metadata["session_id"]
        copy_session_transcript(work_dir, session_id, dest / "sessions")
        if not _is_under(work_dir, dest):
            shutil.copytree(work_dir, dest / "work", dirs_exist_ok=True)
        if self._pending_tempdir is not None:
            self._pending_tempdir.cleanup()
            self._pending_tempdir = None


def _is_under(child: Path, parent: Path) -> bool:
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except OSError:
        return False


def _tail_text(text: str | None, *, max_chars: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[-max_chars:]


def _saw_budget_exhausted(proc: subprocess.CompletedProcess[str]) -> bool:
    """Return whether Claude observed the eval server's exhausted-budget marker."""
    return _BUDGET_EXHAUSTED_MARKER in (proc.stdout or "") or _BUDGET_EXHAUSTED_MARKER in (proc.stderr or "")


def _config_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "off"}:
            return False
        if normalized in {"true", "1", "yes", "on"}:
            return True
    return bool(value)


def _best_aggregate_candidate(server: EvalServer) -> tuple[str, float] | None:
    """Return the best candidate from aggregate ``evaluate_examples`` calls.

    Dataset ``EvalServer.best_score`` is the best per-example score, not a
    candidate-level aggregate. The HTTP ``/evaluate_examples`` endpoint logs
    aggregate checkpoints with candidate ids; use those for agentic engines
    that evaluate through shell scripts.
    """
    progress = list(getattr(server, "progress_log", []) or [])
    registry = getattr(server, "_candidate_registry", {}) or {}
    candidates_by_id = {cid: candidate for candidate, cid in registry.items()}
    best_entry = None
    for entry in progress:
        candidate_id = entry.get("candidate_id")
        if candidate_id not in candidates_by_id:
            continue
        if best_entry is None or float(entry.get("val_score", float("-inf"))) > float(
            best_entry.get("val_score", float("-inf"))
        ):
            best_entry = entry
    if best_entry is None:
        return None
    return candidates_by_id[best_entry["candidate_id"]], float(best_entry.get("val_score", float("-inf")))


def _extract_claude_cost(stdout: str) -> float:
    stdout = (stdout or "").strip()
    if not stdout:
        return 0.0
    try:
        payload = json.loads(stdout)
        return float(payload.get("total_cost_usd", 0.0) or 0.0)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0
