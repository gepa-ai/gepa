"""Claude Code backend: black-box optimization via a Claude Code subprocess.

The api creates an :class:`EvalServer` and starts its HTTP endpoint. This
backend launches one ``claude --print`` session inside a work_dir laid out
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
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from gepa.omni._helpers import example_to_json, warn_unknown_config_keys
from gepa.omni.backend import Result
from gepa.omni.backends.claude_utils import copy_session_transcript
from gepa.omni.budget import BudgetTracker
from gepa.omni.sandbox import DENY_WEB_TOOLS, bwrap_prefix, claude_settings_args

if TYPE_CHECKING:
    from gepa.omni.config import OmniConfig
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import Task


_CC_CONFIG_KEYS: tuple[str, ...] = ("model",)


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


def _budget_section(budget: BudgetTracker) -> str:
    lines: list[str] = []
    if budget.max_evals is not None:
        lines.append(f"- **Evaluation cap:** {budget.max_evals} calls (each eval = 1 unit).")
    if budget.max_token_cost is not None:
        lines.append(
            f"- **LLM token budget:** ${budget.max_token_cost:.2f} "
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
        "- You cannot modify eval.sh or the server.\n"
        f"- Focus on meaningful improvements each iteration.{exhaust_rule}"
    )


def _build_program_md(task: Task, budget: BudgetTracker, *, perfect_score: float | None) -> str:
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
        candidate_len=len(task.initial_candidate),
        eval_section=eval_section,
        budget_section=_budget_section(budget) + _perfect_score_section(perfect_score),
        strategy_section=_strategy_section(task),
        rules_section=_rules_section(task, budget),
    )


def _materialize_sandbox(
    work_dir: Path,
    task: Task,
    server: EvalServer,
    budget: BudgetTracker,
    *,
    perfect_score: float | None = None,
) -> None:
    server_url = server.url
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "program.md").write_text(_build_program_md(task, budget, perfect_score=perfect_score))
    (work_dir / "candidate.txt").write_text(task.initial_candidate)
    (work_dir / "best_candidate.txt").write_text(task.initial_candidate)

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
                (train_dir / f"{eid}.json").write_text(
                    json.dumps(example_to_json(eid, ex), indent=2, default=str)
                )


class ClaudeCodeBackend:
    """Black-box Claude Code optimizer.

    Backend-specific keys read from ``OmniConfig.config``:

    - ``model``: Claude model id (e.g. ``"sonnet"``, ``"opus"``). Default ``"sonnet"``.
    """

    name = "claude_code"

    def __init__(self, config: OmniConfig) -> None:
        extras = config.config
        warn_unknown_config_keys(self.name, extras, _CC_CONFIG_KEYS)
        self.model: str = extras.get("model", "sonnet")
        self.run_dir = config.run_dir
        self.effort = config.effort
        self.stop_at_score = config.stop_at_score
        self.max_thinking_tokens = config.max_thinking_tokens
        self.sandbox = config.sandbox
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
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="omni_cc_")
            work_dir = Path(self._pending_tempdir.name)
        elif self.run_dir:
            work_dir = Path(self.run_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="omni_cc_")
            work_dir = Path(self._pending_tempdir.name)

        _materialize_sandbox(work_dir, task, server, budget, perfect_score=self.stop_at_score)

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
        cmd: list[str] = bwrap_prefix(work_dir) if self.sandbox else []
        cmd += [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--session-id",
            session_id,
            "--permission-mode",
            "bypassPermissions",
            DENY_WEB_TOOLS,
        ]
        if self.sandbox:
            cmd.extend(claude_settings_args(work_dir))  # macOS Seatbelt fallback
        if self.max_thinking_tokens is None and self.effort is not None:
            cmd.extend(["--effort", self.effort])
        if budget.max_token_cost is not None:
            cmd.extend(["--max-budget-usd", str(budget.max_token_cost)])
        cmd.append(prompt)

        env = {**os.environ, "GEPA_OMNI_WORK_DIR": str(work_dir)}
        env.pop("CLAUDECODE", None)
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        if self.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        proc = subprocess.run(cmd, cwd=str(work_dir), env=env, capture_output=True, text=True)
        adapter_cost = _extract_claude_cost(proc.stdout)

        best_candidate = best_file.read_text() if best_file.exists() else task.initial_candidate

        return Result(
            best_candidate=best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={
                "adapter_cost": adapter_cost,
                "session_id": session_id,
                "work_dir": str(work_dir),
            },
        )

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        if output_dir is None:
            # Caller-owned EvalServer without a configured output_dir: nowhere
            # to copy transcripts/work to. Still clean up our tempdir.
            if self._pending_tempdir is not None:
                self._pending_tempdir.cleanup()
                self._pending_tempdir = None
            return
        work_dir = Path(result.metadata["work_dir"])
        session_id = result.metadata["session_id"]
        copy_session_transcript(work_dir, session_id, output_dir / "sessions")
        if not _is_under(work_dir, output_dir):
            shutil.copytree(work_dir, output_dir / "work", dirs_exist_ok=True)
        if self._pending_tempdir is not None:
            self._pending_tempdir.cleanup()
            self._pending_tempdir = None


def _is_under(child: Path, parent: Path) -> bool:
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except OSError:
        return False


def _extract_claude_cost(stdout: str) -> float:
    stdout = (stdout or "").strip()
    if not stdout:
        return 0.0
    try:
        payload = json.loads(stdout)
        return float(payload.get("total_cost_usd", 0.0) or 0.0)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0
