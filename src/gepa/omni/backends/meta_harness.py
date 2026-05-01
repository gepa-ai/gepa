"""Meta-Harness backend: iterative candidate proposal via Claude Code.

The proposer (a Claude subprocess) reads frontier + history and writes
``pending_eval.json`` listing 1+ candidates; the backend benchmarks each one
through the omni eval server, updates state files, and loops until the
budget is exhausted or ``max_iterations`` is hit.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.omni._helpers import example_to_json, warn_unknown_config_keys
from gepa.omni.backend import Result
from gepa.omni.budget import BudgetExhausted, BudgetTracker
from gepa.omni.sandbox import DENY_WEB_TOOLS, bwrap_prefix, claude_settings_args

if TYPE_CHECKING:
    from gepa.omni.config import OmniConfig
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import Task


_MH_CONFIG_KEYS: tuple[str, ...] = ("model", "max_iterations", "max_candidates_per_iter")


SKILL_MD = """\
---
name: gepa-omni-meta-harness
description: Run one iteration of candidate evolution for a task. Called by the meta_harness backend.
---

# Meta-Harness (Candidate Evolution)

Run ONE iteration of candidate evolution. Do all work in the main session — do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run benchmarks.** You analyze results, prototype changes, and write new candidate files. The outer loop (the meta_harness backend) handles benchmarking separately.

## CRITICAL CONSTRAINTS

- You MUST implement up to `max_candidates` new candidates every iteration (cap given in the task prompt; aim for 3 unless told otherwise).
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design candidates as a mix of exploitation and exploration.

### Anti-parameter-tuning rules

The most common failure mode is creating candidates that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried — parameter sweeps (constants, thresholds, length caps, retry counts) almost always regress or tie.

**Good candidates change a fundamental mechanism:**

- A new algorithmic approach (e.g. a different solver structure, a different decomposition)
- A new prompt architecture (e.g. organize by failure clusters instead of listing rules sequentially)
- A new control-flow strategy (e.g. multi-pass refinement vs. single-shot, conditional branching on input shape)
- A new representation (e.g. tabular vs. narrative, normalized vs. raw)

**Bad candidates just tune numbers.** If the candidate text differs from a previous one only by changing constants, it's a parameter variant. Rewrite with a genuinely novel mechanism.

**Combining ideas is valid.** Take one mechanism from candidate A and another from candidate B, or draw on published approaches.

If the last 3 iterations explored the same axis (prompt wording, retrieval count, etc.), pick a different axis.

### Anti-overfitting rules

- **No example-specific hints.** Do not hardcode knowledge about specific test inputs you've seen. Candidates must be general-purpose.
- **Never echo example identifiers** in candidate code, prompts, or comments.
- **General patterns are OK.** Rules like "favor short outputs when ambiguous" or "validate the parse before submitting" are fine — they apply broadly.

## Budget

Token-cost and eval-count budgets are enforced by the backend, not by you. Don't try to ration evals or refuse to propose because "budget might run out" — if the cap is reached, the backend simply stops spawning new proposer sessions. Your only job is to produce strong candidates this iteration.

## WORKFLOW

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

Check the reports directory (path in the task prompt's "Run directories" section). For each past iteration that has results in `evolution_summary.jsonl` but NO report, write one. Each report should be **<=30 lines** covering: what changed, which candidates improved/regressed and why, and a takeaway for future iterations.

### Step 1: Analyze

1. **Read all state files:**
   - `task.md` — task objective + background + evaluation model
   - `evolution_summary.jsonl` — what's been tried (one JSON per candidate)
   - `frontier.json` — current best candidate and best score
   - `agents/baseline.txt` — the seed candidate
   - top-scoring `agents/iter*.txt` files
   - `state/eval_traces/<candidate_name>/*.json` — per-eval records from the
     eval server for every previously-scored candidate. **This is the most
     important source for failure analysis.** Each JSON contains the full
     `info` dict (compile errors, judge messages, per-example scores for
     dataset tasks, logs, status). Read the traces of the best-and-worst
     candidates to understand *why* they scored as they did before designing
     new candidates.

2. Formulate hypotheses — each must be falsifiable and target a different mechanism.

### Step 2: Prototype — MANDATORY

**You MUST prototype your mechanism before writing the final candidate.** Do NOT skip this step. Candidates that skip prototyping tend to have bugs or produce no improvement.

For each candidate:

1. Write a sketch in `scratch/` (inside the run's work dir) that exercises the core idea in isolation (run code candidates manually; for prompt candidates, at least re-read and self-critique).
2. Try 2-3 variants and compare before picking the best one.
3. Delete sketches when done.

### Step 3: Implement

For each candidate:

1. Copy a top-performing existing candidate (or `agents/baseline.txt`) as a starting point, then make targeted modifications. Copy-then-edit ensures correct formatting and proven structure.
2. Implement the new mechanism according to your hypothesis.
3. **Self-critique (mandatory):** After writing, re-read the file and check: does this candidate introduce a genuinely NEW mechanism, or is it just a parameter variant? If only constants differ from the base, REWRITE with a truly novel mechanism.

The benchmark auto-discovers files in `agents/` — you don't need to register candidates anywhere else.

### Step 4: Write pending_eval.json

Write to the path specified in the task prompt (NOT hardcoded — it may be in a run-specific subdirectory):

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<snake_case_name>",
      "file": "agents/iter<N>_<name>.txt",
      "hypothesis": "<falsifiable claim>",
      "axis": "exploitation|exploration",
      "base": "<what it builds on>",
      "components": ["tag1", "tag2", "..."]
    }
  ]
}
```

Output: `CANDIDATES: <name1>, <name2>, <name3>`

## Candidate format

A candidate is the **entire text content of a single file** at `agents/iter<N>_<name>.txt`. Whatever you write there — code, a prompt, JSON, whatever the task expects — is what the eval server scores. Read `task.md` to learn what shape the task expects.

- Use `Write` (not `Edit`) to create new candidate files.
- `agents/baseline.txt` is the seed; treat it as read-only.
- Don't write outside `agents/` or the pending_eval.json path.

## evolution_summary.jsonl Format

One JSON object per line, one line per evaluated candidate:

```json
{"iteration": 1, "name": "example_candidate", "score": 45.0, "axis": "exploitation", "hypothesis": "...", "delta": +2.1, "outcome": "45.00 (+2.1)", "components": ["tag1", "tag2"]}
```

## Component Analysis

Treat `evolution_summary.jsonl`, `frontier.json`, and any prior `agents/iter*.txt` files as the only shipped history sources.
"""


def _build_task_md(task: Task) -> str:
    optional = ""
    if task.objective:
        optional += f"## Objective\n{task.objective}\n\n"
    if task.background:
        optional += f"## Background\n{task.background}\n\n"
    if task.has_dataset and task.train_set:
        val_note = (
            f" (plus a hidden val set of {len(task.val_set)} examples; not visible to the proposer)"
            if task.val_set
            else ""
        )
        eval_section = (
            f"This is a **dataset / generalization** task with {len(task.train_set)} training examples"
            f"{val_note}. The outer loop scores each candidate on the full training split; "
            f"each example costs 1 unit of the evaluation budget."
        )
    else:
        eval_section = (
            "This is a **single-task** optimization. The outer loop scores each candidate "
            "once; one scored candidate costs 1 unit of the evaluation budget."
        )
    return f"# Task: {task.name}\n\n{optional}## Evaluation model\n\n{eval_section}\n"


def _materialize_sandbox(work_dir: Path, task: Task, server: EvalServer, budget: BudgetTracker) -> None:
    del budget
    work_dir.mkdir(parents=True, exist_ok=True)

    (work_dir / "task.md").write_text(_build_task_md(task))

    skill_dir = work_dir / ".claude" / "skills" / "gepa-omni-meta-harness"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD)

    agents_dir = work_dir / "agents"
    agents_dir.mkdir(exist_ok=True)
    (agents_dir / "baseline.txt").write_text(task.initial_candidate)

    (work_dir / "scratch").mkdir(exist_ok=True)
    state_dir = work_dir / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "reports").mkdir(exist_ok=True)
    (state_dir / "eval_traces").mkdir(exist_ok=True)

    frontier = state_dir / "frontier.json"
    if not frontier.exists():
        frontier.write_text(
            json.dumps(
                {
                    "best_name": "baseline",
                    "best_score": None,
                    "best_candidate_file": "agents/baseline.txt",
                },
                indent=2,
            )
        )

    summary = state_dir / "evolution_summary.jsonl"
    if not summary.exists():
        summary.write_text("")

    if task.train_set:
        train_dir = work_dir / "train"
        train_dir.mkdir(exist_ok=True)
        for eid, ex in server.iter_split("train"):
            (train_dir / f"{eid}.json").write_text(json.dumps(example_to_json(eid, ex), indent=2, default=str))


def _run_proposer(
    *,
    work_dir: Path,
    iteration: int,
    model: str,
    effort: str | None,
    max_candidates: int,
    max_budget_usd: float | None,
    pending_path: Path,
    log_dir: Path,
    max_thinking_tokens: int | None = None,
    sandbox: bool = True,
) -> tuple[int, float, str]:
    state = work_dir / "state"
    prompt = (
        f"Run iteration {iteration} of the meta-harness evolution loop. "
        f"Produce up to {max_candidates} candidate(s).\n\n"
        f"## Run directories (absolute paths)\n"
        f"- task.md: `{work_dir / 'task.md'}`\n"
        f"- state/frontier.json: `{state / 'frontier.json'}`\n"
        f"- state/evolution_summary.jsonl: `{state / 'evolution_summary.jsonl'}`\n"
        f"- state/eval_traces/<candidate_name>/: `{state / 'eval_traces'}`\n"
        f"- state/reports/: `{state / 'reports'}`\n"
        f"- agents/: `{work_dir / 'agents'}`\n"
        f"- Write pending_eval.json to: `{pending_path}`\n\n"
        f"Follow the gepa-omni-meta-harness skill in `.claude/skills/`."
    )

    session_id = str(uuid.uuid4())
    cmd: list[str] = bwrap_prefix(work_dir) if sandbox else []
    cmd += [
        "claude",
        "--print",
        prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--session-id",
        session_id,
        "--permission-mode",
        "bypassPermissions",
        DENY_WEB_TOOLS,
    ]
    if sandbox:
        cmd.extend(claude_settings_args(work_dir))  # macOS Seatbelt fallback
    if max_thinking_tokens is None and effort is not None:
        cmd.extend(["--effort", effort])
    if max_budget_usd is not None:
        cmd.extend(["--max-budget-usd", f"{max_budget_usd:.4f}"])

    env = {**os.environ, "GEPA_OMNI_WORK_DIR": str(work_dir)}
    env.pop("CLAUDECODE", None)
    env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
    if max_thinking_tokens is not None:
        env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
        env["MAX_THINKING_TOKENS"] = str(max_thinking_tokens)

    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(work_dir), env=env, capture_output=True, text=True)
    (log_dir / f"iter{iteration}_stdout.json").write_text(proc.stdout or "")
    (log_dir / f"iter{iteration}_stderr.txt").write_text(proc.stderr or "")
    cost_usd, result_payload = _parse_proposer_result(proc.stdout or "")

    (log_dir / f"iter{iteration}_meta.json").write_text(
        json.dumps(
            {
                "iteration": iteration,
                "session_id": session_id,
                "exit_code": proc.returncode,
                "cost_usd": cost_usd,
                "cmd": cmd,
                "stderr_tail": (proc.stderr or "")[-2000:],
                # Echo a few top-level result fields so a reviewer doesn't have
                # to jump into stdout.json.
                "result_summary": {
                    "duration_ms": result_payload.get("duration_ms"),
                    "num_turns": result_payload.get("num_turns"),
                    "usage": result_payload.get("usage"),
                    "is_error": result_payload.get("is_error"),
                    "subtype": result_payload.get("subtype"),
                },
            },
            indent=2,
        )
    )
    return proc.returncode, cost_usd, session_id


def _parse_proposer_result(stdout: str) -> tuple[float, dict[str, Any]]:
    """Extract (cost_usd, result_payload) from ``claude --output-format json``.

    The CLI prints exactly one JSON object to stdout once the session ends:

        {
          "type": "result",
          "subtype": "success",
          "is_error": false,
          "session_id": "...",
          "total_cost_usd": 0.1234,
          "duration_ms": 12345,
          "num_turns": 7,
          "usage": {"input_tokens": ..., "output_tokens": ...},
          "result": "<final assistant text>"
        }

    A missing / malformed payload indicates the CLI crashed before the end of
    the session. Cost is reported as 0 in that case; the caller already
    surfaces the subprocess exit code + stderr tail in the iteration meta
    file, so the user has enough to diagnose.
    """
    payload: dict[str, Any] = {}
    stdout = (stdout or "").strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            payload = {}

    try:
        cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return cost, payload


def _read_pending(pending_path: Path) -> list[dict[str, Any]]:
    if not pending_path.exists():
        return []
    try:
        data = json.loads(pending_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    candidates = data.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    return [c for c in candidates if isinstance(c, dict) and "file" in c and "name" in c]


def _load_candidate(work_dir: Path, relpath: str) -> str | None:
    path = (work_dir / relpath).resolve()
    try:
        path.relative_to(work_dir.resolve())
    except ValueError:
        return None
    if not path.exists() or path.is_dir():
        return None
    try:
        return path.read_text()
    except OSError:
        return None


def _score_candidate(server: EvalServer, task: Task, candidate: str) -> tuple[float, dict[str, Any]]:
    """Score a candidate via the eval server. Dataset tasks prefer val split
    (matches the other backends' search-loop convention); fall back to train
    if no val_set; single-task → evaluate()."""
    if task.has_dataset:
        if task.val_set:
            return server.evaluate_examples(candidate, split="val")
        if task.train_set:
            return server.evaluate_examples(candidate, split="train")
    return server.evaluate(candidate)


def _capture_eval_traces(
    server: EvalServer,
    candidate_name: str,
    eval_idx_range: tuple[int, int],
    work_dir: Path,
) -> int:
    output_dir = getattr(server, "output_dir", None)
    if output_dir is None:
        return 0
    src_dir = Path(output_dir) / "evals"
    if not src_dir.exists():
        return 0
    before, after = eval_idx_range
    if after <= before:
        return 0
    dst_dir = work_dir / "state" / "eval_traces" / candidate_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for idx in range(before, after):
        src = src_dir / f"{idx}.json"
        if not src.exists():
            continue
        try:
            shutil.copy2(src, dst_dir / f"{idx}.json")
            n += 1
        except OSError:
            pass
    return n


def _update_frontier(frontier_path: Path, name: str, file: str, score: float) -> bool:
    frontier: dict[str, Any] = {}
    if frontier_path.exists():
        try:
            frontier = json.loads(frontier_path.read_text())
        except (json.JSONDecodeError, OSError):
            frontier = {}
    prev = frontier.get("best_score")
    improved = prev is None or score > float(prev)
    if improved:
        frontier["best_name"] = name
        frontier["best_candidate_file"] = file
        frontier["best_score"] = score
        frontier_path.write_text(json.dumps(frontier, indent=2))
    return improved


def _append_summary(
    summary_path: Path,
    *,
    iteration: int,
    name: str,
    file: str,
    score: float | None,
    best_score: float | None,
    hypothesis: str,
    axis: str,
    components: list[str],
    outcome: str,
    budget_used: int,
    propose_time: float | None = None,
    bench_time: float | None = None,
) -> None:
    row: dict[str, Any] = {
        "iteration": iteration,
        "name": name,
        "file": file,
        "score": score,
        "delta": (score - best_score) if (score is not None and best_score is not None) else None,
        "hypothesis": hypothesis,
        "axis": axis,
        "components": components,
        "outcome": outcome,
        "budget_used": budget_used,
    }
    if propose_time is not None:
        row["propose_s"] = round(propose_time, 1)
    if bench_time is not None:
        row["bench_s"] = round(bench_time, 1)
    with open(summary_path, "a") as f:
        f.write(json.dumps(row) + "\n")


_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _log(*parts: str) -> None:
    print(" ".join(parts), flush=True)


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _colorize_score(score: float) -> str:
    s = f"{score:.4f}"
    if score > 0:
        return _green(s)
    if score < 0:
        return _red(s)
    return s


class MetaHarnessBackend:
    """Iterative meta-harness optimizer.

    Backend-specific keys read from ``OmniConfig.config``:

    - ``model``: Proposer model. Default ``"opus"``.
    - ``max_iterations``: Hard cap on proposer sessions. ``None`` = until budget.
        Default ``10``.
    - ``max_candidates_per_iter``: Upper bound on candidates per iteration.
        Default ``3``.
    """

    name = "meta_harness"

    def __init__(self, config: OmniConfig) -> None:
        extras = config.config
        warn_unknown_config_keys(self.name, extras, _MH_CONFIG_KEYS)
        self.model: str = extras.get("model", "opus")
        self.max_iterations: int | None = extras.get("max_iterations", 10)
        self.max_candidates_per_iter: int = max(1, int(extras.get("max_candidates_per_iter", 3)))
        self.run_dir = config.run_dir
        self.effort = config.effort
        self.stop_at_score = config.stop_at_score
        self.max_thinking_tokens = config.max_thinking_tokens
        self.sandbox = config.sandbox
        self._pending_tempdir: tempfile.TemporaryDirectory[str] | None = None

    def run(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget

        # When sandbox=True, force tempdir work_dir even if run_dir is set —
        # avoids Claude Code's CLAUDE.md walk-up disclosing the auto-memory
        # pointer, and keeps Seatbelt's allowRead outside ~/. process_result
        # mirrors back to output_dir/work/ so artifacts aren't lost.
        if self.sandbox:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="omni_mh_")
            work_dir = Path(self._pending_tempdir.name)
        elif self.run_dir:
            work_dir = Path(self.run_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="omni_mh_")
            work_dir = Path(self._pending_tempdir.name)

        _materialize_sandbox(work_dir, task, server, budget)

        state_dir = work_dir / "state"
        frontier_path = state_dir / "frontier.json"
        summary_path = state_dir / "evolution_summary.jsonl"
        sessions_dir = work_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        session_ids: list[str] = []
        total_proposer_cost = 0.0
        cap = self.max_iterations if self.max_iterations is not None else 50

        run_start = time.time()
        iteration = 0
        stop_reason = "completed"
        while iteration < cap:
            iteration += 1
            if budget.exhausted:
                stop_reason = "eval_budget_exhausted"
                break

            per_session_cap: float | None = None
            if budget.max_token_cost is not None:
                remaining = budget.max_token_cost - total_proposer_cost
                if remaining <= 0:
                    stop_reason = "token_budget_exhausted"
                    break
                per_session_cap = remaining

            _log(
                f"\n{_bold(f'[meta-harness] iteration {iteration}/{cap}')} "
                f"{_dim(f'(evals used={budget.used}, cost=${total_proposer_cost:.3f})')}"
            )

            pending_path = state_dir / f"pending_eval_iter{iteration}.json"
            if pending_path.exists():
                pending_path.unlink()

            propose_start = time.time()
            exit_code, cost, session_id = _run_proposer(
                work_dir=work_dir,
                iteration=iteration,
                model=self.model,
                effort=self.effort,
                max_candidates=self.max_candidates_per_iter,
                max_budget_usd=per_session_cap,
                pending_path=pending_path,
                log_dir=sessions_dir,
                max_thinking_tokens=self.max_thinking_tokens,
                sandbox=bool(self.sandbox),
            )
            propose_time = time.time() - propose_start
            total_proposer_cost += cost
            session_ids.append(session_id)

            _log(
                f"  {_cyan('proposer')} exit={exit_code} cost=${cost:.3f} "
                f"time={_elapsed(propose_time)} session={session_id[:8]}"
            )

            if exit_code != 0:
                _append_summary(
                    summary_path,
                    iteration=iteration,
                    name="(proposer_failed)",
                    file="",
                    score=None,
                    best_score=self._best_score(frontier_path),
                    hypothesis="",
                    axis="",
                    components=[],
                    outcome=f"proposer_exit_{exit_code}",
                    budget_used=budget.used,
                    propose_time=propose_time,
                )
                stop_reason = "proposer_failed"

            candidates = _read_pending(pending_path)
            if not candidates:
                _log(f"  {_yellow('no candidates')} in pending_eval.json; stopping")
                _append_summary(
                    summary_path,
                    iteration=iteration,
                    name="(no_candidates)",
                    file="",
                    score=None,
                    best_score=self._best_score(frontier_path),
                    hypothesis="",
                    axis="",
                    components=[],
                    outcome="no_pending_eval",
                    budget_used=budget.used,
                    propose_time=propose_time,
                )
                stop_reason = "no_candidates"
                break

            _log(f"  {_cyan('benchmarking')} {len(candidates)} candidate(s)...")

            bench_start = time.time()
            iter_improved = False
            for ci, c in enumerate(candidates, 1):
                name = c["name"]
                file = c["file"]
                cand_text = _load_candidate(work_dir, file)
                if cand_text is None:
                    _log(f"    [{ci}/{len(candidates)}] {_red('missing')} {name} ({file})")
                    _append_summary(
                        summary_path,
                        iteration=iteration,
                        name=name,
                        file=file,
                        score=None,
                        best_score=self._best_score(frontier_path),
                        hypothesis=c.get("hypothesis", ""),
                        axis=c.get("axis", ""),
                        components=c.get("components", []),
                        outcome="file_missing",
                        budget_used=budget.used,
                    )
                    continue

                eval_idx_before = len(server.eval_log)
                try:
                    score, _info = _score_candidate(server, task, cand_text)
                except BudgetExhausted:
                    _capture_eval_traces(server, name, (eval_idx_before, len(server.eval_log)), work_dir)
                    _log(f"    [{ci}/{len(candidates)}] {_red('budget exhausted')} during {name}")
                    _append_summary(
                        summary_path,
                        iteration=iteration,
                        name=name,
                        file=file,
                        score=None,
                        best_score=self._best_score(frontier_path),
                        hypothesis=c.get("hypothesis", ""),
                        axis=c.get("axis", ""),
                        components=c.get("components", []),
                        outcome="budget_exhausted",
                        budget_used=budget.used,
                    )
                    stop_reason = "eval_budget_exhausted"
                    break
                except Exception as e:
                    _log(f"    [{ci}/{len(candidates)}] {_red('eval error')} {name}: {e}")
                    _append_summary(
                        summary_path,
                        iteration=iteration,
                        name=name,
                        file=file,
                        score=None,
                        best_score=self._best_score(frontier_path),
                        hypothesis=c.get("hypothesis", ""),
                        axis=c.get("axis", ""),
                        components=c.get("components", []),
                        outcome=f"eval_error: {type(e).__name__}",
                        budget_used=budget.used,
                    )
                    continue

                trace_count = _capture_eval_traces(
                    server, name, (eval_idx_before, len(server.eval_log)), work_dir
                )

                prev_best = self._best_score(frontier_path)
                improved = _update_frontier(frontier_path, name=name, file=file, score=score)
                if improved:
                    iter_improved = True

                delta_str = "(new_best)" if improved else (
                    f"({score - prev_best:+.4f})" if prev_best is not None else ""
                )
                trace_str = f" traces={trace_count}" if trace_count else ""
                _log(
                    f"    [{ci}/{len(candidates)}] {_bold(name)}: "
                    f"{_colorize_score(score)} {delta_str} "
                    f"{_dim(f'(budget_used={budget.used}{trace_str})')}"
                )

                _append_summary(
                    summary_path,
                    iteration=iteration,
                    name=name,
                    file=file,
                    score=score,
                    best_score=prev_best,
                    hypothesis=c.get("hypothesis", ""),
                    axis=c.get("axis", ""),
                    components=c.get("components", []),
                    outcome=(
                        f"{score:.4f}" if prev_best is None
                        else f"{score:.4f} ({score - prev_best:+.4f})"
                    ),
                    budget_used=budget.used,
                    propose_time=propose_time if ci == 1 else None,
                )

                if task.has_dataset:
                    try:
                        server.log_progress(
                            score,
                            candidate=cand_text,
                            reflection_cost=total_proposer_cost,
                        )
                    except Exception:
                        pass

                if self.stop_at_score is not None and score >= self.stop_at_score:
                    _log(f"    {_green('perfect score reached')}: {score:.4f} >= {self.stop_at_score}")
                    stop_reason = "perfect_score"
                    break

                tracker = getattr(server, "tracker", None)
                if tracker is not None:
                    try:
                        tracker.log_metrics(
                            {
                                "meta_harness/iteration": iteration,
                                "meta_harness/candidate_score": score,
                                "meta_harness/best_score": (
                                    self._best_score(frontier_path) or 0.0
                                ),
                                "meta_harness/proposer_cost": total_proposer_cost,
                            },
                            step=budget.used,
                        )
                    except Exception:
                        pass

            bench_time = time.time() - bench_start

            wall = _elapsed(time.time() - run_start)
            tail = (
                f"iter {iteration} wall={wall} "
                f"propose={_elapsed(propose_time)} bench={_elapsed(bench_time)}"
            )
            improved_tag = f" {_green('IMPROVED')}" if iter_improved else ""
            _log(f"  {_dim(tail)}{improved_tag}")

            if stop_reason in ("eval_budget_exhausted", "perfect_score", "proposer_failed"):
                break

        _log(
            f"\n{_bold('[meta-harness] done')} "
            f"reason={stop_reason} iters={iteration} "
            f"best={self._best_score(frontier_path)} "
            f"evals={budget.used} proposer_cost=${total_proposer_cost:.3f}"
        )

        best_candidate = server.best_candidate
        best_score = self._best_score(frontier_path)
        frontier = self._read_frontier(frontier_path)
        best_file = frontier.get("best_candidate_file") if frontier else None
        if best_file:
            loaded = _load_candidate(work_dir, best_file)
            if loaded is not None:
                best_candidate = loaded
        if best_score is None:
            best_score = server.best_score

        return Result(
            best_candidate=best_candidate,
            best_score=best_score if best_score is not None else server.best_score,
            total_evals=budget.used,
            eval_log=server.eval_log,
            metadata={
                "adapter_cost": total_proposer_cost,
                "meta_harness": {
                    "iterations_run": iteration,
                    "stop_reason": stop_reason,
                    "proposer_cost": total_proposer_cost,
                    "session_ids": session_ids,
                    "work_dir": str(work_dir),
                    "frontier": self._read_frontier(frontier_path),
                },
                "work_dir": str(work_dir),
                "session_ids": session_ids,
                "wall_time": time.time() - run_start,
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
        work_dir = Path(result.metadata.get("work_dir", ""))
        session_ids: list[str] = result.metadata.get("session_ids", []) or []
        transcripts_dir = output_dir / "sessions"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        for sid in session_ids:
            _copy_session_transcript(work_dir, sid, transcripts_dir)
        if work_dir.exists() and not _is_under(work_dir, output_dir):
            shutil.copytree(work_dir, output_dir / "work", dirs_exist_ok=True)
        if self._pending_tempdir is not None:
            self._pending_tempdir.cleanup()
            self._pending_tempdir = None

    def _best_score(self, frontier_path: Path) -> float | None:
        frontier = self._read_frontier(frontier_path)
        val = frontier.get("best_score") if frontier else None
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _read_frontier(self, frontier_path: Path) -> dict[str, Any]:
        if not frontier_path.exists():
            return {}
        try:
            return json.loads(frontier_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}


_SLUG_RE = re.compile(r"[^A-Za-z0-9-]")


def _claude_project_slug(cwd: Path) -> str:
    return _SLUG_RE.sub("-", str(cwd.resolve()))


def _copy_session_transcript(cwd: Path, session_id: str, dst_dir: Path) -> None:
    src = Path.home() / ".claude" / "projects" / _claude_project_slug(cwd) / f"{session_id}.jsonl"
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst_dir / src.name)
    except OSError:
        pass


def _is_under(child: Path, parent: Path) -> bool:
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except OSError:
        return False
