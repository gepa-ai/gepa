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
from gepa.omni.sandbox import DENY_WEB_TOOLS, bwrap_prefix

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

Run ONE iteration of candidate evolution. Do all work in the main session — do NOT delegate to subagents.

**You do NOT run benchmarks.** You analyze results, prototype changes, and write new candidate files. The outer loop (the meta_harness backend) handles benchmarking.

## CRITICAL CONSTRAINTS

- You MUST implement up to `max_candidates` new candidates every iteration (cap given in the task prompt; aim for 3 unless told otherwise).
- Do NOT abort early. ALWAYS complete all steps including prototyping.
- Design candidates as a mix of exploitation and exploration.

### Anti-parameter-tuning rules

The most common failure mode is creating candidates that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried. Good candidates change a fundamental mechanism: a new algorithmic approach, a new prompt architecture, a new control-flow strategy, or a new representation. Bad candidates just tune numbers.

### Anti-overfitting rules

- **No example-specific hints.** Do not hardcode knowledge about specific test inputs you've seen. Candidates must be general-purpose.
- **Never echo example identifiers** in candidate code, prompts, or comments.

## WORKFLOW

### Step 1: Analyze

Read all state files: `task.md`, `evolution_summary.jsonl`, `frontier.json`, `agents/baseline.txt`, top-scoring `agents/iter*.txt` files, and `state/eval_traces/<candidate_name>/*.json` for failure analysis.

### Step 2: Prototype — MANDATORY

Sketch the core idea in `scratch/` first. Try 2-3 variants and compare before picking the best one. Delete sketches when done.

### Step 3: Implement

Copy a top-performing existing candidate (or `agents/baseline.txt`) as a starting point, then make targeted modifications. Self-critique: re-read the file and verify the candidate is genuinely novel, not a parameter variant.

### Step 4: Write pending_eval.json

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
      "components": ["tag1", "tag2"]
    }
  ]
}
```

A candidate is the **entire text content of a single file** at `agents/iter<N>_<name>.txt`. Use `Write` (not `Edit`) for new candidate files. `agents/baseline.txt` is the seed; treat it as read-only.
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
    cost_usd = _parse_proposer_cost(proc.stdout or "")
    return proc.returncode, cost_usd, session_id


def _parse_proposer_cost(stdout: str) -> float:
    stdout = (stdout or "").strip()
    if not stdout:
        return 0.0
    try:
        payload = json.loads(stdout)
        return float(payload.get("total_cost_usd", 0.0) or 0.0)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


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
    if task.has_dataset and task.train_set:
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
    with open(summary_path, "a") as f:
        f.write(json.dumps(row) + "\n")


_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _log(*parts: str) -> None:
    print(" ".join(parts), flush=True)


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

        if self.run_dir:
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

            _log(f"\n[meta-harness] iteration {iteration}/{cap} (evals={budget.used}, cost=${total_proposer_cost:.3f})")

            pending_path = state_dir / f"pending_eval_iter{iteration}.json"
            if pending_path.exists():
                pending_path.unlink()

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
            total_proposer_cost += cost
            session_ids.append(session_id)

            if exit_code != 0:
                stop_reason = "proposer_failed"

            candidates = _read_pending(pending_path)
            if not candidates:
                _log("  no candidates in pending_eval.json; stopping")
                stop_reason = "no_candidates"
                break

            for ci, c in enumerate(candidates, 1):
                name = c["name"]
                file = c["file"]
                cand_text = _load_candidate(work_dir, file)
                if cand_text is None:
                    continue

                eval_idx_before = len(server.eval_log)
                try:
                    score, _ = _score_candidate(server, task, cand_text)
                except BudgetExhausted:
                    _capture_eval_traces(server, name, (eval_idx_before, len(server.eval_log)), work_dir)
                    stop_reason = "eval_budget_exhausted"
                    break
                except Exception as e:
                    _log(f"    [{ci}/{len(candidates)}] eval error {name}: {e}")
                    continue

                _capture_eval_traces(server, name, (eval_idx_before, len(server.eval_log)), work_dir)
                prev_best = self._best_score(frontier_path)
                _update_frontier(frontier_path, name=name, file=file, score=score)

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
                    outcome=(f"{score:.4f}" if prev_best is None else f"{score:.4f} ({score - prev_best:+.4f})"),
                    budget_used=budget.used,
                )

                if task.has_dataset:
                    try:
                        server.log_progress(score, candidate=cand_text, reflection_cost=total_proposer_cost)
                    except Exception:
                        pass

                if self.stop_at_score is not None and score >= self.stop_at_score:
                    stop_reason = "perfect_score"
                    break

            if stop_reason in ("eval_budget_exhausted", "perfect_score", "proposer_failed"):
                break

        _log(
            f"[meta-harness] done reason={stop_reason} iters={iteration} "
            f"best={self._best_score(frontier_path)} evals={budget.used} "
            f"proposer_cost=${total_proposer_cost:.3f}"
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

    def process_result(self, result: Result, output_dir: Path) -> None:
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
