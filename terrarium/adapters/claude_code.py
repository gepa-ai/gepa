"""Claude Code adapter: runs Claude Code as a black-box evolution system.

Architecture:
    terrarium process                   Claude Code subprocess
    ┌─────────────────────┐             ┌──────────────────────┐
    │ EvalServer (HTTP)   │◄── POST ────│ calls eval.sh        │
    │ - enforces budget   │── score ──► │ - reads score        │
    │ - tracks best       │             │ - mutates candidate  │
    └─────────────────────┘             └──────────────────────┘

The runner creates the EvalServer. This adapter just uses its HTTP endpoint.
Budget is enforced server-side. Claude Code cannot modify it.
When budget is exhausted, the server returns HTTP 429.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer

# Unified eval script: supports full-split eval, specific example IDs, or single-task.
# Usage:
#   ./eval.sh <candidate_file>                     → eval on train split (default)
#   ./eval.sh <candidate_file> test                → eval on test split
#   ./eval.sh <candidate_file> --ids id1,id2,id3   → eval on specific examples
EVAL_SCRIPT = """\
#!/usr/bin/env bash
# Usage: ./eval.sh <candidate_file> [split]
#        ./eval.sh <candidate_file> --ids id1,id2,id3
# Evaluates a candidate on the terrarium eval server.
# Without --ids: evaluates across ALL examples in the split (default: train).
# With --ids: evaluates only the specified examples (comma-separated).
# Examples run in parallel server-side.
# Exit code 1 if budget exhausted.
set -euo pipefail

CANDIDATE_FILE="$1"
shift
SERVER_URL="{server_url}"

CANDIDATE=$(cat "$CANDIDATE_FILE")

# Parse args: either --ids or a split name
IDS=""
SPLIT="train"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ids) IDS="$2"; shift 2 ;;
        *) SPLIT="$1"; shift ;;
    esac
done

if [ -n "$IDS" ]; then
    # Convert comma-separated IDs to JSON array
    IDS_JSON=$(echo "$IDS" | jq -R 'split(",")')
    BODY=$(jq -n --arg c "$CANDIDATE" --argjson ids "$IDS_JSON" '{{candidate: $c, example_ids: $ids}}')
else
    BODY=$(jq -n --arg c "$CANDIDATE" --arg s "$SPLIT" '{{candidate: $c, split: $s}}')
fi

RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/evaluate_examples" \\
    -H "Content-Type: application/json" \\
    -d "$BODY")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

echo "$BODY"

if [ "$HTTP_CODE" = "429" ]; then
    echo "BUDGET_EXHAUSTED" >&2
    exit 1
fi
"""

# System prompt given to Claude Code describing the task and eval protocol.
CLAUDE_CODE_SYSTEM = """\
You are an AI research system tasked with evolving and improving a candidate solution.

## Task
{task_description}

## Initial Candidate
The initial candidate is in: {candidate_file}

## How to Evaluate

`{eval_script} <candidate_file> [split | --ids id1,id2,id3]`

One script, flexible usage:

- **Full split** (default): `{eval_script} candidate.txt` or `{eval_script} candidate.txt train`
  Evaluates across ALL examples in the split. Examples run in parallel server-side.
  Costs N budget units (one per example).

- **Specific examples**: `{eval_script} candidate.txt --ids example_1,example_5,example_12`
  Evaluates only the listed examples. Costs 1 budget unit per example listed.
  Useful for targeted debugging after seeing which examples score low.

- **Single-task** (no dataset): `{eval_script} candidate.txt`
  Just evaluates the candidate. Costs 1 budget unit.

Response fields:
- `average_score`: Mean score across evaluated examples.
- `scores`: Per-example scores (dict of example_id → score).
- `num_evaluated` / `num_total`: How many examples were evaluated vs total in split.
- `partial`: True if budget ran out mid-evaluation.
- `errors`: Per-example errors, if any.
- `budget`: Remaining eval budget.

{dataset_info}

## Budget
You have a maximum of {max_evals} individual example evaluations.
Each example = 1 budget unit. A full train split of N examples costs N units.
If budget runs out mid-evaluation, you get partial results.

## Strategy Tips
- Start with a full eval to see overall performance and identify weak examples.
- Use `--ids` to cheaply iterate on specific failing examples.
- Once you've improved on the weak examples, do another full eval to confirm.
- For single-task problems, each eval call costs exactly 1 budget unit.

## Goal
Iteratively improve the candidate to maximize the score. When you're done
(or budget is exhausted), write your best candidate to: {best_file}

## Rules
- Each example evaluation counts as one budget unit.
- You cannot modify the eval script or the server.
- Focus on making meaningful improvements each iteration.
"""


class ClaudeCodeAdapter:
    """Adapter that runs Claude Code as a black-box evolution subprocess.

    Uses the same EvalServer that the runner creates — just its HTTP endpoint.
    """

    def __init__(
        self,
        model: str = "sonnet",
        max_turns: int | None = None,
    ) -> None:
        self.model = model
        self.max_turns = max_turns

    def evolve(self, task: Task, server: EvalServer, max_evals: int) -> Result:
        with tempfile.TemporaryDirectory(prefix="terrarium_cc_") as work_dir:
            work = Path(work_dir)

            # Write initial candidate
            candidate_file = work / "candidate.txt"
            candidate_file.write_text(task.initial_candidate)

            best_file = work / "best_candidate.txt"
            best_file.write_text(task.initial_candidate)

            # Write eval script
            eval_script = work / "eval.sh"
            eval_script.write_text(EVAL_SCRIPT.format(server_url=server.url))
            eval_script.chmod(0o755)

            # Build dataset info section for the prompt
            if task.has_dataset:
                example_ids = []
                if task.train_set:
                    example_ids = [ex.id for ex in task.train_set]
                lines = [f"## Dataset\nThis is a dataset task with {len(example_ids)} training examples."]
                if example_ids:
                    lines.append(f"Example IDs: {', '.join(example_ids[:20])}")
                    if len(example_ids) > 20:
                        lines.append(f"... and {len(example_ids) - 20} more.")
                if task.test_set:
                    lines.append(f"Test set: {len(task.test_set)} examples (use split='test' to evaluate).")
                dataset_info = "\n".join(lines)
            else:
                dataset_info = "## Task Type\nThis is a single-task problem (no dataset). Each eval call costs 1 budget unit."

            # Build the prompt for Claude Code
            prompt = CLAUDE_CODE_SYSTEM.format(
                task_description=task.description,
                candidate_file=str(candidate_file),
                eval_script=str(eval_script),
                max_evals=max_evals,
                best_file=str(best_file),
                dataset_info=dataset_info,
            )

            # Launch Claude Code
            cmd = ["claude", "--print", "--model", self.model, "--prompt", prompt]
            if self.max_turns:
                cmd.extend(["--max-turns", str(self.max_turns)])

            env = {**os.environ, "TERRARIUM_WORK_DIR": str(work)}

            try:
                subprocess.run(
                    cmd,
                    cwd=str(work),
                    env=env,
                    timeout=3600,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                pass

            # Read the best candidate
            best_candidate = best_file.read_text() if best_file.exists() else task.initial_candidate

            return Result(
                best_candidate=best_candidate,
                best_score=server.best_score,
                total_evals=server.budget.used,
                eval_log=server.eval_log,
            )


def create_adapter(**kwargs: Any) -> ClaudeCodeAdapter:
    """Factory for CLI usage."""
    return ClaudeCodeAdapter(**kwargs)
