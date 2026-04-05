"""Claude Code adapter: runs Claude Code as a black-box evolution system.

Architecture:
    terrarium process                   Claude Code subprocess
    ┌─────────────────────┐             ┌──────────────────────┐
    │ EvalServer (HTTP)   │◄── POST ────│ calls eval_client.sh │
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

# Shell script that Claude Code uses to evaluate candidates.
# It POSTs the candidate to the terrarium eval server and prints the result.
EVAL_CLIENT_SCRIPT = """\
#!/usr/bin/env bash
# Usage: ./eval.sh <candidate_file> [example_id]
# Evaluates a candidate via the terrarium eval server.
# Exit code 1 if budget exhausted.
set -euo pipefail

CANDIDATE_FILE="$1"
EXAMPLE_ID="${{2:-}}"
SERVER_URL="{server_url}"

CANDIDATE=$(cat "$CANDIDATE_FILE")

BODY=$(jq -n --arg c "$CANDIDATE" --arg e "$EXAMPLE_ID" '{{candidate: $c}} + (if $e != "" then {{example_id: $e}} else {{}} end)')

RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/evaluate" \\
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
Run: `{eval_script} <candidate_file>`
This prints a JSON response with "score" and "info" fields.
If the budget is exhausted, it exits with code 1 and prints BUDGET_EXHAUSTED to stderr.

## Budget
You have a maximum of {max_evals} evaluations. Use them wisely.

## Goal
Iteratively improve the candidate to maximize the score. When you're done
(or budget is exhausted), write your best candidate to: {best_file}

## Rules
- Each call to the eval script counts as one evaluation.
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

            # Write eval client script
            eval_script = work / "eval.sh"
            eval_script.write_text(EVAL_CLIENT_SCRIPT.format(server_url=server.url))
            eval_script.chmod(0o755)

            # Build the prompt for Claude Code
            prompt = CLAUDE_CODE_SYSTEM.format(
                task_description=task.description,
                candidate_file=str(candidate_file),
                eval_script=str(eval_script),
                max_evals=max_evals,
                best_file=str(best_file),
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
