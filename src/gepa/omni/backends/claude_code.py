"""Claude Code backend: black-box optimization via a Claude Code subprocess.

The api creates an :class:`EvalServer` and starts its HTTP endpoint. This
backend launches one ``claude --print`` session inside a work_dir laid out
with ``program.md``, ``candidate.txt``, ``best_candidate.txt``, and
``eval.sh`` / ``validate.sh`` scripts that POST to the eval server. Budget
enforcement happens server-side (HTTP 429 on exhaustion); LLM cost is bounded
via ``--max-budget-usd``.
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
from gepa.omni.sandbox import DENY_WEB_TOOLS, bwrap_prefix

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
BODY=$(echo "$RESPONSE" | head -n -1)
echo "$BODY"
if [ "$HTTP_CODE" = "429" ]; then echo "BUDGET_EXHAUSTED" >&2; exit 1; fi
"""

EVAL_SCRIPT_DATASET = """\
#!/usr/bin/env bash
# Usage: ./eval.sh <candidate_file> [split]
#        ./eval.sh <candidate_file> --ids id1,id2,id3
set -euo pipefail
CANDIDATE_FILE="$1"; shift
SERVER_URL="{server_url}"
CANDIDATE=$(cat "$CANDIDATE_FILE")
IDS=""; SPLIT="train"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ids) IDS="$2"; shift 2 ;;
        *) SPLIT="$1"; shift ;;
    esac
done
if [ -n "$IDS" ]; then
    IDS_JSON=$(echo "$IDS" | jq -R 'split(",")')
    BODY=$(jq -n --arg c "$CANDIDATE" --argjson ids "$IDS_JSON" '{{candidate: $c, example_ids: $ids}}')
else
    BODY=$(jq -n --arg c "$CANDIDATE" --arg s "$SPLIT" '{{candidate: $c, split: $s}}')
fi
RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/evaluate_examples" \\
    -H "Content-Type: application/json" -d "$BODY")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
echo "$BODY"
if [ "$HTTP_CODE" = "429" ]; then echo "BUDGET_EXHAUSTED" >&2; exit 1; fi
"""

VALIDATE_SCRIPT = """\
#!/usr/bin/env bash
# Usage: ./validate.sh <candidate_file>
set -euo pipefail
CANDIDATE_FILE="$1"
SERVER_URL="{server_url}"
CANDIDATE=$(cat "$CANDIDATE_FILE")
BODY=$(jq -n --arg c "$CANDIDATE" '{{candidate: $c}}')
RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/validate" \\
    -H "Content-Type: application/json" -d "$BODY")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
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


def _build_program_md(task: Task, budget: BudgetTracker, *, perfect_score: float | None) -> str:
    optional = ""
    if task.objective:
        optional += f"## Objective\n{task.objective}\n\n"
    if task.background:
        optional += f"## Background\n{task.background}\n\n"

    has_eval_budget = budget.max_evals is not None
    if task.has_dataset and task.train_set:
        train_size = len(task.train_set)
        cost_line = (
            f"\nEach example costs 1 budget unit. A full train eval costs {train_size} units."
            if has_eval_budget
            else ""
        )
        val_section = ""
        if task.val_set:
            val_size = len(task.val_set)
            val_cost = f" Costs {val_size} budget units." if has_eval_budget else ""
            val_section = (
                f"\n### Validation\nThere is a hidden validation set ({val_size} examples). "
                f"You cannot see individual val examples or their scores.\n"
                f"```bash\n./validate.sh candidate.txt\n```\n"
                f"Returns only the aggregate val_score.{val_cost}"
            )
        eval_section = (
            f"## Evaluation\n"
            f"This is a **generalization** task with {train_size} training examples.\n"
            f"```bash\n# Evaluate on all training examples\n./eval.sh candidate.txt\n\n"
            f"# Evaluate on specific examples\n./eval.sh candidate.txt --ids example_0,example_1\n```"
            f"{cost_line}{val_section}"
        )
    else:
        cost_line = "\nEach eval costs 1 budget unit." if has_eval_budget else ""
        eval_section = (
            f"## Evaluation\nThis is a **single-task** optimization.\n```bash\n./eval.sh candidate.txt\n```{cost_line}"
        )

    budget_lines: list[str] = []
    if budget.max_evals is not None:
        budget_lines.append(f"- **Evaluation cap:** {budget.max_evals} calls (each eval = 1 unit).")
    if budget.max_token_cost is not None:
        budget_lines.append(
            f"- **LLM token budget:** ${budget.max_token_cost:.2f} "
            "(enforced by the Claude CLI; you'll stop automatically when exhausted)."
        )
    if not budget_lines:
        budget_lines.append("- No hard budget limit — iterate freely.")
    budget_section = "\n".join(budget_lines)
    if perfect_score is not None:
        budget_section += (
            f"\n\n## Perfect Score\nThe maximum achievable score is **{perfect_score}**. "
            f"Once you reach this score, stop iterating — further improvements are impossible."
        )

    if task.has_dataset and task.train_set:
        val_step = "5. Validate periodically with `./validate.sh candidate.txt`.\n" if task.val_set else ""
        final_n = 6 if task.val_set else 5
        strategy_section = (
            "1. Read the training examples in `train/` to understand the task.\n"
            "2. Run `./eval.sh candidate.txt` for a baseline.\n"
            "3. Analyze failures — inspect examples that scored 0.\n"
            "4. Improve the candidate; spot-check with `--ids`.\n"
            f"{val_step}{final_n}. Write your best candidate to `best_candidate.txt`."
        )
    else:
        strategy_section = (
            "1. Read the candidate (`candidate.txt`) and the problem description above.\n"
            "2. Run `./eval.sh candidate.txt` for a baseline score and any judge feedback.\n"
            "3. Revise the candidate based on what you learned.\n"
            "4. Iterate; write your best solution to `best_candidate.txt` as you improve."
        )

    scripts = "eval.sh, validate.sh" if task.val_set else "eval.sh"
    val_rule = "\n- You cannot see the validation examples." if task.val_set else ""
    exhaust_rule = (
        "\n- When the budget is exhausted, scripts return BUDGET_EXHAUSTED." if budget.max_evals is not None else ""
    )
    rules_section = (
        f"- You cannot modify {scripts} or the server.{val_rule}\n"
        f"- Focus on meaningful improvements each iteration.{exhaust_rule}"
    )

    return _PROGRAM_MD.format(
        name=task.name,
        optional_sections=optional,
        candidate_len=len(task.initial_candidate),
        eval_section=eval_section,
        budget_section=budget_section,
        strategy_section=strategy_section,
        rules_section=rules_section,
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

    if task.val_set:
        validate_script = work_dir / "validate.sh"
        validate_script.write_text(VALIDATE_SCRIPT.format(server_url=server_url))
        validate_script.chmod(0o755)

    if task.train_set:
        train_dir = work_dir / "train"
        train_dir.mkdir(exist_ok=True)
        for eid, ex in server.iter_split("train"):
            (train_dir / f"{eid}.json").write_text(json.dumps(example_to_json(eid, ex), indent=2, default=str))


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

        if self.run_dir:
            work_dir = Path(self.run_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="omni_cc_")
            work_dir = Path(self._pending_tempdir.name)

        _materialize_sandbox(work_dir, task, server, budget, perfect_score=self.stop_at_score)

        candidate_file = work_dir / "candidate.txt"
        best_file = work_dir / "best_candidate.txt"
        eval_script = work_dir / "eval.sh"
        validate_script = work_dir / "validate.sh"

        prompt = (
            f"Read program.md for full task instructions. "
            f"The candidate to improve is in {candidate_file}. "
            f"Write your best candidate to {best_file}. "
            f"Use {eval_script} to evaluate."
        )
        if task.val_set:
            prompt += f" Use {validate_script} to check validation score."

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

    def process_result(self, result: Result, output_dir: Path) -> None:
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
