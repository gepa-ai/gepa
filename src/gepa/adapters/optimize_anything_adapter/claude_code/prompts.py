"""Claude Code prompt strings and builders for optimize_anything."""

from __future__ import annotations

import json
from typing import Any

CC_RLM_SYSTEM_PROMPT = """\
## Sub-Agent Spawning (RLM)

You have access to `python -m rlm.cli` to spawn sub-agents. Each sub-agent is a \
separate `claude -p` call with its own context window. Use this to divide expensive \
work across cheap and expensive models.

IMPORTANT: stdout is not captured when running inside Claude Code. \
Always use `-o /tmp/rlm_result.txt` to write output to a file, then read the file.

### Usage

```bash
# Single sub-agent — always use -o to capture output
python -m rlm.cli --model haiku -o /tmp/rlm_out.txt "Your focused sub-query here"
cat /tmp/rlm_out.txt

# JSON output (structured, easier to parse)
python -m rlm.cli --json --model haiku -o /tmp/rlm_out.txt "Summarize eval results"
cat /tmp/rlm_out.txt

# Parallel sub-agents (run concurrently, share budget)
python -m rlm.cli --parallel --model haiku -o /tmp/rlm_parallel.txt \
  "Read engine.py and summarize" "Read state.py and summarize"
cat /tmp/rlm_parallel.txt

# Multiple independent sub-agents — use different output files
python -m rlm.cli --model haiku -o /tmp/rlm_1.txt "Read and summarize core/" &
python -m rlm.cli --model haiku -o /tmp/rlm_2.txt "Read and summarize proposer/" &
wait
cat /tmp/rlm_1.txt
cat /tmp/rlm_2.txt
```

### Model Cascade Strategy

Use **haiku** for bulk reading, extraction, and summarization (cheap, fast). \
Use **sonnet** for analysis, reasoning, and synthesis (more capable).

Example cascade:
1. Spawn haiku sub-agents to read and summarize raw data
2. Use the summaries yourself for the final reasoning

### Cost Guidelines

- haiku: ~$0.001/call for short tasks, ~$0.01-0.05 for reading many files
- sonnet: ~$0.02-0.10 per call depending on context size
- A haiku-read → sonnet-synthesize cascade is typically 3x cheaper than sonnet-does-everything

### Rules

- ALWAYS use `-o <file>` then `cat <file>` — stdout does not work inside Claude Code
- Keep sub-queries focused and specific — one clear task per spawn
- Prefer haiku for any task that is mostly reading/extracting/summarizing
- Use parallel mode or background `&` + `wait` when sub-tasks are independent
- Sub-agents have depth limits and shared budget — they will error if limits are hit"""


CC_AGENTIC_MUTATION_SYSTEM_PROMPT = """\
You are an optimization agent. Your job is to study an optimization problem, \
read relevant eval history, and propose an improved candidate solution.

## Sub-Agent Spawning (RLM — Recursive Language Model)

You can spawn cheaper Haiku sub-agents to read eval files without consuming your \
own context window. Use `python -m rlm.cli` via your Bash tool.

IMPORTANT: stdout is NOT captured when running inside Claude Code. \
Always write output to a file with -o, then read it:

  # Read a single file
  python -m rlm.cli --model haiku -o /tmp/rlm_out.txt \\
    "Read evals/c00003/tasks/task_00007_eval_00000.json and return its full contents"
  cat /tmp/rlm_out.txt

  # Read multiple files in parallel
  python -m rlm.cli --model haiku --parallel -o /tmp/rlm_out.txt \\
    "Read evals/c00003/tasks/task_00007_eval_00000.json" \\
    "Read evals/c00003/tasks/task_00012_eval_00000.json"
  cat /tmp/rlm_out.txt

## Model strategy

- haiku (~20x cheaper): reading files, extracting outputs, summarizing
- you (sonnet): reasoning, synthesizing insights, writing the improved candidate

## Output format

Write your improved candidate in a single fenced code block at the end of your \
response. Do not write anything after the closing fence.
"""


def build_note_update_prompt(
    current_note: str,
    iteration: int,
    parent_candidate: str,
    child_candidate: str,
    example_ids: list[Any],
    parent_scores: list[float],
    child_scores: list[float],
    parent_outputs: list[Any],
    child_outputs: list[Any],
    parent_objective_scores: list[dict[str, float]] | None = None,
    child_objective_scores: list[dict[str, float]] | None = None,
    objective: str | None = None,
) -> str:
    """Build the prompt used to append a note entry after each mutation."""
    parent_avg = sum(parent_scores) / len(parent_scores) if parent_scores else 0.0
    child_avg = sum(child_scores) / len(child_scores) if child_scores else 0.0
    delta = child_avg - parent_avg
    direction = "improvement" if delta > 0 else "regression" if delta < 0 else "no change"
    parent_eval_results = _format_eval_results(
        example_ids=example_ids,
        scores=parent_scores,
        outputs=parent_outputs,
        objective_scores=parent_objective_scores,
    )
    child_eval_results = _format_eval_results(
        example_ids=example_ids,
        scores=child_scores,
        outputs=child_outputs,
        objective_scores=child_objective_scores,
    )

    return f"""\
You are updating a reflection journal for an optimization run.

## Current note
{current_note if current_note else "(empty — this is the first entry)"}

## Iteration {iteration}
Objective: {objective or "not specified"}

Parent score: {parent_avg:.4f}
Child score:  {child_avg:.4f}
Delta: {delta:+.4f} ({direction})

Parent candidate:
{parent_candidate[:1500]}{"..." if len(parent_candidate) > 1500 else ""}

Child candidate:
{child_candidate[:1500]}{"..." if len(child_candidate) > 1500 else ""}

## Parent eval results for this iteration
```json
{parent_eval_results}
```

## Child eval results for this iteration
```json
{child_eval_results}
```

## Instructions
Append one concise reflection entry (3-5 sentences). Start with "## Iteration {iteration}".
Focus on:
- What the child changed vs the parent
- Whether it helped or hurt and why
- Any generalizable insight for future mutations
Output ONLY the new entry text. Do not repeat the existing note.
"""


def _format_eval_results(
    example_ids: list[Any],
    scores: list[float],
    outputs: list[Any],
    objective_scores: list[dict[str, float]] | None,
) -> str:
    records = []
    for idx, (example_id, score, output) in enumerate(zip(example_ids, scores, outputs, strict=False)):
        record: dict[str, Any] = {
            "example_id": example_id,
            "score": score,
            "output": output,
        }
        if objective_scores is not None and idx < len(objective_scores):
            record["objective_scores"] = objective_scores[idx]
        records.append(record)
    return json.dumps(records, indent=2, default=str)
