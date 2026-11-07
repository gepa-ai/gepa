# GEPA

GEPA (Genetic-Pareto) is a Python framework for optimizing text components (AI prompts, code, instructions) using LLM-based reflection and Pareto-efficient evolutionary search.

## Agent Skill

This repository ships an **Agent Skill** at `.claude/skills/gepa-optimize-anything/`. If you are a coding agent and the user wants to **auto-optimize, tune, or search over any scorable text artifact** (a prompt, program/code, config, regex/SQL, or agent scaffold) with `gepa.optimize_anything`, read `.claude/skills/gepa-optimize-anything/SKILL.md` first and follow it. It is auto-discovered by Claude Code and by other agents that read `.claude/skills/` (Cursor, VS Code/Copilot, Codex, Gemini CLI). To install it standalone in another project: `/plugin marketplace add gepa-ai/gepa` then `/plugin install gepa-optimize-anything@gepa`.

## Setup

We use **uv** for dependency management. The project uses setuptools as the build backend. All python executions must be done through uv.

```bash
uv sync --extra dev
```

## Project Structure

- `src/gepa/` — main package source
  - `core/` — optimization loop, state, evaluation
  - `proposer/` — candidate proposal and mutation logic
  - `adapters/` — integration adapters (DSPy, RAG, MCP, etc.)
  - `strategies/` — batch sampling and candidate selection
  - `logging/` — experiment tracking and logging
- `tests/` — pytest test suite
- `docs/` — mkdocs documentation site

## Build & Test

```bash
uv run pytest
uv run ruff check src/
uv run ruff format src/
uv run pyright src/
```

## Code Style

- Linter/formatter: ruff (line length 120, double quotes, space indent)
- Type checking: pyright
- Python target: 3.10+
- No relative imports (enforced by ruff)

## Running GEPA from inside an autonomous loop

If you are a coding agent invoking `gepa.optimize` (or `dspy.GEPA`) as part of executing a user's plan, pick the stopping strategy deliberately. There are two paths:

- **`stop_callbacks` only** (preferred default for unattended runs): skip `max_metric_calls` and pass e.g. `NoImprovementStopper(max_iterations_without_improvement=10)`. GEPA runs until it actually converges; you don't have to guess a budget upfront.
- **`max_metric_calls`**: fixed budget. If you choose this path, the budget must be `> 15 * len(valset)` (rule of thumb) or precisely `len(valset) + 15 * (reflection_minibatch_size + len(valset))` (with `reflection_minibatch_size` defaulting to 3). Anything smaller silently produces a short trajectory that looks like optimization but is not.

Do not pick a small `max_metric_calls` "to save time" — that's the documented failure mode.

**Required behavior:**

1. **Verify proposal depth before declaring the goal reached.**
   After `gepa.optimize` returns, check `result.num_candidates - 1` (excludes the baseline). If it's less than 5, the optimization was not meaningful — re-run with a larger budget or a looser stopper, regardless of whether the final validation score looks better than baseline.

2. **Treat `gepa.GEPABudgetWarning` as a hard failure of the experiment, not a soft warning.** It fires at the start of the run if you set `max_metric_calls` below the recommended floor, and again at the end if fewer than 3 proposals were accepted (regardless of which stopper fired). Do not consider a GEPA run as valid if either warning is active.

3. **Inspect `<run_dir>/gepa-result.json` before drawing any "scores across iterations" chart.** Confirm `num_candidates >= 6` and that `val_aggregate_scores` has at least 5 distinct rising values. A 2-point trajectory is one proposal step, not optimization.
