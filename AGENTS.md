# GEPA

GEPA (Genetic-Pareto) is a Python framework for optimizing text components (AI prompts, code, instructions) using LLM-based reflection and Pareto-efficient evolutionary search.

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

If you are a coding agent invoking `gepa.optimize` (or `dspy.GEPA`) as part of executing a user's plan, you must size the budget correctly. **`max_metric_calls` is the only knob controlling iteration count — there is no separate iterations flag.** Setting it too low silently produces a 2-point trajectory that looks like optimization but is not.

**Required behavior:**

1. **Compute the budget from the data, don't pick a number.**
   `max_metric_calls >= len(valset) + 10 * (reflection_minibatch_size + len(valset))`
   (`reflection_minibatch_size` defaults to 3 if unset.) This supports ~10 proposal attempts. If you change the valset size or minibatch size, recompute.

2. **Verify proposal depth before declaring the goal reached.**
   After `gepa.optimize` returns, check `result.num_candidates - 1` (this excludes the baseline). If it's less than 5, the optimization was not meaningful — re-run with a larger budget regardless of whether the final validation score looks better than baseline.

3. **Treat `gepa.GEPABudgetWarning` as a hard failure of the experiment, not a soft warning.** It fires at the start of the run if the budget is below the recommended floor, and again at the end if fewer than 3 proposals were accepted. Do not present score graphs while either warning is active.

4. **Inspect `<run_dir>/gepa-result.json` before drawing any "scores across iterations" chart.** Confirm `num_candidates >= 6` and that `val_aggregate_scores` has at least 5 distinct rising values. A 2-point trajectory is one proposal step, not optimization.

These rules exist because of a real incident — see [#375](https://github.com/gepa-ai/gepa/issues/375) and [docs/guides/coding-agents.md](docs/docs/guides/coding-agents.md) for the full story and copy-paste prompt blocks.
