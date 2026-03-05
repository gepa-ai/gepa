# Blackbox Optimization

Optimize Python code that minimizes a blackbox objective function within a fixed evaluation budget.

## How it works

- GEPA evolves a `solve()` function that calls `objective_function(x)` to search for the global minimum
- Previous best solutions (`best_xs`) are passed in for warm-starting
- Score is negated (lower objective = higher GEPA score)

## Setup

From the repo root (`gepa/`):

```bash
uv venv
uv pip install numpy scipy optuna scikit-learn
uv pip install -e .
```

## Run

```bash
export OPENAI_API_KEY=...
uv run python -m examples.blackbox.main
```

Results are saved to `outputs/blackbox/<problem_index>/`.
