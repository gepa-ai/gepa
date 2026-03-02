# Blackbox Optimization

Optimize Python code that minimizes a blackbox objective function within a fixed evaluation budget.

## How it works

- GEPA evolves a `solve()` function that calls `objective_function(x)` to search for the global minimum
- Previous best solutions (`best_xs`) are passed in for warm-starting
- Score is negated (lower objective = higher GEPA score)

## Setup

```bash
uv pip install numpy scipy optuna scikit-learn
```

## Run

From the repo root (`gepa-optimize-anything/`):

```bash
export OPENAI_API_KEY=...
python -m examples.blackbox.main
```

Results are saved to `outputs/blackbox/<problem_index>/`.
