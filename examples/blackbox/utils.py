"""Utilities for blackbox optimization: code execution and seed templates."""

import numpy as np
import json
from pathlib import Path

from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode
from examples.blackbox.evalset.problems import problems


def execute_code(
    code: str,
    problem_index: int,
    budget: int,
    best_xs: list[dict] | None,
    seed: int = 0,
) -> dict:
    """Execute optimization code and return structured result."""
    fn = problems[problem_index]

    def objective_function(x):
        return fn.do_evaluate(np.array(x))

    result = _execute_code(
        code=code,
        timeout=budget * 2,
        mode=ExecutionMode.IN_PROCESS,
        entry_point="solve",
        entry_point_kwargs={
            "objective_function": objective_function,
            "config": {"bounds": fn.bounds, "dim": fn.dim, "budget": budget},
            "best_xs": best_xs or [],
        },
        seed=seed,
    )

    base = {"stdout": result.stdout, "stderr": result.stderr}

    fail = {
        "success": False,
        "score": -1e9,
        "all_attempts": [],
        "all_trials": [],
        **base,
    }

    if not result.success:
        return {**fail, "error": result.error or "Execution failed", "traceback": result.traceback or ""}

    ret = result.variables.get("__return__")
    if not isinstance(ret, dict) or "x" not in ret or "score" not in ret or "all_attempts" not in ret:
        return {**fail, "error": "solve() must return {'x': array, 'score': float, 'all_attempts': [...]}"}

    all_attempts = ret["all_attempts"]
    sorted_attempts = sorted(all_attempts, key=lambda a: a["score"])
    return {
        "success": True,
        "score": -ret["score"],
        "all_attempts": all_attempts,
        "all_trials": serialize_attempts(sorted_attempts),
        **base,
    }


def extract_best_xs(opt_state, top_k: int = 200) -> list[dict]:
    """Extract best_xs from OptimizationState, sorted by score (best first)."""
    if opt_state is None:
        return []
    best_example_evals = opt_state.best_example_evals
    if not best_example_evals:
        return []
    all_attempts = []
    for e in best_example_evals:
        side_info = e.get("side_info", {})
        all_attempts.extend(side_info.get("all_trials", []))
    sorted_attempts = sorted(all_attempts, key=lambda t: t["score"])[:top_k]
    return [{"x": np.array(t["x"]), "score": t["score"]} for t in sorted_attempts]


def serialize_attempts(attempts):
    return [
        {
            "x": a["x"].tolist() if hasattr(a["x"], "tolist") else a["x"],
            "score": a["score"],
        }
        for a in attempts
    ]


class BudgetTracker:
    """Counts actual objective calls even on crash."""

    def __init__(self, total, num_candidates):
        self.total = total
        self.num_candidates = num_candidates
        self.used = 0
        self.candidates = 0

    @property
    def remaining(self):
        return self.total - self.used

    @property
    def per_candidate(self):
        left = self.num_candidates - self.candidates
        if left <= 0:
            return 0
        return self.remaining // left

    def record(self, result):
        self.candidates += 1
        n = result["actual_call_count"]
        self.used += n
        return n



BACKGROUND = """
You are optimizing code that solves blackbox minimization problems (lower is better).

## Function Signature
```python
def solve(objective_function, config, best_xs=None):
    # config contains: bounds (array of [min, max] per dim), dim (int), budget (int)
    # best_xs: list of {"x": array, "score": float} sorted by score (best first)
    # Returns: {"x": best_x, "score": best_score, "all_attempts": [{"x": x, "score": score}, ...]}
```

## Code Requirements
- Always include necessary imports (e.g., `import numpy as np`)
- Return a dict with "x" (best solution found) and "all_attempts" (list of all evaluations)
- Each attempt in all_attempts must have "x" (numpy array) and "score" (float)
- Use `objective_function(x)` to evaluate candidates (lower score is better)
- Stay within `config['budget']` calls
- Full use of all the allowed evaluation budget leads to better performance
- Use `best_xs` to leverage previous evaluation data (if available)

## Using Trajectory Data
The `best_xs` parameter provides ALL previous (x, score) evaluations sorted by score (best first).
This enables sophisticated strategies:
1. **Multi-start optimization**: Initialize from multiple top-K solutions in best_xs
2. **Surrogate modeling**: Build GP/RBF models from best_xs to guide search
3. **Density-based exploration**: Avoid crowded regions already explored
4. **Gradient estimation**: Estimate gradients from nearby points in best_xs
5. **Trust region**: Define regions around good solutions to focus search

## Available Libraries
Any package is ready to use. You can import them freely to maximize the performance.

## Mutation Strategies
1. Hyperparameter tuning (learning rates, population sizes, iterations)
2. Algorithm changes (evolutionary, gradient-free, bayesian optimization)
3. Initialization strategies (random, latin hypercube, from trajectory)
4. Hybrid approaches (local + global search)
5. Exploitation vs exploration balance
6. Trajectory-informed search (surrogate models, multi-start, density avoidance)

## Output Format
Provide the improved code in a single code block with triple backticks.
"""
