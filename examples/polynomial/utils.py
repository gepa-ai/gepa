"""Utilities for polynomial optimization: code execution and seed templates."""

import numpy as np
import json
from pathlib import Path

from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode
from examples.polynomial.evalset.problems import problems


def execute_code(
    code: str,
    problem_index: int,
    budget: int,
    best_xs: list[dict] | None,
    timeout: int = 300,
    seed: int = 0,
) -> dict:
    """Execute optimization code and return structured result."""
    fn = problems[problem_index]

    def objective_function(x):
        return fn.do_evaluate(np.array(x))

    result = _execute_code(
        code=code,
        timeout=timeout,
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
        "top_50_attempts": [],
        "bottom_50_attempts": [],
        **base,
    }

    if not result.success:
        return {**fail, "error": result.error or "Execution failed", "traceback": result.traceback or ""}

    ret = result.variables.get("__return__")
    if not isinstance(ret, dict) or "x" not in ret or "score" not in ret or "all_attempts" not in ret:
        return {**fail, "error": "solve() must return {'x': array, 'score': float, 'all_attempts': [...]}"}

    all_attempts = ret["all_attempts"]
    top_50, bottom_50 = extract_attempts(all_attempts)
    return {
        "success": True,
        "score": -ret["score"],
        "all_attempts": all_attempts,
        "top_50_attempts": top_50,
        "bottom_50_attempts": bottom_50,
        **base,
    }


def append_eval_history(log_dir: str, all_attempts: list[dict]):
    """Append all attempts from one evaluation to history."""
    path = Path(log_dir) / "evaluation_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for a in all_attempts:
            entry = {
                "x": a["x"].tolist() if hasattr(a["x"], "tolist") else a["x"],
                "score": a["score"],
            }
            f.write(json.dumps(entry) + "\n")


def extract_best_xs(best_example_evals: list[dict], top_k: int = 100) -> list[dict]:
    """Extract best_xs from best evaluations, sorted by score (best first)."""
    all_attempts = []
    for e in best_example_evals or []:
        side_info = e.get("side_info", {})
        all_attempts.extend(side_info.get("top_50_attempts", []))
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


def extract_attempts(attempts: list[dict]):
    sorted_attempts = sorted(attempts, key=lambda a: a["score"])
    top_50 = serialize_attempts(sorted_attempts[:50])
    bottom_50 = (
        serialize_attempts(sorted_attempts[-50:])
        if len(sorted_attempts) > 50
        else []
    )
    return top_50, bottom_50



# =============================================================================
# SEED CODE
# =============================================================================

SEED_CODE = '''
import numpy as np

def solve(objective_function, config, best_xs=None):
    bounds = np.array(config['bounds'])
    all_attempts = []

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    score = objective_function(x)
    all_attempts.append({"x": x.copy(), "score": score})

    return {"x": x, "score": score, "all_attempts": all_attempts}
'''


# =============================================================================
# PROMPTS
# =============================================================================

OBJECTIVE = "Evolve Python code that minimizes a blackbox objective function using the available evaluation budget efficiently."

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
