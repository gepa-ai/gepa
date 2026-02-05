#!/usr/bin/env python3
"""Blackbox optimization with GEPA."""

import os
import time

import numpy as np

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)
from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode
from examples.polynomial.evalset.problems import problems, problem_configs

# =============================================================================
# CONFIGURATION
# =============================================================================

LLM = "openai/gpt-5"
PROBLEM_INDEX = 0
EVALUATION_BUDGET = 100
TIMEOUT = 300

# =============================================================================
# PROMPTS
# =============================================================================

SEED_CODE = '''
import numpy as np

def solve(objective_function, config, best_xs=None):
    bounds = np.array(config['bounds'])
    all_attempts = []

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    score = objective_function(x)
    all_attempts.append({"x": x.copy(), "score": score})

    return {"x": x, "all_attempts": all_attempts}
'''

OBJECTIVE = "Evolve Python code that minimizes a blackbox objective function using the available evaluation budget efficiently."

BACKGROUND = """
You are optimizing code that solves blackbox minimization problems (lower is better).

## Function Signature
```python
def solve(objective_function, config, best_xs=None):
    # config contains: bounds (array of [min, max] per dim), dim (int), budget (int)
    # best_xs: list of {"x": array, "score": float} sorted by score (best first)
    # Returns: {"x": best_x, "all_attempts": [{"x": x, "score": score}, ...]}
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

## Mutation Strategies
1. Hyperparameter tuning (learning rates, population sizes, iterations)
2. Algorithm changes (evolutionary, gradient-free, bayesian optimization)
3. Initialization strategies (random, latin hypercube, from trajectory)
4. Hybrid approaches (local + global search)

## Output Format
Provide the improved code in a single code block with triple backticks.
"""

# =============================================================================
# HELPERS
# =============================================================================


def execute_code(code: str, best_xs: list[dict] | None) -> dict:
    """Execute optimization code. Returns dict with success, result, error, stdout."""
    fn = problems[PROBLEM_INDEX]

    def objective_function(x):
        return fn.do_evaluate(np.array(x))

    result = _execute_code(
        code=code,
        timeout=TIMEOUT,
        mode=ExecutionMode.IN_PROCESS,
        entry_point="solve",
        entry_point_kwargs={
            "objective_function": objective_function,
            "config": {"bounds": fn.bounds, "dim": fn.dim, "budget": EVALUATION_BUDGET},
            "best_xs": best_xs or [],
        },
        seed=0,
    )

    if not result.success:
        return {
            "success": False,
            "error": result.error or "Execution failed",
            "traceback": result.traceback or "",
            "stdout": result.stdout,
        }

    ret = result.variables.get("__return__")
    if not isinstance(ret, dict) or "x" not in ret or "all_attempts" not in ret:
        return {
            "success": False,
            "error": "solve() must return {'x': array, 'all_attempts': [...]}",
            "stdout": result.stdout,
        }

    return {"success": True, "result": ret, "stdout": result.stdout}


def extract_best_xs(best_example_evals: list[dict], top_k: int = 100) -> list[dict]:
    """Extract best_xs from previous evaluations, sorted by score."""
    all_attempts = []
    for e in best_example_evals or []:
        side_info = e.get("side_info", {})
        all_attempts.extend(side_info.get("top_attempts", []))
    sorted_attempts = sorted(all_attempts, key=lambda t: t["score"])[:top_k]
    return [{"x": np.array(t["x"]), "score": t["score"]} for t in sorted_attempts]


def build_side_info(result: dict, all_attempts: list[dict]) -> dict:
    """Build side_info from execution result."""
    # Serialize attempts for JSON
    def serialize(attempts):
        return [
            {"x": a["x"].tolist() if hasattr(a["x"], "tolist") else a["x"], "score": a["score"]}
            for a in attempts
        ]

    sorted_attempts = sorted(all_attempts, key=lambda a: a["score"])
    top_attempts = serialize(sorted_attempts[:50])

    return {
        "Input": problem_configs[PROBLEM_INDEX]["name"],
        "top_attempts": top_attempts,
        "num_attempts": len(all_attempts),
        "stdout": result.get("stdout", "")[:2000],
        "error": result.get("error", ""),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    log_dir = f"outputs/polynomial/{time.strftime('%y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    def fitness_fn(candidate, best_example_evals):
        """Evaluate optimization code."""
        code = candidate["code"]
        best_xs = extract_best_xs(best_example_evals)

        result = execute_code(code, best_xs)

        if result["success"]:
            all_attempts = result["result"]["all_attempts"]
            score = -min(a["score"] for a in all_attempts)  # Negate for GEPA (maximizes)
        else:
            all_attempts = []
            score = -1e9

        side_info = build_side_info(result, all_attempts)
        return score, side_info

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm=LLM,
        ),
    )

    optimize_anything(
        seed_candidate={"code": SEED_CODE},
        fitness_fn=fitness_fn,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
