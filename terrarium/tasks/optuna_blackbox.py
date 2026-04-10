"""Blackbox optimization task: evolve code that minimizes a blackbox function.

This is a single-task problem. The candidate is Python code that implements
an optimization strategy against an unknown objective function.
"""

from __future__ import annotations

from typing import Any

from terrarium.registry import register_task
from terrarium.task import Task

DESCRIPTION = """\
Evolve Python code that minimizes a blackbox objective function.
The candidate is Python code defining `solve(objective_function, config, best_xs)`
where config has 'bounds', 'dim', 'budget'. The code should efficiently explore
the search space using the given evaluation budget.
Score = negative function value (higher is better, since we minimize the objective).
"""

INITIAL_CANDIDATE = """\
import numpy as np

def solve(objective_function, config, best_xs=None):
    bounds = np.array(config['bounds'])
    all_attempts = []

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    score = objective_function(x)
    all_attempts.append({"x": x.copy(), "score": score})

    return {"x": x, "score": score, "all_attempts": all_attempts}
"""


def evaluate(candidate: str, problem_index: int = 46, budget: int = 200) -> tuple[float, dict[str, Any]]:
    """Evaluate blackbox optimization code on a specific problem."""
    from examples.blackbox.utils import execute_code

    result = execute_code(
        code=candidate,
        problem_index=problem_index,
        budget=budget,
        best_xs=None,
    )

    score = result.get("score", -1e9)
    return score, {
        "score": score,
        "all_trials": result.get("all_trials", []),
        "stdout": result.get("stdout", ""),
        "error": result.get("error", ""),
        "traceback": result.get("traceback", ""),
    }


TASK = register_task(Task(
    name="optuna_blackbox",
    description=DESCRIPTION,
    initial_candidate=INITIAL_CANDIDATE,
    eval_fn=evaluate,
    metadata={
        "type": "single_task",
        "candidate_type": "code",
        "objective": "Minimize a blackbox objective function using the evaluation budget efficiently.",
    },
))
