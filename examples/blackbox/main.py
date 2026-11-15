#!/usr/bin/env python3
"""Blackbox optimization with GEPA + external budget tracking."""

from examples.blackbox.utils import (
    BACKGROUND,
    BudgetTracker,
    execute_code,
    extract_best_xs,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

NUM_PROPOSALS = 10
EVALUATION_BUDGET = 2000
SEED_CODE = """
import numpy as np

def solve(objective_function, config, best_xs=None):
    bounds = np.array(config['bounds'])
    all_attempts = []

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    score = objective_function(x)
    all_attempts.append({"x": x.copy(), "score": score})

    return {"x": x, "score": score, "all_attempts": all_attempts}
"""


def main(problem_index: int = 46):
    budget = BudgetTracker(EVALUATION_BUDGET, NUM_PROPOSALS)

    def evaluate(candidate, opt_state):
        if budget.remaining <= 0:
            return -1e9, {"score": -1e9, "error": "No budget remaining"}

        candidate_budget = budget.per_candidate
        best_xs = extract_best_xs(opt_state)

        result = execute_code(
            code=candidate,
            problem_index=problem_index,
            budget=candidate_budget,
            best_xs=best_xs,
        )

        budget.record(result)

        side_info = {
            "score": result["score"],
            "all_trials": result.get("all_trials", []),
            "stdout": result.get("stdout", ""),
            "error": result.get("error", ""),
            "traceback": result.get("traceback", ""),
            "budget_total": budget.total,
            "budget_used": budget.used,
            "proposal_total": NUM_PROPOSALS,
            "proposal_completed": budget.candidates,
        }
        return result["score"], side_info

    optimize_anything(
        evaluator=evaluate,
        seed_candidate=SEED_CODE,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=f"outputs/blackbox/{problem_index}",
                max_candidate_proposals=20,
                track_best_outputs=True,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openai/gpt-5",
            ),
        ),
        objective="Evolve Python code that minimizes a blackbox objective function using the available evaluation budget efficiently.",
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
