#!/usr/bin/env python3
"""Blackbox optimization with GEPA."""

import time

from examples.polynomial.utils import (
    execute_code,
    extract_best_xs,
    append_eval_history,
    SEED_CODE,
    OBJECTIVE,
    BACKGROUND,
)

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

LLM = "openai/gpt-5"
PROBLEM_INDEX = 0
EVALUATION_BUDGET = 100
TIMEOUT = 300


def main():
    log_dir = f"outputs/polynomial/{time.strftime('%y%m%d_%H%M%S')}"

    def fitness_fn(candidate, best_example_evals):
        code = candidate["code"]
        best_xs = extract_best_xs(best_example_evals)

        result = execute_code(
            code=code,
            problem_index=PROBLEM_INDEX,
            timeout=TIMEOUT,
            budget=EVALUATION_BUDGET,
            best_xs=best_xs,
        )

        append_eval_history(log_dir, result["all_attempts"])

        side_info = {
            "score": result["score"],
            "top_50_attempts": result["top_50_attempts"],
            "bottom_50_attempts": result["bottom_50_attempts"],
            "Stdout": result.get("stdout", ""),
            "Error": result.get("error", ""),
            "Num attempts": len(result["all_attempts"]),
            "Allowed attempts": EVALUATION_BUDGET,
            "Traceback": result.get("traceback", ""),
        }

        return (result["score"], side_info)

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_candidate_proposals=20,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm=LLM,
        ),
    )

    optimize_anything(
        seed_candidate={"code": SEED_CODE},
        evaluator=fitness_fn,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
