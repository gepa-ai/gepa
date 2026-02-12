#!/usr/bin/env python3
from examples.circle_packing.utils import (
    execute_code,
    extract_best_circles,
    SEED_CODE1,
    compute_multiple_metrics,
    CIRCLE_PACKING_BACKGROUND,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    OptimizationState,
    ReflectionConfig,
    optimize_anything,
)


def fitness_fn(candidate, opt_state: OptimizationState | None = None):
    code = candidate["code"]
    warm_start = extract_best_circles(opt_state)
    result = execute_code(code, 600, warm_start)

    if result["success"]:
        circles = result["result"]["circles"]
        score = result["result"]["validation_details"]["sum_radii"]
        metrics = compute_multiple_metrics(result["result"]["all_scores"])
    else:
        circles = None
        score = 0.0
        metrics = {"sum_radii": 0.0}
        
    side_info = {
        "scores": {"sum_radii": score},
        "metrics": metrics,
        "code": code,
        "circles": circles,
        "stdout": result.get("stdout", ""),
        "error": result.get("error"),
        "traceback": result.get("traceback"),
        "validation_details": result.get("validation_details"),
    }

    return score, side_info


def main():
    seed_candidate = {
        "code": SEED_CODE1, 
    }

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir="outputs/circle_packing",
            max_metric_calls=150,
            track_best_outputs=True,
            frontier_type="objective",
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(reflection_lm="openai/gpt-5"),
    )

    optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=fitness_fn,
        config=gepa_config,
        objective="Optimize circle packing code to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
    )


if __name__ == "__main__":
    main()
