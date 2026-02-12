#!/usr/bin/env python3
from examples.circle_packing.utils import (
    execute_code,
    SEED_CODE,
    compute_multiple_metrics,
    CIRCLE_PACKING_BACKGROUND,
    extract_best_circles
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    RefinerConfig,
    ReflectionConfig,
    optimize_anything,
)


def fitness_fn(candidate, best_example_evals):
    code = candidate["code"]
    warm_start = extract_best_circles(best_example_evals)
    result = execute_code(code, 600, warm_start)

    if result["success"]:
        circles = result["result"]["circles"]
        score = result["result"]["validation_details"]["sum_radii"]
        metrics = compute_multiple_metrics(result["result"]["all_scores"])
    else:
        circles = None
        score = 0.0
        metrics = {"max_score": 0.0, "mean_score": 0.0, "ema_score_fixed": 0.0, "ema_score_adaptive": 0.0}
        
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
    import time

    run_dir = "outputs/circle_packing/" + time.strftime('%y%m%d_%H%M%S')
    print(f"Run directory: {run_dir}")

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir,
            max_metric_calls=300,
            track_best_outputs=True,
            frontier_type="objective",
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm="openai/gpt-5",
        ),
        refiner=RefinerConfig(),
    )

    optimize_anything(
        seed_candidate={"code": SEED_CODE},
        fitness_fn=fitness_fn,
        config=gepa_config,
        objective="Optimize circle packing code to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
    )


if __name__ == "__main__":
    main()
