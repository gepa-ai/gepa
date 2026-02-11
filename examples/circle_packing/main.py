#!/usr/bin/env python3
import numpy as np

from examples.circle_packing.utils import (
    execute_code,
    SEED_CODE1,
    SEED_CODE2,
    compute_multiple_metrics,
    CIRCLE_PACKING_BACKGROUND,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    RefinerConfig,
    ReflectionConfig,
    optimize_anything,
)

# Constants
MAX_METRIC_CALLS = 150
NUM_CIRCLES = 26
TIMEOUT = 600  # 10 minutes, same as OpenEvolve



def extract_best_circles(best_example_evals: list[dict]) -> np.ndarray | None:
    """Extract best circles from result."""
    if not best_example_evals:
        return None
    circles = [
        e["side_info"]["Circles"]
        for e in best_example_evals
        if e["side_info"].get("Circles") is not None
    ]
    return np.array(circles) if circles else None


def fitness_fn(candidate, best_example_evals):
    code = candidate["code"]
    warm_start = extract_best_circles(best_example_evals)
    result = execute_code(code, TIMEOUT, warm_start)

    if result["success"]:
        circles = result["result"]["circles"]
        score = result["result"]["validation_details"]["sum_radii"]
        metrics = compute_multiple_metrics(result["result"]["all_scores"])
    else:
        circles = None
        score = 0.0
        metrics = {"sum_radii": 0.0}
        
    side_info = {
        "score": {"sum_radii": score}, 
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
            max_metric_calls=MAX_METRIC_CALLS,
            track_best_outputs=True,
            frontier_type="objective",
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(),
        refiner=RefinerConfig(),
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
