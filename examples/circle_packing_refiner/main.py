#!/usr/bin/env python3
import numpy as np

from examples.circle_packing_refiner.utils import (
    execute_code,
    SEED_CODE1,
    SEED_CODE2,
    compute_multiple_metrics,
    CIRCLE_PACKING_BACKGROUND,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    RefinerConfig,
    optimize_anything,
)

# Constants
MAX_METRIC_CALLS = 150
NUM_CIRCLES = 26
LLM_MODEL = "openai/gpt-5.1"
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
        scores = compute_multiple_metrics(result["result"]["all_scores"])
    else:
        circles = None
        score = 0.0
        scores = {"sum_radii": 0.0}  # NO ERROR? (for now, it's not validated because it can't beat the parent)
        # TODO: return all the different metrics
        # import sys
        # print("=" * 50, file=sys.stderr, flush=True)
        # print("FAILURE:", result.get("error"), file=sys.stderr, flush=True)
        # print("Traceback:", result.get("traceback"), file=sys.stderr, flush=True)
        # print("Validation:", result.get("validation_details"), file=sys.stderr, flush=True)
        # print("=" * 50, file=sys.stderr, flush=True)

    side_info = {
        "scores": scores,  # lowercase for multi-objective support
        "Code": code,
        "Circles": circles,
        "Stdout": result.get("stdout", ""),
        "Error": result.get("error"),
        "Traceback": result.get("traceback"),
        "Validation Details": result.get("validation_details"),
    }

    return score, side_info


def main():
    log_dir = f"outputs/circle_packing"

    # 1. clear the log directory
    import os
    import shutil
    if os.path.exists(log_dir):
        os.remove(f"{log_dir}/gepa_state.bin")
        shutil.rmtree(f"{log_dir}/fitness_cache")
        shutil.rmtree(f"{log_dir}/generated_best_outputs_valset")
    os.makedirs(log_dir, exist_ok=True)

    print("Circle Packing Evolution with RefinerConfig")
    print(f"LLM Model: {LLM_MODEL} | Problem size: N={NUM_CIRCLES} | Max metric calls: {MAX_METRIC_CALLS} | Log directory: {log_dir}")

    seed_candidate = {
        "code": SEED_CODE2, 
    }

    # GEPA config with RefinerConfig
    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=MAX_METRIC_CALLS,
            track_best_outputs=True,
            frontier_type="objective",
            cache_evaluation=True,
            best_example_evals_k=10, 
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm=LLM_MODEL,
        ),
        refiner=RefinerConfig(
            refiner_lm=LLM_MODEL,
        ),
    )

    # Run optimization
    print("\n" + "=" * 70)
    print("Running GEPA Optimization with RefinerConfig")
    print("=" * 70 + "\n")

    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        config=gepa_config,
        objective="Optimize circle packing code to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
    )

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print("Results:", result)


if __name__ == "__main__":
    main()
