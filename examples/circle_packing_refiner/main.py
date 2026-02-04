#!/usr/bin/env python3
"""
Circle Packing Evolution with RefinerConfig.

Problem: Pack N circles inside a unit square [0,1]x[0,1] to maximize sum of radii.

Optimizes the circle packing algorithm (code) with automatic refinement.
The refiner uses the default prompt (populated with objective/background).

Uses RefinerConfig for automatic refinement (adapter handles refinement loop).
Uses example_best_evals for warm-start (adapter tracks per-example history).
"""

from typing import Any

from examples.circle_packing.utils import (
    execute_code,
    SEED_CODE,
)
from examples.circle_packing.llms import CIRCLE_PACKING_BACKGROUND
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    RefinerConfig,
    SideInfo,
    optimize_anything,
)

# Constants
MAX_METRIC_CALLS = 150
NUM_CIRCLES = 26
LLM_MODEL = "openai/gpt-5.1"
TIMEOUT = 600  # 10 minutes, same as OpenEvolve

CIRCLE_PACKING_BACKGROUND = """
Make BREAKTHROUGH improvements by trying fundamentally different approaches.

Pack 26 non-overlapping circles inside a UNIT SQUARE [0,1] x [0,1].

SCORING: Sum of all circle radii (higher is better!)

CRITICAL CODE FORMAT:
- Function name MUST be: `def main(timeout, current_best_solution):`
- `current_best_solutions` is a list of numpy arrays of shape (26, 3) or None.
- Return a dictionary with:
    - 'circles': numpy array shape (26, 3) where each row is (x, y, radius)
    - 'all_scores': list of floats (even if just one score)

CRITICAL CONSTRAINTS:
1. All circles fully inside [0,1]×[0,1]: 0 ≤ x-r, x+r ≤ 1 and 0 ≤ y-r, y+r ≤ 1
2. No overlaps: distance between centers ≥ sum of radii

NOTES on the execution of your proposal:
- 

INNOVATION STRATEGIES:
1. **Algorithmic diversity**: Physics-based, optimization-based, geometric, hybrid, meta-heuristics
2. **Geometric insights**: Hexagonal patterns, corner utilization, variable radii
3. **Optimization techniques**: Multiple restarts, hierarchical approaches, gradient-free methods
4. **Hyperparameter auto-tuning**: Use optuna/hyperopt to find best parameters automatically
5. Imagine you have all the packages available in the environment already and freely explore any of the packages you need.

ANALYSIS STRATEGY:
1. If scores plateau → try fundamentally different algorithm
2. If errors persist → address root cause, don't just patch
3. The refiner LLM will handle the refinement process using the `refiner_prompt`, so you focus on making a big leap in the global strategy.

OUTPUT REQUIREMENTS:
- Return ONLY executable Python code (no markdown, no explanations)
- Focus on BREAKTHROUGH ideas, not incremental tweaks
"""



def compute_multiple_metrics(all_scores: list[float]) -> dict[str, float]:
    """Compute various metrics from score history."""
    alpha_fixed = 0.1
    ema_fixed = all_scores[0]
    for s in all_scores[1:]:
        ema_fixed = alpha_fixed * s + (1 - alpha_fixed) * ema_fixed

    alpha_adaptive = 2.0 / (len(all_scores) + 1)
    ema_adaptive = all_scores[0]
    for s in all_scores[1:]:
        ema_adaptive = alpha_adaptive * s + (1 - alpha_adaptive) * ema_adaptive

    return {
        "max_score": max(all_scores),
        "mean_score": sum(all_scores) / len(all_scores),
        "ema_score_fixed": ema_fixed,
        "ema_score_adaptive": ema_adaptive,
    }


def fitness_fn(
    candidate: dict[str, str],
    example_best_evals: list[dict] | None = None,
) -> tuple[float, Any, SideInfo]:
    """Evaluate code candidate."""
    code = candidate["code"]
    warm_start = example_best_evals[0]["side_info"].get("circles") if example_best_evals else None

    result = execute_code(code, TIMEOUT, warm_start)

    success = result["success"]
    if success:
        circles = result["result"]["circles"]
        score = result["result"]["validation_details"]["sum_radii"]
        scores = compute_multiple_metrics(result["result"]["all_scores"])
    else:
        circles = None
        score = 0.0
        scores = {"sum_radii": 0.0}

    side_info = {
        "scores": scores,
        "Code": code,
        "circles": circles,
        "Stdout": result.get("stdout", ""),
        "Error": result.get("error") if not success else None,
        "Traceback": result.get("traceback") if not success else None,
        "Validation Details": result.get("validation_details") if not success else None,
    }

    return score, side_info, side_info


def main():
    log_dir = f"outputs/circle_packing"

    print("Circle Packing Evolution with RefinerConfig")
    print(f"LLM Model: {LLM_MODEL} | Problem size: N={NUM_CIRCLES} | Max metric calls: {MAX_METRIC_CALLS} | Log directory: {log_dir}")

    seed_candidate = {
        "code": SEED_CODE,
    }

    # GEPA config with RefinerConfig
    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=MAX_METRIC_CALLS,
            track_best_outputs=True,
            frontier_type="objective",
            cache_evaluation=True,
            example_best_evals_k=10, 
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm=LLM_MODEL,
        ),
        refiner=RefinerConfig(
            refiner_lm=LLM_MODEL,
            max_refinements=2,
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
