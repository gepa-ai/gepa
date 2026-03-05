#!/usr/bin/env python3
from examples.circle_packing.utils import (
    CIRCLE_PACKING_BACKGROUND,
    compute_multiple_metrics,
    execute_code,
    extract_best_circles,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    OptimizationState,
    RefinerConfig,
    ReflectionConfig,
    optimize_anything,
)

# Seed code from ShinkaEvolve
SEED_CODE = '''
import numpy as np

def main(timeout, current_best_solution):
    """Circle packing: returns dict with 'circles' (n,3) and 'all_scores'."""
    n = 26

    if current_best_solution is not None:
        circles = current_best_solution.copy()
    else:
        centers = np.zeros((n, 2))
        centers[0] = [0.5, 0.5]

        # Ring of 8 around center
        angles = 2 * np.pi * np.arange(8) / 8
        centers[1:9] = np.column_stack([0.5 + 0.3 * np.cos(angles), 0.5 + 0.3 * np.sin(angles)])

        # Outer ring for remaining 17
        angles = 2 * np.pi * np.arange(17) / 17
        centers[9:] = np.column_stack([0.5 + 0.7 * np.cos(angles), 0.5 + 0.7 * np.sin(angles)])

        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        circles = np.column_stack([centers, radii])

    return {'circles': circles, 'all_scores': [float(circles[:, 2].sum())]}


def compute_max_radii(centers):
    """Compute maximum radii that don't overlap and stay in unit square."""
    n = len(centers)
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
'''



def evaluate(candidate, opt_state: OptimizationState | None = None):
    warm_start = extract_best_circles(opt_state)
    result = execute_code(candidate, 600, warm_start)

    circles = result["circles"]
    score = result["validation_details"]["sum_radii"]
    metrics = compute_multiple_metrics(result["all_scores"])

    side_info = {
        "scores": {"sum_radii": score},
        "metrics": metrics,
        "code": candidate,
        "circles": circles,
        "stdout": result.get("stdout", ""),
        "error": result.get("error"),
        "traceback": result.get("traceback"),
        "validation_details": result.get("validation_details"),
    }

    return score, side_info


def main():
    optimize_anything(
        seed_candidate=SEED_CODE,
        evaluator=evaluate,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir="outputs/circle_packing",
                max_metric_calls=150,
                track_best_outputs=True,
                cache_evaluation=True,
                frontier_type="objective",
            ),
            reflection=ReflectionConfig(reflection_lm="openai/gpt-5"),
            refiner=RefinerConfig(),  # A refiner LLM will try to improve the candidate based on the evaluation feedback.
        ),
        objective="Optimize circle packing code to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
    )


if __name__ == "__main__":
    main()
