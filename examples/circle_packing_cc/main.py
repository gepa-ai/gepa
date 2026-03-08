#!/usr/bin/env python3
"""
Optimize circle packing code using Claude Code as the reflection LM (seedless mode, no refiner).

CC writes code that packs 26 circles into a unit square, maximizing sum of radii.
Single-task search: no dataset, no valset.

Usage:
    uv run python -m examples.circle_packing_cc.main
"""

from pathlib import Path

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
    ReflectionConfig,
    optimize_anything,
)


def evaluate(candidate, opt_state: OptimizationState | None = None):
    """Evaluate a candidate circle packing code."""
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
    run_dir = str(Path(__file__).parent / "run_output")
    print(f"[main] Run dir: {run_dir}")
    print("[main] Starting optimize_anything (seedless + claude_code, no refiner)...")

    result = optimize_anything(
        seed_candidate=None,
        evaluator=evaluate,
        objective="Optimize circle packing code to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=run_dir,
                max_candidate_proposals=30,
                track_best_outputs=True,
                cache_evaluation=True,
                frontier_type="objective",
            ),
            reflection=ReflectionConfig(
                reflection_lm="claude_code",
            ),
        ),
    )

    print(f"[main] Optimization complete. Best candidate index: {result.best_idx}")
    print(f"[main] Best score: {result.best_score}")


if __name__ == "__main__":
    main()
