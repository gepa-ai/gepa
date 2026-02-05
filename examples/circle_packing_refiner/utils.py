#!/usr/bin/env python3
"""Utilities for circle packing: code execution, validation, and seed templates."""

import time
import traceback as tb
from typing import Any

import numpy as np

from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode


def execute_code(
    code: str,
    timeout: int,
    current_best_solution: Any = None,
    num_circles: int = 26,
) -> dict:
    """Execute code in subprocess and validate the circle packing result."""
    start_time = time.time()

    result = _execute_code(
        code=code,
        timeout=timeout,
        mode=ExecutionMode.SUBPROCESS,
        entry_point="main",
        entry_point_args=(),
        entry_point_kwargs={"timeout": timeout, "current_best_solution": current_best_solution},
    )

    base = {
        "execution_time": time.time() - start_time,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    def fail(error: str, **extra) -> dict:
        return {"success": False, "error": error, "traceback": "", **base, **extra}

    if not result.success:
        return fail(result.error or "Execution failed", traceback=result.traceback)

    main_result = result.variables.get("__return__")
    if not isinstance(main_result, dict):
        return fail(f"main() must return a dict, got {type(main_result).__name__}")
    if "circles" not in main_result:
        return fail("main() return dict must contain 'circles' key")
    if "all_scores" not in main_result:
        return fail("main() return dict must contain 'all_scores' key")

    try:
        circles = np.array(main_result["circles"])
        is_valid, details = validate_packing(num_circles, circles)
        main_result["validation_details"] = details
    except Exception as e:
        return fail(str(e), traceback=tb.format_exc())

    if is_valid:
        return {"success": True, "result": main_result, **base}

    # Build error message from validation details
    errors = [msg for cond, msg in [
        (details["shape_errors"], f"Shape: {details['shape_errors']}"),
        (details["boundary_violations"], f"{len(details['boundary_violations'])} boundary violations"),
        (details["overlaps"], f"{len(details['overlaps'])} overlaps"),
        (details["negative_radii"], f"{len(details['negative_radii'])} negative radii"),
    ] if cond]

    return fail("Validation failed: " + "; ".join(errors), result=main_result, validation_details=details)


def validate_packing(n: int, circles: np.ndarray, atol: float = 1e-6) -> tuple[bool, dict[str, Any]]:
    """Validate circles don't overlap and stay inside the unit square."""
    details = {
        "expected_circles": n,
        "actual_circles": circles.shape[0],
        "boundary_violations": [],
        "overlaps": [],
        "nan_detected": False,
        "negative_radii": [],
        "shape_errors": [],
    }

    if circles.shape != (n, 3):
        details["shape_errors"].append(f"Expected ({n}, 3), got {circles.shape}")
        return False, details

    if np.isnan(circles).any():
        details["nan_detected"] = True
        details["shape_errors"].append("NaN values detected")
        return False, details

    centers, radii = circles[:, :2], circles[:, 2]

    # Check negative radii
    neg_mask = radii < 0
    if neg_mask.any():
        details["negative_radii"] = [
            f"Circle {i} has negative radius {radii[i]:.6f}" for i in np.where(neg_mask)[0]
        ]
        return False, details

    # Check boundary violations (vectorized)
    out_left = centers[:, 0] - radii < -atol
    out_right = centers[:, 0] + radii > 1 + atol
    out_bottom = centers[:, 1] - radii < -atol
    out_top = centers[:, 1] + radii > 1 + atol
    for i in np.where(out_left | out_right | out_bottom | out_top)[0]:
        x, y, r = circles[i]
        details["boundary_violations"].append(f"Circle {i} at ({x:.6f}, {y:.6f}) r={r:.6f} outside unit square")

    # Check overlaps (vectorized distance computation)
    dists = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    r_sums = radii[:, None] + radii[None, :]
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < r_sums[i, j] - atol:
                details["overlaps"].append(
                    f"Circles {i} and {j} overlap: dist={dists[i,j]:.6f}, r_sum={r_sums[i,j]:.6f}"
                )

    # Statistics
    details.update(
        min_radius=float(radii.min()),
        max_radius=float(radii.max()),
        avg_radius=float(radii.mean()),
        sum_radii=float(radii.sum()),
    )

    is_valid = not (details["boundary_violations"] or details["overlaps"] or
                    details["shape_errors"] or details["negative_radii"] or details["nan_detected"])
    return is_valid, details


# =============================================================================
# SEED CODE TEMPLATES
# =============================================================================

SEED_CODE1 = '''
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


SEED_CODE2 = '''
import numpy as np

def main(timeout, current_best_solution):
    # Fit 26 circles into a 6x6 grid (radius = 1/12)
    r = 1.0 / 12.0
    i = np.arange(26)
    circles = np.column_stack((r + (i % 6) * 2 * r, r + (i // 6) * 2 * r, np.full(26, r)))
    return {'circles': circles, 'all_scores': [np.sum(circles[:, 2])]}
'''



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
3. Your code should run in <550 seconds. Otherwise, the score will be 0.

INNOVATION STRATEGIES:
1. **Algorithmic diversity**: Physics-based, optimization-based, geometric, hybrid, meta-heuristics
2. **Geometric insights**: Hexagonal patterns, corner utilization, variable radii
3. **Optimization techniques**: Multiple restarts, hierarchical approaches, gradient-free methods
4. **Hyperparameter auto-tuning**: Use optuna/hyperopt to find best parameters automatically
5. Imagine you have all the packages available (optuna, scipy, etc.) in the environment already and freely explore any of the packages you need.

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