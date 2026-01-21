#!/usr/bin/env python3
"""
Utilities for circle packing experiments.

Contains:
- execute_code: Clean subprocess-based code execution with timeout
- validate_packing: Validate circle packing constraints
- BASELINE_CODE_TEMPLATE: Seed code for optimization
- create_circle_packing_dataset: Create DSPy dataset
"""

import time
from typing import Any, Tuple

import numpy as np
import dspy

from gepa.utils.code_execution import (
    execute_code as _execute_code,
    ExecutionMode,
)


# =============================================================================
# CODE EXECUTION
# =============================================================================


def execute_code(
    code: str,
    timeout: int,
    current_best_solution: Any = None,
    num_circles: int = 26,
) -> dict:
    """
    Execute standalone code candidate in isolated subprocess with validation.

    The code must define a main(timeout, current_best_solution) function that returns
    a dict with 'circles' and 'all_scores' keys.

    Args:
        code: Python code string defining main(timeout, current_best_solution) -> dict
        timeout: Time limit in seconds (passed to code AND enforced on subprocess)
        current_best_solution: Best solution found so far (numpy array or None)
        num_circles: Expected number of circles for validation

    Returns:
        dict with:
            - success: bool (True only if execution AND validation pass)
            - result: dict with 'circles', 'all_scores', 'validation_details' if successful
            - error: str if failed
            - traceback: str if failed
            - execution_time: float
            - stdout: str (prints from code)
            - stderr: str (warnings/logs)
    """
    start_time = time.time()

    # Execute code using shared utility with entry point
    result = _execute_code(
        code=code,
        timeout=timeout,
        mode=ExecutionMode.SUBPROCESS,
        entry_point="main",
        entry_point_args=(),
        entry_point_kwargs={
            "timeout": timeout,
            "current_best_solution": current_best_solution,
        },
    )

    execution_time = time.time() - start_time

    # Handle execution failure
    if not result.success:
        return {
            "success": False,
            "error": result.error or "Execution failed",
            "traceback": result.traceback,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # Get main() return value
    main_result = result.variables.get("__return__")

    # Validate result has required keys
    if not isinstance(main_result, dict):
        return {
            "success": False,
            "error": f"main() must return a dict, got {type(main_result).__name__}",
            "traceback": "",
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    if "circles" not in main_result:
        return {
            "success": False,
            "error": "main() return dict must contain 'circles' key",
            "traceback": "",
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    if "all_scores" not in main_result:
        return {
            "success": False,
            "error": "main() return dict must contain 'all_scores' key",
            "traceback": "",
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # Validate the packing (wrapped in try-except to match original subprocess behavior)
    try:
        circles = np.array(main_result["circles"])
        is_valid, validation_details = validate_packing(num_circles, circles)
        main_result["validation_details"] = validation_details
    except Exception as e:
        import traceback as tb
        return {
            "success": False,
            "error": str(e),
            "traceback": tb.format_exc(),
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    if is_valid:
        return {
            "success": True,
            "result": main_result,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    else:
        # Build error message from validation details
        errors = []
        if validation_details["shape_errors"]:
            errors.append(f"Shape: {validation_details['shape_errors']}")
        if validation_details["boundary_violations"]:
            errors.append(f"{len(validation_details['boundary_violations'])} boundary violations")
        if validation_details["overlaps"]:
            errors.append(f"{len(validation_details['overlaps'])} overlaps")
        if validation_details["negative_radii"]:
            errors.append(f"{len(validation_details['negative_radii'])} negative radii")

        return {
            "success": False,
            "error": "Validation failed: " + "; ".join(errors),
            "result": main_result,  # Still include result for debugging
            "validation_details": validation_details,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


# =============================================================================
# VALIDATION
# =============================================================================


def validate_packing(
    n: int, circles: np.ndarray, atol: float = 1e-6
) -> Tuple[bool, dict[str, Any]]:
    """
    Validate that circles don't overlap and are inside the unit square.

    Args:
        n: Expected number of circles
        circles: np.array of shape (n, 3) with (x, y, r) for each circle
        atol: Absolute tolerance for numerical comparisons

    Returns:
        Tuple of (is_valid: bool, validation_details: dict)
    """
    validation_details = {
        "expected_circles": n,
        "actual_circles": circles.shape[0],
        "boundary_violations": [],
        "overlaps": [],
        "nan_detected": False,
        "negative_radii": [],
        "shape_errors": [],
    }

    # Check shape
    if circles.shape != (n, 3):
        validation_details["shape_errors"].append(
            f"Circles shape incorrect. Expected ({n}, 3), got {circles.shape}"
        )
        return False, validation_details

    # Check for NaN values
    if np.isnan(circles).any():
        validation_details["nan_detected"] = True
        validation_details["shape_errors"].append("NaN values detected in circles")
        return False, validation_details

    centers = circles[:, :2]
    radii = circles[:, 2]

    # Check for negative radii
    for i in range(n):
        if radii[i] < 0:
            validation_details["negative_radii"].append(
                f"Circle {i} has negative radius {radii[i]:.6f}"
            )

    if validation_details["negative_radii"]:
        return False, validation_details

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            validation_details["boundary_violations"].append(
                f"Circle {i} at ({x:.6f}, {y:.6f}) with r={r:.6f} outside unit square"
            )

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                validation_details["overlaps"].append(
                    f"Circles {i} and {j} overlap: dist={dist:.6f}, r_sum={radii[i] + radii[j]:.6f}"
                )

    # Compute statistics
    validation_details["min_radius"] = float(np.min(radii))
    validation_details["max_radius"] = float(np.max(radii))
    validation_details["avg_radius"] = float(np.mean(radii))
    validation_details["sum_radii"] = float(np.sum(radii))

    # Valid if no violations
    is_valid = (
        len(validation_details["boundary_violations"]) == 0
        and len(validation_details["overlaps"]) == 0
        and len(validation_details["shape_errors"]) == 0
        and len(validation_details["negative_radii"]) == 0
        and not validation_details["nan_detected"]
    )

    return is_valid, validation_details


# =============================================================================
# BASELINE CODE TEMPLATE
# =============================================================================

BASELINE_CODE_TEMPLATE = '''
import numpy as np

def main(timeout, current_best_solution):
    """
    Circle packing optimization.

    Args:
        timeout: Time budget in seconds
        current_best_solution: Previous best circles array (n, 3) or None

    Returns:
        dict with 'circles' (n, 3) array and 'all_scores' list
    """
    n = 26

    # Use current_best_solution if provided, otherwise start fresh
    if current_best_solution is not None:
        circles = current_best_solution.copy()
    else:
        # Simple initial placement
        centers = np.zeros((n, 2))

        # Center circle
        centers[0] = [0.5, 0.5]

        # Ring of 8 around center
        for i in range(min(8, n - 1)):
            angle = 2 * np.pi * i / 8
            centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

        # Outer ring for remaining
        if n > 9:
            remaining = n - 9
            for i in range(remaining):
                angle = 2 * np.pi * i / remaining
                centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        circles = np.hstack([centers, radii.reshape(-1, 1)])

    score = float(np.sum(circles[:, 2]))
    return {'circles': circles, 'all_scores': [score]}


def compute_max_radii(centers):
    """Compute maximum radii that don't overlap and stay in unit square."""
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit by distance to borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Limit by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
'''


# =============================================================================
# DATASET
# =============================================================================


def create_circle_packing_dataset():
    """Create dataset for circle packing problem (N=26)."""
    description = """Circle Packing Problem (N=26)

Pack 26 non-overlapping circles inside a UNIT SQUARE [0,1] x [0,1].
Goal: MAXIMIZE the sum of all circle radii.

CONSTRAINTS:
1. All circles must be non-overlapping: distance between centers >= r1 + r2
2. All circles must be fully inside [0,1] x [0,1]
3. All radii must be non-negative

Your code must define:
    def main(timeout, current_best_solution) -> dict

Args:
    timeout: Time budget in seconds
    current_best_solution: Previous best (n,3) array or None

Returns:
    {'circles': np.ndarray (n,3), 'all_scores': list[float]}

SCORING: Sum of all circle radii (higher is better!)
"""

    example = dspy.Example(
        {
            "problem_description": description,
            "baseline_code": BASELINE_CODE_TEMPLATE,
        }
    ).with_inputs("problem_description", "baseline_code")

    return [example]
