import io
import numpy as np
import traceback
import signal
from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple, Any, Sequence

from gepa.optimize_anything import SideInfo


# ============================================================================
# Circle Packing Evaluation
# ============================================================================


def evaluate_circles(circles, num_circles):
    """Evaluate a circle packing configuration."""
    sum_of_radii = np.sum(circles[:, 2])
    is_valid, _ = validate_packing(num_circles, circles[:, :2], circles[:, 2])
    score = sum_of_radii if is_valid else 0.0
    return score


def evaluate_code(code_string, num_circles):
    """Evaluate circle packing code."""
    try:
        results = execute_code(
            code_string,
            {
                "evaluate_circles": evaluate_circles,
                "num_circles": num_circles,
            },
            timeout=300,  # 5 minutes timeout
        )
        output = results.get("output", "")
        logs = results.get("logs", "")
        error = results.get("error", "")
        execution_context = results.get("results", {})

        score = execution_context.get("best_score", 0)
        if score == 0:
            error += "No best score found"

        best_circles = execution_context.get("best_circles", [])
        if len(best_circles) == 0:
            error += "No best circles found"
            circles = np.array([]).reshape(0, 3)
            is_valid = False
            validation_details = {"error": "No circles found"}
        else:
            circles = np.array(best_circles)
            is_valid, validation_details = validate_packing(
                num_circles, circles[:, :2], circles[:, 2]
            )

    except Exception as e:
        traceback.print_exc()
        score = 0
        output = ""
        logs = ""
        error = str(e) + "\n" + traceback.format_exc()
        execution_context = {}
        circles = None
        is_valid = False
        validation_details = None

    metrics = {"sum_of_radii": score}
    artifacts = {
        "output": output,
        "best_circles": circles.tolist() if circles is not None else [],
        "logs": logs,
        "error": error,
        "is valid": is_valid,
        "validation details": validation_details,
        "num_circles": num_circles,
    }

    evaluation_result = {
        "metrics": metrics,
        "artifacts": artifacts,
    }

    return evaluation_result


def create_fitness_function():
    """Create fitness function for Circle Packing optimization."""

    def fitness_fn(
        candidate: dict[str, str], batch: Sequence[Any], **kwargs
    ) -> list[tuple[float, Any, SideInfo]]:
        """
        Evaluate code candidate on batch of Circle Packing problems.
        """
        results = []
        for example in batch:
            num_circles = example["num_circles"]
            code = candidate["code"]
            result = evaluate_code(code, num_circles)
            score = result["metrics"]["sum_of_radii"]
            output = result.copy()
            output["code"] = code
            results += ((score, output, result),)
        return results

    return fitness_fn


# ============================================================================
# Code Execution Utilities
# ============================================================================


class TimeLimitError(Exception):
    """Raised when code execution exceeds the time limit."""


def _alarm_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeLimitError("Time Limit Exceeded")


def execute_code(code_string, global_vars=None, timeout=None):
    """Execute code and capture results with timeout support."""
    f_out = io.StringIO()
    f_err = io.StringIO()

    if global_vars is None:
        context = {"__name__": "__main__"}
    else:
        context = global_vars.copy()
        context["__name__"] = "__main__"

    error = ""

    # Set up timeout if specified (only works on Unix systems)
    old_handler = None
    if timeout is not None and hasattr(signal, "SIGALRM"):
        # Save the previous signal handler
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        # Set the alarm (in seconds) - use setitimer for more precision
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
        except AttributeError:
            # Fallback to alarm if setitimer not available
            signal.alarm(timeout)

    try:
        with redirect_stdout(f_out), redirect_stderr(f_err):
            exec(code_string, context)
    except TimeLimitError:
        error = f"TimeLimitError: Code execution exceeded {timeout} seconds. The code took too long to execute and was terminated."
    except Exception as e:
        error = str(e) + "\n" + traceback.format_exc()
    finally:
        # CRITICAL: Disable the alarm and restore previous handler
        if timeout is not None and hasattr(signal, "SIGALRM"):
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except AttributeError:
                signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    return {
        "output": f_out.getvalue(),
        "logs": f_err.getvalue(),
        "results": context,
        "error": error,
    }


# ============================================================================
# Circle Packing Validation
# ============================================================================


def validate_packing(
    n: int, centers: np.ndarray, radii: np.ndarray, atol: float = 1e-6
) -> Tuple[bool, dict[str, Any]]:
    """
    Validate that circles don't overlap and are inside the unit square.

    Note: We validate against unit square (perimeter=4 means width+height=2,
    so we use 1x1 square which satisfies this).

    Args:
        n: Expected number of circles
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n,) with radius of each circle
        atol: Absolute tolerance for numerical comparisons

    Returns:
        Tuple of (is_valid: bool, validation_details: dict)
    """
    validation_details = {
        "expected_circles": n,
        "actual_circles": centers.shape[0] if isinstance(centers, np.ndarray) else 0,
        "boundary_violations": [],
        "overlaps": [],
        "nan_detected": False,
        "negative_radii": [],
        "shape_errors": [],
    }

    # Convert to numpy arrays if needed
    if not isinstance(centers, np.ndarray):
        try:
            centers = np.array(centers)
        except Exception as e:
            validation_details["shape_errors"].append(
                f"Cannot convert centers to numpy array: {e}"
            )
            return False, validation_details

    if not isinstance(radii, np.ndarray):
        try:
            radii = np.array(radii)
        except Exception as e:
            validation_details["shape_errors"].append(
                f"Cannot convert radii to numpy array: {e}"
            )
            return False, validation_details

    # Check shapes
    if centers.shape != (n, 2):
        validation_details["shape_errors"].append(
            f"Centers shape incorrect. Expected ({n}, 2), got {centers.shape}"
        )
        return False, validation_details

    if radii.shape != (n,):
        validation_details["shape_errors"].append(
            f"Radii shape incorrect. Expected ({n},), got {radii.shape}"
        )
        return False, validation_details

    # Check for NaN values
    if np.isnan(centers).any():
        validation_details["nan_detected"] = True
        validation_details["shape_errors"].append(
            "NaN values detected in circle centers"
        )
        return False, validation_details

    if np.isnan(radii).any():
        validation_details["nan_detected"] = True
        validation_details["shape_errors"].append("NaN values detected in circle radii")
        return False, validation_details

    # Check for negative radii
    for i in range(n):
        if radii[i] < 0:
            validation_details["negative_radii"].append(
                f"Circle {i} has negative radius {radii[i]:.6f}"
            )

    if validation_details["negative_radii"]:
        return False, validation_details

    # Check if circles are inside the unit square (perimeter 4 = width+height=2, so 1x1 square)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            validation_details["boundary_violations"].append(
                f"Circle {i} at ({x:.6f}, {y:.6f}) with radius {r:.6f} is outside unit square"
            )

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:  # Allow for tiny numerical errors
                validation_details["overlaps"].append(
                    f"Circles {i} and {j} overlap: dist={dist:.6f}, r_sum={radii[i] + radii[j]:.6f}"
                )

    # Compute statistics
    validation_details["min_radius"] = float(np.min(radii))
    validation_details["max_radius"] = float(np.max(radii))
    validation_details["avg_radius"] = float(np.mean(radii))
    validation_details["std_radius"] = float(np.std(radii))

    # Valid if no violations
    is_valid = (
        len(validation_details["boundary_violations"]) == 0
        and len(validation_details["overlaps"]) == 0
        and len(validation_details["shape_errors"]) == 0
        and len(validation_details["negative_radii"]) == 0
        and not validation_details["nan_detected"]
    )

    return is_valid, validation_details
