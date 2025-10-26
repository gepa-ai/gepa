#!/usr/bin/env python3
"""
Circle Packing with GEPA - Multi-N Workload Evolution

This example demonstrates using GEPA's evolve API to solve the circle packing problem
for multiple values of n simultaneously, allowing the system to learn insights across
different problem sizes.

The circle packing problem: Pack n circles into a unit square to maximize the sum of radii,
with no overlaps and all circles fully contained within the square.

We use the following n values as different workloads:
[7, 10, 21, 22, 26, 28, 29, 31, 32, 33, 52, 68, 99, 143, 216, 446, 992]

This allows GEPA to:
1. Learn general packing strategies that work across different n
2. Discover algorithmic approaches that scale well
3. Use insights from easier problems (small n) to solve harder ones (large n)
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np

# Add parent directory to path to import gepa
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.gepa.optimize import WorkloadResult, evolve

# Known best values for circle packing (from literature and AlphaEvolve paper)
# These are for reference only - NOT used in scoring
# The goal is to maximize sum of radii without explicit targets
KNOWN_BEST_VALUES = {
    7: 1.716,  # Estimated based on scaling
    10: 2.121,  # Estimated
    21: 3.182,  # Estimated
    22: 3.281,  # Estimated
    26: 2.635,  # AlphaEvolve paper (main benchmark)
    28: 3.589,  # Estimated
    29: 3.663,  # Estimated
    31: 3.798,  # Estimated
    32: 3.866,  # Estimated
    33: 3.931,  # Estimated
    52: 5.447,  # Estimated
    68: 6.651,  # Estimated
    99: 8.678,  # Estimated
    143: 11.236,  # Estimated
    216: 14.697,  # Estimated
    446: 22.876,  # Estimated (scaled from smaller values)
    992: 35.891,  # Estimated (scaled from smaller values)
}


class TimeoutError(Exception):
    """Custom timeout exception"""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(n: int, centers: np.ndarray, radii: np.ndarray, atol: float = 1e-6) -> tuple[bool, dict[str, Any]]:
    """
    Validate that circles don't overlap and are inside the unit square.

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
            validation_details["shape_errors"].append(f"Cannot convert centers to numpy array: {e}")
            return False, validation_details

    if not isinstance(radii, np.ndarray):
        try:
            radii = np.array(radii)
        except Exception as e:
            validation_details["shape_errors"].append(f"Cannot convert radii to numpy array: {e}")
            return False, validation_details

    # Check shapes
    if centers.shape != (n, 2):
        validation_details["shape_errors"].append(f"Centers shape incorrect. Expected ({n}, 2), got {centers.shape}")
        return False, validation_details

    if radii.shape != (n,):
        validation_details["shape_errors"].append(f"Radii shape incorrect. Expected ({n},), got {radii.shape}")
        return False, validation_details

    # Check for NaN values
    if np.isnan(centers).any():
        validation_details["nan_detected"] = True
        validation_details["shape_errors"].append("NaN values detected in circle centers")
        return False, validation_details

    if np.isnan(radii).any():
        validation_details["nan_detected"] = True
        validation_details["shape_errors"].append("NaN values detected in circle radii")
        return False, validation_details

    # Check for negative radii
    for i in range(n):
        if radii[i] < 0:
            validation_details["negative_radii"].append(f"Circle {i} has negative radius {radii[i]:.6f}")

    if validation_details["negative_radii"]:
        return False, validation_details

    # Check if circles are inside the unit square
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


def run_packing_with_timeout(code: str, n: int, timeout_seconds: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the packing code in a separate process with timeout.

    Args:
        code: The Python code defining construct_packing() function
        n: Number of circles to pack
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Tuple of (centers, radii, sum_radii) from the program

    Raises:
        TimeoutError: If execution exceeds timeout
        RuntimeError: If execution fails
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        # Write a complete script that executes the code
        script = f"""
import sys
import numpy as np
import traceback
import pickle

# The evolved code
{code}

try:
    # Call construct_packing with n
    centers, radii, sum_radii = construct_packing({n})
    
    # Save results
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii,
        'success': True
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
        
except Exception as e:
    # Save error
    results = {{
        'error': str(e),
        'traceback': traceback.format_exc(),
        'success': False
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
"""
        temp_file.write(script)
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen([sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                if not results.get("success", False):
                    error_msg = results.get("error", "Unknown error")
                    raise RuntimeError(f"Program execution failed: {error_msg}")

                return results["centers"], results["radii"], results["sum_radii"]
            else:
                raise RuntimeError(f"Results file not found. Exit code: {exit_code}")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate_circle_packing(candidate: dict[str, str], batch: list[int]) -> list[WorkloadResult]:
    """
    Evaluate a circle packing candidate on a batch of workloads (different n values).

    Args:
        candidate: Dict with 'packing_function' key containing the Python code
        batch: List of n values to test

    Returns:
        List of WorkloadResult dicts, one per workload
    """
    results = []

    # Extract the code
    code = candidate.get("packing_function", "")

    if not code.strip():
        # Empty code - return zeros
        for n in batch:
            results.append(
                {
                    "score": 0.0,
                    "context_and_feedback": {
                        "inputs": f"n={n}",
                        "outputs": "Empty code provided",
                        "feedback": "Error: No code provided in packing_function",
                        "n": n,
                        "valid": False,
                        "error": "Empty code",
                    },
                }
            )
        return results

    # Check that the code defines construct_packing
    if "def construct_packing" not in code:
        for n in batch:
            results.append(
                {
                    "score": 0.0,
                    "context_and_feedback": {
                        "inputs": f"n={n}",
                        "outputs": "Code does not define construct_packing function",
                        "feedback": "Error: Code must define a construct_packing(n) function that returns (centers, radii, sum_radii)",
                        "n": n,
                        "valid": False,
                        "error": "Missing construct_packing function",
                    },
                }
            )
        return results

    # Evaluate each workload
    for n in batch:
        start_time = time.time()

        try:
            # Determine timeout based on n (larger n needs more time)
            # if n <= 33:
            #     timeout = 60
            # elif n <= 100:
            #     timeout = 120
            # else:
            #     timeout = 180
            timeout = 600

            # Run the packing code
            centers, radii, reported_sum = run_packing_with_timeout(code, n, timeout_seconds=timeout)

            eval_time = time.time() - start_time

            # Validate the result
            is_valid, validation_details = validate_packing(n, centers, radii)

            if is_valid:
                # Calculate actual sum - this is our score
                actual_sum = float(np.sum(radii))
                score = actual_sum

                # Generate feedback encouraging improvement
                feedback = f"Valid packing with sum of radii = {actual_sum:.6f}. Try to increase this sum further."

                # Add radius statistics to feedback
                feedback += f"\nRadius stats - min: {validation_details['min_radius']:.6f}, max: {validation_details['max_radius']:.6f}, avg: {validation_details['avg_radius']:.6f}"

                results.append(
                    {
                        "score": score,
                        "context_and_feedback": {
                            "inputs": f"n={n} circles in unit square",
                            "outputs": f"Valid packing with sum_radii={actual_sum:.6f}",
                            "feedback": feedback,
                            "n": n,
                            "sum_radii": actual_sum,
                            "valid": True,
                            "eval_time": eval_time,
                            "validation_details": validation_details,
                        },
                    }
                )
            else:
                # Invalid packing
                error_summary = []
                if validation_details["shape_errors"]:
                    error_summary.append(f"{len(validation_details['shape_errors'])} shape errors")
                if validation_details["boundary_violations"]:
                    error_summary.append(f"{len(validation_details['boundary_violations'])} boundary violations")
                if validation_details["overlaps"]:
                    error_summary.append(f"{len(validation_details['overlaps'])} overlaps")
                if validation_details["negative_radii"]:
                    error_summary.append(f"{len(validation_details['negative_radii'])} negative radii")

                error_str = ", ".join(error_summary)

                feedback = f"Invalid packing for n={n}: {error_str}\n"

                # Add details for first few errors of each type
                if validation_details["shape_errors"]:
                    feedback += f"Shape errors: {validation_details['shape_errors'][0]}\n"
                if validation_details["boundary_violations"]:
                    feedback += f"Example boundary violation: {validation_details['boundary_violations'][0]}\n"
                if validation_details["overlaps"]:
                    feedback += f"Example overlap: {validation_details['overlaps'][0]}\n"

                results.append(
                    {
                        "score": 0.0,
                        "context_and_feedback": {
                            "inputs": f"n={n} circles in unit square",
                            "outputs": f"Invalid packing: {error_str}",
                            "feedback": feedback,
                            "n": n,
                            "valid": False,
                            "eval_time": eval_time,
                            "validation_details": validation_details,
                        },
                    }
                )

        except TimeoutError:
            eval_time = time.time() - start_time
            feedback = f"Timeout: Execution took longer than {timeout}s for n={n}. Consider using a faster approach to packing."

            results.append(
                {
                    "score": 0.0,
                    "context_and_feedback": {
                        "inputs": f"n={n} circles in unit square",
                        "outputs": f"Timeout after {timeout}s",
                        "feedback": feedback,
                        "n": n,
                        "valid": False,
                        "error": "timeout",
                        "eval_time": eval_time,
                    },
                }
            )

        except Exception as e:
            eval_time = time.time() - start_time
            error_msg = str(e)
            error_trace = traceback.format_exc()

            feedback = f"Error evaluating n={n}: {error_msg}\n"
            if "syntax" in error_msg.lower():
                feedback += "Check for Python syntax errors in your code."
            elif "import" in error_msg.lower():
                feedback += "Check that all imports are available (numpy, scipy, etc.)."
            elif "name" in error_msg.lower():
                feedback += "Check for undefined variables or functions."
            else:
                feedback += "Check the error traceback for details."

            results.append(
                {
                    "score": 0.0,
                    "context_and_feedback": {
                        "inputs": f"n={n}",
                        "outputs": f"Error: {error_msg}",
                        "feedback": feedback,
                        "n": n,
                        "valid": False,
                        "error": error_msg,
                        "traceback": error_trace,
                        "eval_time": eval_time,
                    },
                }
            )

    return results


# Initial seed program - simple constructor-based approach
INITIAL_PACKING_CODE = '''
import numpy as np

def construct_packing(n):
    """
    Construct a circle packing for n circles in a unit square.
    
    Args:
        n: Number of circles to pack
        
    Returns:
        Tuple of (centers, radii, sum_of_radii)
        - centers: np.array of shape (n, 2) with (x, y) coordinates
        - radii: np.array of shape (n,) with radius of each circle
        - sum_of_radii: Sum of all radii
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Simple grid-based placement
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            centers[idx] = [(i + 1) * spacing, (j + 1) * spacing]
            idx += 1
            if idx >= n:
                break
    
    # Compute maximum radii without overlaps
    for i in range(n):
        # Start with distance to borders
        x, y = centers[i]
        max_r = min(x, y, 1 - x, 1 - y)
        
        # Limit by distance to other circles
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                max_r = min(max_r, dist / 2.0)
        
        radii[i] = max(0.001, max_r * 0.95)  # 95% to ensure no numerical issues
    
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
'''


# Custom reflection prompt for circle packing
REFLECTION_PROMPT = """You are an expert mathematician and computational geometry specialist. Your task is to improve a Python function that packs n circles into a unit square (0,0) to (1,1), maximizing the sum of their radii.

The function `construct_packing(n)` must return a tuple of (centers, radii, sum_radii) where:
- centers: numpy array of shape (n, 2) with (x, y) coordinates
- radii: numpy array of shape (n,) with the radius of each circle
- sum_radii: float, the sum of all radii

Available packages:
- numpy (imported as np)
- scipy (all submodules)
- matplotlib (for visualization, if needed)

Constraints (CRITICAL):
1. All circles must be fully inside the unit square: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1 for each circle
2. No circles may overlap: distance between any two centers >= sum of their radii
3. All radii must be positive
4. Must return exactly n circles

Below are traces from running your current implementation on various values of n. Learn from successes and failures across different problem sizes.

{examples}

Key insights for improvement:
1. **Algorithmic approaches**: Consider constructor-based (explicit placement), optimization-based, physics-based (force simulation), or hybrid approaches
2. **Geometric patterns**: Hexagonal packing is known to be dense for infinite plane circle packing. Consider how edge effects in a square container might require adaptations
3. **Circle sizes**: Variable-sized circles may allow better space utilization than uniform sizes
4. **Edge effects**: Corners and edges of the square constrain circle placement differently than the interior
5. **Scalability**: Your solution should work efficiently for both small (n<30) and large (n>100) values
6. **Optimization**: If using optimization-based approaches, good initial placement can significantly improve results

Based on the examples above, improve the construct_packing function to maximize the sum of radii across all n values.

RESPOND ONLY WITH THE IMPROVED PYTHON CODE. Do not include explanations or markdown.
"""


def main():
    """Run circle packing optimization with GEPA."""

    print("=" * 80)
    print("Circle Packing Multi-N Optimization with GEPA")
    print("=" * 80)
    print()
    print("Problem: Pack n circles in a unit square to maximize sum of radii")
    print("Approach: Evolve a general construct_packing(n) function")
    print()
    print("Training workloads (n values):")

    # Define trainset: list of n values to optimize for
    trainset = [7, 10, 21, 22, 26, 28, 29, 31, 32, 33, 52, 68, 99, 143, 216, 446, 992]

    for n in trainset:
        best_known = KNOWN_BEST_VALUES.get(n, "unknown")
        print(f"  n={n:4d}  (best known from literature: {best_known})")
    print()
    print("Note: Best known values are for reference only - NOT used in scoring.")
    print("Goal: Maximize sum of radii for each n through evolution.")
    print()

    # Define seed candidate
    seed_candidate = {"packing_function": INITIAL_PACKING_CODE}

    print("Starting evolution...")
    print()

    # Run GEPA optimization
    result = evolve(
        seed_candidate=seed_candidate,
        trainset=trainset,
        evaluate=evaluate_circle_packing,
        reflection_prompt=REFLECTION_PROMPT,
        failure_score=0.0,
        num_iterations=100,
        minibatch_size=5,  # Evaluate on 5 random n values per iteration
        teacher_lm="anthropic/claude-3.7-sonnet",  # Use Claude Sonnet (best from configs)
        random_seed=42,
        output_dir="./circle_packing_output",
        verbose=True,
        num_threads=3,  # Parallel evaluation of different n values
        minibatch_full_eval_steps=10,  # Every 10 steps, evaluate on all n values
    )

    print()
    print("=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    print()
    print(f"Best aggregate score: {result['best_score']:.6f}")
    print(f"Results saved to: {result['output_dir']}")
    print()
    print("Best packing function:")
    print("-" * 80)
    print(result["best_candidate"]["packing_function"][:500] + "...")
    print("-" * 80)

    # Evaluate best candidate on all n values for final report
    print()
    print("Final evaluation on all n values:")
    print()
    final_results = evaluate_circle_packing(result["best_candidate"], trainset)

    print(f"{'n':>4} | {'Score':>8} | {'Sum Radii':>10} | {'Target':>10} | {'Status'}")
    print("-" * 60)
    for i, n in enumerate(trainset):
        res = final_results[i]
        score = res["score"]
        ctx = res["context_and_feedback"]
        sum_r = ctx.get("sum_radii", 0.0)
        target = ctx.get("target", 0.0)
        status = "✓ Valid" if ctx.get("valid", False) else "✗ Invalid"
        print(f"{n:4d} | {score:8.4f} | {sum_r:10.6f} | {target:10.3f} | {status}")

    print()
    print("Evolution complete! Check output directory for detailed results.")


if __name__ == "__main__":
    main()
