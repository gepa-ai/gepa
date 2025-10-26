#!/usr/bin/env python3
"""
Example 5: Circle Packing Optimization with GEPA

This example demonstrates using GEPA to evolve circle packing algorithms,
matching the classic OpenEvolve example. We pack 26 circles into a unit square
to maximize the sum of their radii.

This showcases:
- Evolving executable Python code
- Rich artifacts/feedback (execution errors, validation details, performance metrics)
- Single-workload optimization (the task itself is the workload)
- Timeout handling and error recovery

Target: Sum of radii ≈ 2.635 (AlphaEvolve paper result for n=26)
"""

import os
import time
import traceback
from typing import Any

import numpy as np


def validate_packing(centers: np.ndarray, radii: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """
    Validate that circles don't overlap and are inside the unit square.

    Returns:
        (is_valid, validation_details dict with boundary_violations, overlaps, stats)
    """
    n = centers.shape[0]
    validation_details = {
        "total_circles": n,
        "boundary_violations": [],
        "overlaps": [],
        "min_radius": float(np.min(radii)),
        "max_radius": float(np.max(radii)),
        "avg_radius": float(np.mean(radii)),
    }

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            violation = f"Circle {i} at ({x:.6f}, {y:.6f}) with radius {r:.6f} is outside unit square"
            validation_details["boundary_violations"].append(violation)

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow tiny numerical errors
                overlap = f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i] + radii[j]:.6f}"
                validation_details["overlaps"].append(overlap)

    is_valid = len(validation_details["boundary_violations"]) == 0 and len(validation_details["overlaps"]) == 0
    validation_details["is_valid"] = is_valid

    return is_valid, validation_details


def execute_packing_code(code: str, timeout_seconds: int = 60) -> tuple[Any, Any, float, str]:
    """
    Execute circle packing code in a controlled environment with timeout.

    Returns:
        (centers, radii, sum_radii, error_message)
        If error occurs, returns (None, None, 0.0, error_msg)
    """
    # Create a temporary module to execute the code
    exec_globals = {
        "np": np,
        "numpy": np,
        "__name__": "__main__",
    }

    try:
        # First, compile to check for syntax errors
        compiled = compile(code, "<evolved_code>", "exec")

        # Execute the code
        exec(compiled, exec_globals)

        # Check if required function exists
        if "run_packing" not in exec_globals:
            return (
                None,
                None,
                0.0,
                "Code must define a 'run_packing()' function that returns (centers, radii, sum_radii)",
            )

        # Run the packing function with timeout
        run_packing = exec_globals["run_packing"]

        # Simple timeout using alarm (Unix-like systems only)
        # For cross-platform, we'd use threading, but this is simpler for the example
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {timeout_seconds} seconds")

        # Set up timeout (note: signal.alarm only works on Unix)
        if hasattr(signal, "alarm"):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

        try:
            centers, radii, sum_radii = run_packing()
        finally:
            if hasattr(signal, "alarm"):
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)

        # Validate return types
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        return centers, radii, float(sum_radii), ""

    except SyntaxError as e:
        error_msg = f"Syntax Error: {e}\n{traceback.format_exc()}"
        return None, None, 0.0, error_msg
    except TimeoutError as e:
        error_msg = f"Timeout: {e}"
        return None, None, 0.0, error_msg
    except Exception as e:
        error_msg = f"Runtime Error: {e}\n{traceback.format_exc()}"
        return None, None, 0.0, error_msg


def evaluate_circle_packing(candidate: dict[str, str], batch: list[dict[str, Any]]) -> list[dict]:
    """
    Evaluate circle packing code.

    The 'batch' contains a single workload (the circle packing task itself).
    Each workload specifies the target value and other parameters.
    """
    results = []
    packing_code = candidate["packing_algorithm"]

    for workload in batch:
        target_value = workload.get("target_value", 2.635)  # AlphaEvolve result
        timeout = workload.get("timeout", 60)

        # Execute the code
        start_time = time.time()
        centers, radii, sum_radii, error_msg = execute_packing_code(packing_code, timeout)
        exec_time = time.time() - start_time

        # Prepare feedback artifacts
        artifacts = {
            "execution_time": f"{exec_time:.2f}s",
        }

        # Handle execution errors
        if error_msg:
            score = 0.0
            feedback = f"❌ Execution Failed:\n{error_msg}\n\n"
            feedback += "Common issues:\n"
            feedback += "- Syntax errors (check indentation, colons, parentheses)\n"
            feedback += "- Undefined variables or imports\n"
            feedback += "- Missing 'run_packing()' function\n"
            feedback += "- Function doesn't return (centers, radii, sum_radii)\n"

            artifacts["stderr"] = error_msg
            artifacts["failure_stage"] = "code_execution"
            artifacts["suggestion"] = "Fix syntax/runtime errors before attempting optimization"

        # Handle shape validation errors
        elif centers is None or radii is None:
            score = 0.0
            feedback = "❌ Function returned None values"
            artifacts["failure_stage"] = "invalid_return"

        elif centers.shape != (26, 2) or radii.shape != (26,):
            score = 0.0
            feedback = "❌ Invalid shapes:\n"
            feedback += "  Expected: centers=(26, 2), radii=(26,)\n"
            feedback += f"  Got: centers={centers.shape}, radii={radii.shape}\n"

            artifacts["stderr"] = f"Shape mismatch: centers={centers.shape}, radii={radii.shape}"
            artifacts["failure_stage"] = "shape_validation"
            artifacts["expected_shapes"] = "centers: (26, 2), radii: (26,)"

        # Validate geometric constraints
        else:
            valid, validation_details = validate_packing(centers, radii)
            target_ratio = sum_radii / target_value if valid else 0.0

            # Score: ratio achieved (0 if invalid, up to ~1.0 for perfect)
            score = target_ratio if valid else 0.0

            # Build feedback
            if not valid:
                feedback = "⚠️  Invalid Packing (Score: 0.0)\n\n"
                feedback += f"Sum of radii: {sum_radii:.6f} (would be {target_ratio:.2%} of target)\n"
                feedback += f"Target: {target_value}\n\n"
                feedback += "Validation failures:\n"

                if validation_details["boundary_violations"]:
                    feedback += f"\n🔴 Boundary Violations ({len(validation_details['boundary_violations'])}):\n"
                    # Show first few violations
                    for v in validation_details["boundary_violations"][:3]:
                        feedback += f"  - {v}\n"
                    if len(validation_details["boundary_violations"]) > 3:
                        feedback += f"  ... and {len(validation_details['boundary_violations']) - 3} more\n"

                    artifacts["boundary_violations"] = "\n".join(validation_details["boundary_violations"][:5])

                if validation_details["overlaps"]:
                    feedback += f"\n🔴 Circle Overlaps ({len(validation_details['overlaps'])}):\n"
                    for v in validation_details["overlaps"][:3]:
                        feedback += f"  - {v}\n"
                    if len(validation_details["overlaps"]) > 3:
                        feedback += f"  ... and {len(validation_details['overlaps']) - 3} more\n"

                    artifacts["overlap_violations"] = "\n".join(validation_details["overlaps"][:5])

                artifacts["failure_stage"] = "geometric_validation"
                artifacts["validation_report"] = (
                    f"Valid: False, Violations: {len(validation_details['boundary_violations'])} boundary, {len(validation_details['overlaps'])} overlaps"
                )

            else:
                # Valid packing!
                percentage = target_ratio * 100

                if target_ratio >= 0.95:
                    feedback = f"✅ Excellent packing! {percentage:.1f}% of target!\n\n"
                elif target_ratio >= 0.75:
                    feedback = f"✓ Good packing: {percentage:.1f}% of target\n\n"
                else:
                    feedback = f"○ Valid packing: {percentage:.1f}% of target\n\n"

                feedback += f"Sum of radii: {sum_radii:.6f} / {target_value}\n"
                feedback += "Radius stats:\n"
                feedback += f"  Min: {validation_details['min_radius']:.6f}\n"
                feedback += f"  Max: {validation_details['max_radius']:.6f}\n"
                feedback += f"  Avg: {validation_details['avg_radius']:.6f}\n"

                if target_ratio < 0.95:
                    feedback += "\n💡 To improve:\n"
                    feedback += "  - Try mathematical optimization (scipy.optimize)\n"
                    feedback += "  - Use better initial placements\n"
                    feedback += "  - Consider hexagonal packing patterns\n"

                artifacts["packing_summary"] = f"Sum of radii: {sum_radii:.6f}/{target_value} = {target_ratio:.4f}"
                artifacts["radius_stats"] = (
                    f"Min: {validation_details['min_radius']:.6f}, Max: {validation_details['max_radius']:.6f}, Avg: {validation_details['avg_radius']:.6f}"
                )

                if target_ratio >= 0.95:
                    artifacts["stdout"] = f"Excellent packing! Achieved {percentage:.1f}% of target value"

        results.append(
            {
                "score": score,
                "context_and_feedback": {
                    "inputs": f"Circle Packing Task: n=26 circles, target sum={target_value}",
                    "outputs": f"Sum of radii: {sum_radii:.6f} (ratio: {score:.4f})",
                    "feedback": feedback,
                    **artifacts,  # Include all artifacts in context
                },
            }
        )

    return results


# Initial seed algorithm (simple concentric rings)
SEED_ALGORITHM = '''
"""Constructor-based circle packing for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Place a large circle in the center
    centers[0] = [0.5, 0.5]
    
    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
    
    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]
    
    # Clip to ensure inside unit square
    centers = np.clip(centers, 0.01, 0.99)
    
    # Compute maximum valid radii
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

def compute_max_radii(centers):
    """Compute maximum possible radii without overlaps"""
    n = centers.shape[0]
    radii = np.ones(n)
    
    # Limit by distance to square borders
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

def run_packing():
    """Entry point - called by evaluator"""
    return construct_packing()
'''


def main():
    """Run circle packing evolution using GEPA."""

    from gepa import evolve

    # Single workload: the circle packing task
    workload = [
        {
            "target_value": 2.635,  # AlphaEvolve result
            "timeout": 60,
        }
    ]

    print("=" * 80)
    print("GEPA Circle Packing Evolution")
    print("=" * 80)
    print("\nTask: Pack 26 circles in a unit square to maximize sum of radii")
    print("Target: 2.635 (from AlphaEvolve paper)")
    print("\nStarting with simple concentric rings approach...")
    print("=" * 80)
    print()

    # Run GEPA evolution
    result = evolve(
        seed_candidate={"packing_algorithm": SEED_ALGORITHM},
        trainset=workload,
        evaluate=evaluate_circle_packing,
        reflection_prompt="""You are evolving a circle packing algorithm for n=26 circles in a unit square.

GOAL: Maximize the sum of circle radii while ensuring:
1. All circles stay inside the unit square (0 ≤ x±r ≤ 1, 0 ≤ y±r ≤ 1)
2. No circles overlap (distance ≥ r1 + r2)
3. Return exactly 26 circles with shape (26, 2) for centers and (26,) for radii

CURRENT ISSUES from feedback:
- Check validation failures (boundary violations, overlaps)
- Look at execution errors (syntax, runtime, timeouts)
- Note achieved ratio vs target (2.635)

OPTIMIZATION STRATEGIES:
1. **Better Initial Placement**:
   - Hexagonal/triangular packing patterns
   - Grid-based layouts with variable spacing
   - Corner and edge optimization
   - Concentric rings with careful spacing

2. **Mathematical Optimization**:
   - Use scipy.optimize.minimize with SLSQP method
   - Define objective: maximize sum of radii (or minimize negative sum)
   - Add constraints for non-overlap and boundaries
   - Start from good initial guess

3. **Iterative Refinement**:
   - Place circles greedily, then optimize
   - Iteratively adjust positions and radii
   - Use gradient descent or simulated annealing

4. **Hybrid Approaches**:
   - Construct initial layout, then optimize
   - Use different strategies for different regions
   - Combine multiple packing patterns

REQUIRED: Your code must define `run_packing()` that returns (centers, radii, sum_radii).

Provide only the improved Python code, no explanations.""",
        num_iterations=30,  # Can increase for better results
        minibatch_size=1,  # Single workload
        teacher_lm="openai/gpt-4o",
        random_seed=42,
        output_dir="./circle_packing_output",
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)

    best_code = result["best_candidate"]["packing_algorithm"]
    best_score = result["best_score"]

    print(f"\nBest Score: {best_score:.4f} (ratio of target)")
    print(f"Estimated Sum of Radii: {best_score * 2.635:.4f}")

    # Save best code
    output_file = "./circle_packing_output/best_packing_algorithm.py"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(best_code)
    print(f"\nBest algorithm saved to: {output_file}")

    # Try to visualize if matplotlib is available
    print("\n" + "=" * 80)
    print("Testing Best Algorithm:")
    print("=" * 80)

    try:
        centers, radii, sum_radii, error = execute_packing_code(best_code, timeout_seconds=60)

        if error:
            print(f"Error executing best code: {error}")
        else:
            valid, validation = validate_packing(centers, radii)
            print(f"\nSum of radii: {sum_radii:.6f}")
            print("Target: 2.635")
            print(f"Ratio: {sum_radii / 2.635:.4f} ({sum_radii / 2.635 * 100:.1f}%)")
            print(f"Valid: {valid}")

            if not valid:
                print(f"Boundary violations: {len(validation['boundary_violations'])}")
                print(f"Overlaps: {len(validation['overlaps'])}")

            # Try to visualize
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Circle

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

                # Draw circles
                for i, (center, radius) in enumerate(zip(centers, radii, strict=False)):
                    circle = Circle(center, radius, alpha=0.5, edgecolor="black", linewidth=1)
                    ax.add_patch(circle)

                ax.set_title(f"Best Circle Packing (sum={sum_radii:.4f}, ratio={sum_radii / 2.635:.4f})")
                plt.savefig("./circle_packing_output/best_packing_visualization.png", dpi=150, bbox_inches="tight")
                print("\nVisualization saved to: ./circle_packing_output/best_packing_visualization.png")

            except ImportError:
                print("\nNote: Install matplotlib to visualize the packing")

    except Exception as e:
        print(f"Error testing best algorithm: {e}")


if __name__ == "__main__":
    main()
