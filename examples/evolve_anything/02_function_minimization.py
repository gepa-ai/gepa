#!/usr/bin/env python3
"""
Example 2: Evolving Function Minimization Algorithms

This example demonstrates using GEPA to evolve optimization algorithms themselves.
Starting from random search, GEPA discovers sophisticated algorithms like simulated
annealing, gradient-free optimization, and adaptive techniques.

Inspired by: openevolve/examples/function_minimization/
"""

import math
import random
from typing import Any


# Test functions for optimization (standard benchmark functions)
def rastrigin(x):
    """Rastrigin function: Many local minima, global minimum at origin."""
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock(x):
    """Rosenbrock function: Narrow valley, global minimum at (1, 1, ...)."""
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def sphere(x):
    """Sphere function: Simple convex function, global minimum at origin."""
    return sum(xi**2 for xi in x)


def ackley(x):
    """Ackley function: Nearly flat outer region, deep hole at origin."""
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e


# Benchmark configurations
BENCHMARKS = [
    {"func": sphere, "dims": 5, "bounds": (-10, 10), "name": "Sphere (5D)"},
    {"func": rastrigin, "dims": 5, "bounds": (-5.12, 5.12), "name": "Rastrigin (5D)"},
    {"func": rosenbrock, "dims": 4, "bounds": (-5, 10), "name": "Rosenbrock (4D)"},
    {"func": ackley, "dims": 5, "bounds": (-32.768, 32.768), "name": "Ackley (5D)"},
]


def evaluate_optimizer(candidate: dict[str, str], batch: list[dict[str, Any]]) -> list[dict]:
    """
    Evaluate an optimization algorithm on a batch of benchmark functions.

    The candidate contains Python code defining a `minimize` function that
    should find the minimum of a given function.
    """
    results = []
    optimizer_code = candidate["optimizer"]

    for benchmark in batch:
        func = benchmark["func"]
        dims = benchmark["dims"]
        bounds = benchmark["bounds"]
        func_name = benchmark["name"]

        # Maximum function evaluations allowed
        max_evals = 500

        # Create execution context
        exec_globals = {
            "random": random,
            "math": math,
            "Random": random.Random,
        }

        try:
            # Compile and execute the optimizer code
            exec(optimizer_code, exec_globals)

            # Check if minimize function exists
            if "minimize" not in exec_globals:
                raise ValueError("Code must define a `minimize` function")

            minimize_fn = exec_globals["minimize"]

            # Run the optimizer
            best_x, best_val = minimize_fn(func, bounds, dims, max_evals)

            # Score: negative log of (final value + epsilon) - lower is better, so we negate
            # We add small epsilon to avoid log(0)
            epsilon = 1e-10
            raw_score = -math.log10(abs(best_val) + epsilon)

            # Normalize to 0-1 range (higher is better)
            # For these functions, values < 1.0 are considered excellent
            score = max(0.0, min(1.0, 1.0 - (abs(best_val) / 10.0)))

            feedback = f"Function: {func_name}\n"
            feedback += f"Final value: {best_val:.6f}\n"
            feedback += f"Final point: {[f'{xi:.4f}' for xi in best_x]}\n"

            if abs(best_val) < 1.0:
                feedback += "✓ Excellent! Found a very good minimum."
            elif abs(best_val) < 10.0:
                feedback += "Good progress, but can improve further."
            else:
                feedback += "Poor result. The algorithm may be stuck in local minima or exploring inefficiently."

        except Exception as e:
            # Execution failed
            score = 0.0
            best_val = float("inf")
            feedback = f"Function: {func_name}\n"
            feedback += f"✗ Error executing optimizer: {e!s}\n"
            feedback += "Common issues:\n"
            feedback += "- Syntax errors in code\n"
            feedback += "- Missing return statement\n"
            feedback += "- Using undefined variables/functions\n"
            feedback += "- Type errors (e.g., treating list as float)\n"

        results.append(
            {
                "score": score,
                "context_and_feedback": {
                    "inputs": f"Function: {func_name}, Dimensions: {dims}, Bounds: {bounds}",
                    "outputs": f"Final value: {best_val:.6e}",
                    "feedback": feedback,
                    "benchmark": func_name,
                },
            }
        )

    return results


# Initial simple optimizer (random search)
SEED_OPTIMIZER = """
import random

def minimize(func, bounds, dims, max_evals):
    \"\"\"Minimize a function using random search.\"\"\"
    best_x = None
    best_val = float('inf')
    
    for _ in range(max_evals):
        # Generate random point
        x = [random.uniform(bounds[0], bounds[1]) for _ in range(dims)]
        val = func(x)
        
        if val < best_val:
            best_val = val
            best_x = x
    
    return best_x, best_val
"""


def main():
    """Run optimization algorithm evolution using GEPA."""

    from gepa import evolve

    print("=" * 80)
    print("GEPA Evolve-Anything Example: Function Minimization Algorithm Evolution")
    print("=" * 80)
    print(f"\nBenchmark functions: {', '.join(b['name'] for b in BENCHMARKS)}")
    print("Starting with: Random Search")
    print("\nGoal: Evolve a better optimization algorithm\n")
    print("Starting evolution...\n")

    # Run GEPA evolution
    result = evolve(
        seed_candidate={"optimizer": SEED_OPTIMIZER},
        trainset=BENCHMARKS,
        evaluate=evaluate_optimizer,
        reflection_prompt="""You are evolving an optimization algorithm to minimize mathematical functions.

The current code defines a `minimize(func, bounds, dims, max_evals)` function that should:
- Take a function `func`, bounds (min, max), number of dimensions, and max evaluations
- Return (best_x, best_val) - the best point found and its function value

Current algorithm limitations from the feedback:
- May get stuck in local minima (especially for Rastrigin, Ackley)
- May explore inefficiently (especially for Rosenbrock's narrow valley)
- May not adapt to different function landscapes

Improvement strategies to consider:
1. **Simulated Annealing**: Accept worse solutions with decreasing probability
2. **Momentum/Inertia**: Build on previous directions that worked
3. **Adaptive Step Sizes**: Large steps for exploration, small for refinement
4. **Population-Based**: Maintain multiple candidate solutions
5. **Gradient Estimation**: Approximate gradients using finite differences
6. **Restart Mechanisms**: Escape local minima by periodic restarts

Keep the function signature unchanged: `def minimize(func, bounds, dims, max_evals):`

Provide only the improved Python code without explanations.""",
        num_iterations=30,
        minibatch_size=2,  # Evaluate on 2 functions at a time
        teacher_lm="openai/gpt-4o",
        random_seed=42,
        output_dir="./function_minimization_output",
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"\nBest Score: {result['best_score']:.3f}")
    print("\nEvolved Optimizer Code:")
    print("-" * 80)
    print(result["best_candidate"]["optimizer"])
    print("-" * 80)

    # Test on all benchmarks
    print("\n" + "=" * 80)
    print("Testing Evolved Algorithm on All Benchmarks:")
    print("=" * 80)

    eval_results = evaluate_optimizer(result["best_candidate"], BENCHMARKS)

    print(f"\nOverall Score: {sum(r['score'] for r in eval_results) / len(eval_results):.3f}")

    for benchmark, eval_result in zip(BENCHMARKS, eval_results, strict=False):
        print(f"\n{benchmark['name']}:")
        print(f"  Score: {eval_result['score']:.3f}")
        print(f"  {eval_result['context_and_feedback']['outputs']}")


if __name__ == "__main__":
    main()
