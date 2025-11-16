"""
Simple example demonstrating optimize_anything.

This example optimizes parameters of a linear function (y = a*x + b)
to fit target data points. The algorithm uses LLM-based reflection
to iteratively improve the parameters based on evaluation feedback.
"""

import json
import random
from typing import Any, Sequence
from gepa.optimize_anything import SideInfo, optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

# Dataset: target data points (x, y) that we want to fit
# The true function is y = 3*x + 2
dataset = [
    {"x": x, "target_y": 10.5*x + 2.3}
    for x in range(100)
]

def fitness_fn(candidate: dict[str, str], batch: Sequence[Any], **kwargs) -> list[tuple[float, Any, SideInfo]]:
    """
    Evaluates how well the candidate parameters fit the data.
    
    Candidate contains: {"a": "3.5", "b": "1.8"}
    We evaluate: y = a*x + b against target values
    
    Returns: list of (score, output, side_info) tuples
    """
    try:
        import json
        candidate = json.loads(candidate['function_params'])
        a = float(candidate["a"])
        b = float(candidate["b"])
    except Exception as e:

        return [
            (0.0, None, {"Feedback": f"The candidate is not a valid JSON object or missing 'a'/'b' keys. Error: {e}"})
            for _ in batch
        ]

    results = []
    for instance in batch:
        x = instance["x"]
        target_y = instance["target_y"]
        
        # Compute prediction using candidate parameters
        predicted_y = b * x + a
        
        # Calculate error and score (higher is better)
        error = abs(predicted_y - target_y)
        score = 1.0 / (1.0 + error)  # Score between 0 and 1
        
        # Build diagnostic information for the LLM
        side_info = {
            "absolute_error (lower is better)": error,
            "Input": f"x={x}",
            "Output": f"predicted y={predicted_y:.2f}",
            "Expected": f"target y={target_y}",
            "Feedback": f"Adjust the parameters a and b appropriately to fit the data points.",
        }
        
        results.append((score, predicted_y, side_info))
    
    return results

def main():
    # Initial guess
    seed_candidate = {
        "function_params": json.dumps({"a": 1.0, "b": 0.0})
    }

    # Configure optimization
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=1000,  # Stop after 15 evaluations
            seed=42,
        ),
        reflection=ReflectionConfig(
            reflection_lm="gpt-5-nano",  # LLM for proposing improved parameters
            reflection_minibatch_size=15,  # Use 3 data points per reflection
        ),
    )

    # Run optimization
    print("="*60)
    print("Optimizing y = a*x + b to fit data points")
    print("="*60)
    print(f"Initial guess: {seed_candidate['function_params']}")

    random.Random(0).shuffle(dataset)
    trainset = dataset[:len(dataset)//2]
    valset = dataset[len(dataset)//2:]

    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        dataset=trainset,
        valset=valset,
        config=config,
    )

    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best parameters found:")
    print(f"  {result.best_candidate['function_params']}")
    print(f"\nBest score: {result.val_aggregate_scores[result.best_idx]:.4f}")
    print(f"Total evaluations: {result.total_metric_calls}")

if __name__ == "__main__":
    main()
