from pathlib import Path
from gepa import optimize
from gepa.adapters.evolve_adapter.evolve_adapter import EvolveAdapter

# Path to your OpenEvolve project directory
# This should contain: config.yaml, evaluator.py, initial_program.py
project_path = Path("your-project-path")

# Create the adapter
adapter = EvolveAdapter(
    path=project_path
)

# Define training data (for this example, a batch of function minimization problems)
# Each item represents a different problem instance
trainset = [
    {
        "global_min_x": -1.704,
        "global_min_y": 0.678,
        "global_min_value": -1.519,
        "bounds": (-5, 5),
        "function_name": "sin_cos_function"
    },
    {
        "global_min_x": 0.0,
        "global_min_y": 0.0,
        "global_min_value": 0.0,
        "bounds": (-3, 3),
        "function_name": "quadratic_function"
    },
    # Add more problem instances as needed
]

# Read initial program
with open(project_path / "initial_program.py", "r") as f:
    initial_program = f.read()

# Define seed candidate (the program to evolve)
seed_candidate = {
    "program": initial_program
}

# Run GEPA optimization
result = optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    adapter=adapter,
    max_metric_calls=60,  # Budget for evaluation calls -  adjust as needed
    display_progress_bar=True
)

# Get the best score (GEPAResult doesn't have best_score, use val_aggregate_scores[best_idx])
best_score = result.val_aggregate_scores[result.best_idx]
print(f"Best score: {best_score}")
print(f"Best candidate index: {result.best_idx}")
print(f"Total candidates evaluated: {len(result.candidates)}")
print(f"Total metric calls: {result.total_metric_calls}")

# The evolved program is in result.best_candidate["program"]
print(f"\nBest candidate program:")
print(result.best_candidate.get("program", "N/A")[:500] + "..." if len(result.best_candidate.get("program", "")) > 500 else result.best_candidate.get("program", "N/A"))