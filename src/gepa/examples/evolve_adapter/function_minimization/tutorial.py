"""
Tutorial: Using EvolveAdapter with OpenEvolve Projects

Prerequisites:
1. Clone the GEPA repository to get the tutorial files:
   git clone https://github.com/gepa-ai/gepa.git
   cd gepa

2. Install GEPA and dependencies:
   pip install gepa[full] openevolve numpy scipy pyyaml litellm

3. Set up your API key for the LLM (e.g. set OPENAI_API_KEY environment variable)

This script should be run from: src/gepa/examples/evolve_adapter/function_minimization/
"""

import importlib.util
import os
import tempfile
from pathlib import Path

from gepa import optimize
from gepa.adapters.evolve_adapter.evolve_adapter import EvolveAdapter

# Path to your OpenEvolve project directory
# This should contain: config.yaml, evaluator.py, initial_program.py
# For this example, we use the tutorial_example directory
project_path = Path(__file__).parent / "tutorial_example"

# Create the adapter
adapter = EvolveAdapter(path=project_path)

# Define training data (for this example, function minimization problems)
# Each item represents a different problem instance
# The adapter will call your evaluate function for each instance in the batch
#
# Note: Data instances can be of any type (dict, string, custom object, etc.) - use
# whatever type best suits your original OpenEvolve project setup. For our function,
# minimization example we use dicts.
#
# Also note: You can use just one data instance if that's more analogous to your original
# OpenEvolve project setup. For example, the original function minimization project
# evolved on one fixed problem, so using a single data instance mimics that setup.
trainset = [
    {
        "global_min_x": -1.704,
        "global_min_y": 0.678,
        "global_min_value": -1.519,
        "bounds": (-5, 5),
        "function_name": "sin_cos_function",
    },
    # Uncomment the following to add more problem instances:
    # {
    #     "global_min_x": 0.0,
    #     "global_min_y": 0.0,
    #     "global_min_value": 0.0,
    #     "bounds": (-3, 3),
    #     "function_name": "quadratic_function",
    # },
    # Add more problem instances as needed
]

# Read initial program
with open(project_path / "initial_program.py") as f:
    initial_program = f.read()

# Define seed candidate (the program to evolve)
seed_candidate = {"program": initial_program}

# Run GEPA optimization
result = optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    adapter=adapter,
    max_metric_calls=500,  # Budget for evaluation calls -  adjust as needed
    display_progress_bar=True,
)

# Get the best score
best_score = result.val_aggregate_scores[result.best_idx]
print(f"Best score: {best_score}")
print(f"Best candidate index: {result.best_idx}")
print(f"Total candidates evaluated: {len(result.candidates)}")
print(f"Total metric calls: {result.total_metric_calls}")

# The evolved program is in result.best_candidate["program"]
print("\nBest candidate program:")
print(result.best_candidate.get("program", "N/A"))

# Re-evaluate the best candidate to get detailed metrics for comparison
evaluator_path = project_path / "evaluator.py"
spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load evaluator module from {evaluator_path}")
evaluator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator_module)
evaluate = evaluator_module.evaluate

# Use the adapter's _construct_complete_program method to ensure
# the full program is constructed (including run_search() and other fixed code)
# (as best_candidate["program"] may only contain the evolved block)
best_program_text = result.best_candidate.get("program", "")
complete_program = adapter._construct_complete_program(best_program_text)

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(complete_program)
    temp_program_path = f.name

best_result = evaluate(temp_program_path, trainset[0])
gepa_metrics = best_result.metrics

gepa_detailed = {
    "value_score": gepa_metrics.get("value_score", 0.0),
    "distance_score": gepa_metrics.get("distance_score", 0.0),
    "reliability_score": gepa_metrics.get("reliability_score", 0.0),
    "combined_score": gepa_metrics.get("combined_score", 0.0),
}

os.unlink(temp_program_path)

print("\nGEPA Best Candidate Detailed Metrics:")
print("=" * 50)
for metric, value in gepa_detailed.items():
    print(f"{metric}: {value:.4f}")
