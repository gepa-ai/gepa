"""
Tutorial: Using EvolveAdapter with Signal Processing Example

This script demonstrates how to adapt the OpenEvolve signal processing example
to work with GEPA's EvolveAdapter.

Prerequisites:
1. Clone the GEPA repository
2. Install GEPA in editable mode: pip install -e ".[full]"
3. Set OPENAI_API_KEY environment variable
4. Run from: src/gepa/examples/evolve_adapter/signal_processing/
"""

# Import the signal generation function from evaluator
import sys
from pathlib import Path

from gepa import optimize
from gepa.adapters.evolve_adapter.evolve_adapter import EvolveAdapter

sys.path.insert(0, str(Path(__file__).parent / "tutorial_example"))
from evaluator import generate_test_signals

# Path to your modified OpenEvolve project directory
project_path = Path(__file__).parent / "tutorial_example"

# Create the adapter
adapter = EvolveAdapter(path=project_path)

# Define training data (signal pairs)
# Each data_instance is a tuple (noisy_signal, clean_signal) or dict
# Note: data_instance can be of any type - use whatever best suits your project
test_signals = generate_test_signals(5)  # Generate 5 different test signals
trainset = [
    # Use tuple format: (noisy_signal, clean_signal)
    signal_pair
    for signal_pair in test_signals
]

# You can also use dict format if preferred:
# trainset = [
#     {"noisy_signal": noisy, "clean_signal": clean, "window_size": 20}
#     for noisy, clean in test_signals
# ]

# Note: You can use just one data instance if that's more analogous to your original
# OpenEvolve project setup. For example, if your original project evaluated on one
# fixed signal, using a single data instance mimics that setup.

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
    max_metric_calls=100,  # Budget for evaluation calls - adjust as needed
    display_progress_bar=True,
)

# Display results
best_score = result.val_aggregate_scores[result.best_idx]
print(f"\nBest score: {best_score}")
print(f"Best candidate index: {result.best_idx}")
print(f"Total candidates evaluated: {len(result.candidates)}")
print(f"Total metric calls: {result.total_metric_calls}")
