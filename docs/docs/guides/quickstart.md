# Quick Start

This guide will help you get started with GEPA in just a few minutes.

## Installation

Install GEPA from PyPI:

```bash
pip install gepa
```

For the latest development version:

```bash
pip install git+https://github.com/gepa-ai/gepa.git
```

To install with all optional dependencies:

```bash
pip install gepa[full]
```

## Your First Optimization

### Option 1: Using the Default Adapter

The simplest way to use GEPA is with the built-in `DefaultAdapter` for single-turn LLM tasks:

```python
import gepa

# Define your training data
# Each example should have an input and expected output
trainset = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    # ... more examples
]

# Define your seed prompt
seed_prompt = {
    "system_prompt": "You are a helpful assistant. Answer questions concisely."
}

# Run optimization
result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    task_lm="openai/gpt-4o-mini",      # Model to optimize
    reflection_lm="openai/gpt-4o",      # Model for reflection
    max_metric_calls=50,                # Budget
)

# Get the optimized prompt
print("Best prompt:", result.best_candidate['system_prompt'])
print("Best score:", result.best_score)
```

### Option 2: Using DSPy (Recommended)

For more complex programs, use GEPA through DSPy:

```python
import dspy

# Configure the LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define your program
class QAProgram(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate(question=question)

# Prepare data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    # ... more examples
]

# Optimize with GEPA
optimizer = dspy.GEPA(
    metric=lambda example, pred: pred.answer == example.answer,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
)

optimized_program = optimizer.compile(QAProgram(), trainset=trainset)
```

## Understanding the Output

The `GEPAResult` object contains:

```python
result.best_candidate    # Dict[str, str] - the optimized text components
result.best_score        # float - validation score of best candidate
result.pareto_frontier   # List of candidates on the Pareto frontier
result.history           # Optimization history
```

## Configuration Options

### Stop Conditions

Control when optimization stops:

```python
from gepa import MaxMetricCallsStopper, TimeoutStopCondition, NoImprovementStopper

result = gepa.optimize(
    # ... other args ...
    max_metric_calls=100,                          # Stop after 100 evaluations
    stop_callbacks=[
        TimeoutStopCondition(seconds=3600),        # Or after 1 hour
        NoImprovementStopper(patience=10),         # Or after 10 iterations without improvement
    ],
)
```

### Candidate Selection Strategies

Choose how candidates are selected for mutation:

```python
result = gepa.optimize(
    # ... other args ...
    candidate_selection_strategy="pareto",      # Default: sample from Pareto frontier
    # candidate_selection_strategy="current_best",  # Always use best candidate
    # candidate_selection_strategy="epsilon_greedy", # Explore vs exploit
)
```

### Logging and Tracking

Track optimization progress:

```python
result = gepa.optimize(
    # ... other args ...
    use_wandb=True,                    # Log to Weights & Biases
    use_mlflow=True,                   # Log to MLflow
    run_dir="./gepa_runs/my_exp",      # Save state to disk
    display_progress_bar=True,         # Show progress
)
```

## Next Steps

- [Creating Adapters](adapters.md) - Build custom adapters for your system
- [API Reference](../api/index.md) - Detailed API documentation
- [Tutorials](../tutorials/index.md) - Step-by-step examples
