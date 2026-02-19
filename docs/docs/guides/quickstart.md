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

### Option 2: Using optimize_anything

The `optimize_anything` API can optimize any text artifact — code, prompts, agent architectures, configurations, SVG graphics — not just prompts. You provide an evaluator that scores candidates and returns diagnostic feedback (Actionable Side Information), and the system handles the search.

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

def evaluate(candidate: str) -> float:
    """Score a candidate and log diagnostics as ASI."""
    result = run_my_system(candidate)
    oa.log(f"Output: {result.output}")
    oa.log(f"Error: {result.error}")
    return result.score

result = optimize_anything(
    seed_candidate="<your initial artifact>",
    evaluator=evaluate,
    objective="Describe what you want to optimize for.",
    config=GEPAConfig(engine=EngineConfig(max_metric_calls=100)),
)

print("Best candidate:", result.best_candidate)
```

For richer feedback, return a `(score, side_info_dict)` tuple from your evaluator:

```python
def evaluate(candidate: str) -> tuple[float, dict]:
    result = run_my_system(candidate)
    return result.score, {
        "Error": result.stderr,
        "Output": result.stdout,
    }
```

See the [optimize_anything blog post](../blog/posts/2026-02-18-introducing-optimize-anything/index.md) for examples across seven domains.

### Option 3: Using DSPy (Recommended for prompt optimization)

For more complex programs, use GEPA through DSPy. GEPA works best with **feedback metrics** that provide textual explanations, not just scores:

```python
import dspy

# Configure the task LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define your program
class QAProgram(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate(question=question)

# Define a metric WITH feedback - this is key for GEPA!
def metric_with_feedback(example, pred, trace=None):
    correct = example.answer.lower() in pred.answer.lower()
    score = 1.0 if correct else 0.0
    
    # Provide textual feedback to guide GEPA's reflection
    if correct:
        feedback = f"Correct! The answer '{pred.answer}' matches the expected answer '{example.answer}'."
    else:
        feedback = (
            f"Incorrect. Expected '{example.answer}' but got '{pred.answer}'. "
            f"Think about how to reason more carefully to arrive at the correct answer."
        )
    
    return dspy.Prediction(score=score, feedback=feedback)

# Prepare data (aim for 30-300 examples for best results)
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    # ... more examples
]

# Optimize with GEPA
optimizer = dspy.GEPA(
    metric=metric_with_feedback,
    reflection_lm=dspy.LM("openai/gpt-4o"),  # Strong model for reflection
    auto="light",  # Automatic budget configuration
    num_threads=8,
    track_stats=True,
)

optimized_program = optimizer.compile(QAProgram(), trainset=trainset)

# View the optimized prompt
print(optimized_program.generate.signature.instructions)
```

!!! tip "Feedback is Key"
    GEPA's strength lies in leveraging textual feedback. The more informative your feedback, the better GEPA can reflect and propose improvements. Include details like:
    
    - What went wrong and what was expected
    - Hints for improvement
    - Reference solutions (if available)
    - Breakdown of sub-scores for multi-objective tasks

For more detailed examples, see the [dspy.GEPA tutorials](https://dspy.ai/tutorials/gepa_ai_program/).

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

For best practices, tips on data preparation, feedback quality, model selection, and more, see the [FAQ](faq.md).

- [Creating Adapters](adapters.md) - Build custom adapters for your system
- [API Reference](../api/index.md) - Detailed API documentation
- [Tutorials](../tutorials/index.md) - Step-by-step examples
