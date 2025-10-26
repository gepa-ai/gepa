# Optimize Anything with GEPA

GEPA's `optimize_anything` API makes it trivial to evolve any system with string components - prompts, code, configurations, documents, or any text-based parameters.

## Quick Start

```python
from gepa import optimize_anything

# 1. Define what to evolve
seed_candidate = {
    "system_prompt": "You are a helpful assistant."
}

# 2. Define your workload
trainset = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 5*3?", "answer": "15"},
    # ... more examples
]

# 3. Define how to evaluate
def evaluate(candidate, batch):
    results = []
    for item in batch:
        # Use candidate in your system
        response = your_system(candidate["system_prompt"], item["question"])
        
        # Score it
        score = 1.0 if response == item["answer"] else 0.0
        
        # Provide context for reflection
        context = {
            "feedback": f"Expected {item['answer']}, got {response}"
        }
        
        results.append((score, context))
    return results

# 4. Evolve!
result = optimize_anything(
    seed_candidate=seed_candidate,
    trainset=trainset,
    evaluate=evaluate,
    reflection_lm="gpt-4",
    max_metric_calls=100
)

print(f"Best candidate: {result.best_candidate}")
print(f"Best score: {result.best_score}")
```

## Why optimize_anything?

### Before: Implementing GEPAAdapter (Complex)
```python
class MyAdapter(GEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        # 20+ lines of boilerplate
        ...
    
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        # 30+ lines of data transformation
        ...
```

### After: Using optimize_anything (Simple)
```python
def evaluate(candidate, batch):
    return [(score, context) for score, context in run_evaluation(candidate, batch)]

result = optimize_anything(seed_candidate, trainset, evaluate, reflection_lm="gpt-4", max_metric_calls=100)
```

**10x less code!**

## Key Features

### 1. **Flexible Evaluation**
Your `evaluate` function can return either:

**Simple: Just Scores**
```python
def evaluate(candidate, batch):
    return [0.8, 1.0, 0.5, ...]  # One score per batch item
```

**Advanced: Scores + Rich Context**
```python
def evaluate(candidate, batch):
    return [
        (0.8, {"feedback": "Good but...", "trace": trace_data}),
        (1.0, {"feedback": "Perfect!", "details": details}),
        (0.5, {"feedback": "Failed because...", "error": error})
    ]
```

The context can be **anything**: dicts, strings, objects. GEPA uses it for reflection.

### 2. **Per-Workload Granularity**
GEPA evaluates and tracks scores **per workload instance**, enabling:
- **Pareto frontiers**: Find candidates that excel on different subsets
- **Fine-grained optimization**: Target specific weaknesses
- **Better merging**: Combine strengths from multiple candidates

### 3. **Production-Ready**
- **Checkpointing**: Resume from `run_dir` if interrupted
- **Logging**: Integrated W&B and MLflow support
- **Stopping conditions**: Time limits, score thresholds, file watchers
- **Error handling**: Graceful degradation on failures

## Complete Examples

See the detailed examples in [examples/evolve_anything/](examples/evolve_anything/) for:
- Prompt optimization
- Code evolution
- Multi-component systems
- RAG document optimization
- Configuration tuning

## API Reference

### Core Function

```python
def optimize_anything(
    # Required
    seed_candidate: dict[str, str],
    trainset: list,
    evaluate: Callable,
    
    # Reflection
    reflection_lm: str | LanguageModel,
    
    # Optional
    valset: list | None = None,
    context_to_feedback: Callable | None = None,
    max_metric_calls: int | None = None,
    ...
) -> GEPAResult
```

### Parameters

#### **seed_candidate** : `dict[str, str]` (Required)
Initial values for components to evolve.

#### **trainset** : `list` (Required)
Workload instances for training. Can be anything.

#### **evaluate** : `Callable` (Required)
Function that evaluates candidates.

**Returns:** Either `list[float]` or `list[tuple[float, Any]]`

#### **reflection_lm** : `str | LanguageModel` (Required)
LLM for generating improved candidates (e.g., `"gpt-4"`).

#### **max_metric_calls** : `int`
Maximum evaluation calls before stopping.

For complete API reference, see the function docstring in `src/gepa/api.py`.

## Next Steps

- **Examples**: Check out [examples/evolve_anything/](examples/evolve_anything/)
- **Paper**: [GEPA paper](https://arxiv.org/abs/2507.19457)
- **GitHub**: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- **Discord**: [Join us](https://discord.gg/A7dABbtmFw)

