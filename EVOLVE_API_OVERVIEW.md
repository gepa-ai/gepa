# GEPA Evolve-Anything API: Complete Overview

## What Was Built

I've implemented a simple, powerful **"evolve-anything" API** for GEPA that makes it trivially easy to apply evolutionary optimization to any system with string components. This addresses the goals from your conversation with Matei about creating a general framework for evolving "anything with strings."

## The Core API

### Location
- **Implementation**: `src/gepa/optimize.py`
- **Export**: `from gepa import evolve, WorkloadResult`
- **Examples**: `examples/evolve_anything/`

### Signature

```python
def evolve(
    seed_candidate: dict[str, str],           # Initial strings to evolve
    trainset: list[Any],                      # Your workloads
    evaluate: Callable[[dict[str, str], list[Any]], list[WorkloadResult]],
    *,
    reflection_prompt: str | dict[str, str] | None = None,
    failure_score: float = 0.0,
    num_iterations: int = 50,
    minibatch_size: int = 25,
    teacher_lm: str = "openai/gpt-4o",
    random_seed: int | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, Any]
```

### Minimal Example

```python
from gepa import evolve

def evaluate(candidate, batch):
    results = []
    for workload in batch:
        # Execute your system
        output = run_system(candidate["component"], workload)
        score = compute_score(output, workload)
        
        results.append({
            "score": score,
            "context_and_feedback": {
                "inputs": workload,
                "outputs": output,
                "feedback": "explanation of performance"
            }
        })
    return results

result = evolve(
    seed_candidate={"component": "initial text"},
    trainset=[...],
    evaluate=evaluate,
)
```

## Key Design Principles (From Your Conversation)

### 1. **Candidate = List of Strings** ✓
- `dict[str, str]` mapping component names to text
- Not tied to filesystem
- Can be prompts, code, configs, schemas, anything

### 2. **Trainset = Arbitrary Workloads** ✓
- `list[Any]` - GEPA doesn't interpret, just passes to evaluate
- Can be test cases, queries, scenarios, configs, etc.

### 3. **Evaluate = Your System Logic** ✓
```python
def evaluate(candidate: dict[str, str], batch: list[Any]) -> list[WorkloadResult]:
    # Returns per-workload scores and context
```
- **Granular scoring**: Individual scores for each workload
- **Dynamic context**: Include only relevant info per workload
- Complete flexibility: execute code, call APIs, anything

### 4. **Optional Reflection Prompts** ✓
- Domain-specific guidance for improvements
- Can be different per component
- Falls back to GEPA default if not provided

### 5. **No Filesystem Dependency** ✓
- Everything in-memory
- Code can be compiled/executed directly
- Dynamic context discovery

## What Can You Evolve?

Demonstrated across 4 diverse examples:

### 1. **Prompts** (`01_prompt_optimization_hotpotqa.py`)
- HotpotQA question-answering
- Multi-hop reasoning
- Dynamic context from LLM responses

### 2. **Code/Algorithms** (`02_function_minimization.py`)
- Python optimization algorithms
- Benchmark function evaluation
- Discovers simulated annealing, adaptive techniques

### 3. **Structured Data** (`03_api_schema_evolution.py`)
- JSON API schemas
- Evaluated against developer usability
- Non-executable text evolution

### 4. **Documentation** (`04_tutorial_evolution.py`)
- Educational content
- Simulated student learning
- "Soft" text optimization

### 5. **Circle Packing** (`05_circle_packing.py`)
- Mathematical optimization algorithms
- Ported from OpenEvolve example
- Rich artifacts (errors, validation, metrics)
- Target: 2.635 sum of radii (AlphaEvolve)

## Advantages Over Full GEPAAdapter

| Aspect | Full GEPAAdapter | `evolve()` API |
|--------|------------------|----------------|
| **Complexity** | Implement 3 protocol methods | 3 required parameters |
| **Filesystem** | Often tied to files | Pure in-memory |
| **Context** | Static structure | Dynamic per-workload |
| **Scoring** | Aggregate focus | Per-workload granular |
| **Workload Types** | Generic type variables | Literally `Any` |

## Comparison with OpenEvolve

Both are powerful and complementary:

| Feature | OpenEvolve | GEPA `evolve()` |
|---------|-----------|-----------------|
| **Input** | File paths | In-memory strings |
| **Evaluation** | Separate evaluator script | Inline function |
| **System Messages** | Global config | Per-component prompts |
| **Granularity** | Combined metrics | Per-workload scores |
| **File I/O** | Required (temp files) | Optional |
| **Best For** | Full programs, hardware | Components, prompts, schemas |

**Key Insight**: OpenEvolve excels at evolving complete programs with complex execution environments. GEPA `evolve()` excels at component-level optimization with flexible, in-memory evaluation.

**Cross-Compatibility**: We successfully ported OpenEvolve's `circle_packing_with_artifacts` example to GEPA's evolve API (see `05_circle_packing.py`), demonstrating that both frameworks can handle sophisticated algorithm evolution. The same task can be solved with either approach, each offering different tradeoffs. See `examples/evolve_anything/CIRCLE_PACKING_NOTES.md` for detailed comparison.

## Implementation Details

### Internal Architecture

```
User calls: gepa.evolve()
     ↓
Creates: _OptimizeAdapter (bridges to GEPAAdapter protocol)
     ↓
Calls: gepa.api.optimize() (full GEPA engine)
     ↓
Returns: Formatted results with best candidate
```

The `_OptimizeAdapter` class:
1. Wraps user's evaluate function to return `EvaluationBatch`
2. Converts `WorkloadResult` dicts to GEPA's internal trajectory format
3. Builds reflective datasets from `context_and_feedback`
4. Handles custom reflection prompts via template parameter

### Return Value Structure

```python
{
    "best_candidate": dict[str, str],  # Best evolved component texts
    "best_score": float,               # Score on full trainset
    "history": GEPAResult,            # Full evolution history/state
    "pareto_frontier": dict,          # Non-dominated candidates per task
    "output_dir": str | None,         # Where checkpoints saved
}
```

## Files Created

```
src/gepa/
    optimize.py                          # Core evolve() API (422 lines)
    __init__.py                          # Updated to export evolve

examples/evolve_anything/
    README.md                            # User-facing documentation
    SUMMARY.md                           # Technical summary
    CIRCLE_PACKING_NOTES.md              # Circle packing port from OpenEvolve
    00_minimal_test.py                   # Quick verification test
    01_prompt_optimization_hotpotqa.py   # Prompt evolution example
    02_function_minimization.py          # Code evolution example
    03_api_schema_evolution.py           # Schema evolution example
    04_tutorial_evolution.py             # Documentation evolution example
    05_circle_packing.py                 # Circle packing (from OpenEvolve)

EVOLVE_API_OVERVIEW.md                   # This file
```

## Usage Examples

### Prompt Optimization

```python
from gepa import evolve

def evaluate_prompt(candidate, batch):
    results = []
    for question, answer in batch:
        response = llm(candidate["prompt"], question)
        score = 1.0 if answer in response else 0.0
        results.append({
            "score": score,
            "context_and_feedback": {
                "inputs": question,
                "outputs": response,
                "feedback": f"Expected: {answer}"
            }
        })
    return results

result = evolve(
    seed_candidate={"prompt": "Answer questions accurately."},
    trainset=[(q, a) for q, a in qa_pairs],
    evaluate=evaluate_prompt,
    num_iterations=50,
)
```

### Code Evolution

```python
def evaluate_code(candidate, batch):
    results = []
    for test_case in batch:
        try:
            exec_globals = {}
            exec(candidate["function"], exec_globals)
            output = exec_globals["func"](test_case["input"])
            score = 1.0 if output == test_case["expected"] else 0.0
            feedback = f"Got {output}, expected {test_case['expected']}"
        except Exception as e:
            score = 0.0
            feedback = f"Error: {e}"
        
        results.append({
            "score": score,
            "context_and_feedback": {
                "inputs": test_case["input"],
                "outputs": str(output) if 'output' in locals() else "Error",
                "feedback": feedback
            }
        })
    return results

result = evolve(
    seed_candidate={"function": "def func(x): return x"},
    trainset=test_cases,
    evaluate=evaluate_code,
    reflection_prompt="Optimize for correctness and performance.",
)
```

## Testing

Run the minimal test to verify installation:

```bash
cd examples/evolve_anything
export OPENAI_API_KEY="your-key"
python 00_minimal_test.py
```

Run full examples:

```bash
python 01_prompt_optimization_hotpotqa.py   # ~2-3 minutes
python 02_function_minimization.py          # ~3-5 minutes
python 03_api_schema_evolution.py           # ~2-3 minutes
python 04_tutorial_evolution.py             # ~3-4 minutes
```

## Extensibility

### Custom Reflection Prompts

```python
result = evolve(
    seed_candidate={
        "component_a": "...",
        "component_b": "..."
    },
    trainset=[...],
    evaluate=evaluate_fn,
    reflection_prompt={
        "component_a": "Optimize for clarity...",
        "component_b": "Optimize for performance...",
    }
)
```

### Multi-Component Evolution

```python
def evaluate_agent(candidate, batch):
    # candidate has multiple components
    agent = Agent(
        system_prompt=candidate["system_prompt"],
        tool_descriptions=candidate["tools"],
    )
    # ... evaluate agent
```

### Dynamic Context

```python
def evaluate(candidate, batch):
    for workload in batch:
        # ... execute system ...
        
        # Include only relevant context dynamically
        relevant_tools = [t for t in all_tools if t.was_called]
        relevant_code = extract_code_used(execution_trace)
        
        results.append({
            "score": score,
            "context_and_feedback": {
                "inputs": workload,
                "outputs": output,
                "feedback": feedback,
                "relevant_tools": relevant_tools,  # Dynamic!
                "relevant_code": relevant_code,     # Dynamic!
            }
        })
```

## Future Directions

Potential extensions:

1. **More OpenEvolve Examples**: Port symbolic regression, attention optimization
2. **RAG Document Evolution**: Evolve documents to improve retrieval
3. **Agent Tool Descriptions**: Evolve tool schemas for better agent performance
4. **Config Parameter Evolution**: Numeric hyperparameters as strings
5. **Multi-Modal**: Combine text evolution with other modalities

## Related Work

### Within GEPA
- Full `GEPAAdapter` for complex systems (DSPy, MCP, RAG)
- `gepa.optimize()` for advanced control
- Various pre-built adapters in `src/gepa/adapters/`

### External
- **OpenEvolve**: Full program evolution with MAP-Elites
- **DSPy MIPRO**: Prompt optimization for DSPy programs
- **EuroLLM**: Genetic algorithms for prompt optimization

## Citation

If you use this API in research, cite GEPA:

```bibtex
@software{gepa2025,
  title = {GEPA: General Evolutionary Prompt Adaptation},
  author = {Lakshya A Agrawal},
  year = {2025},
  url = {https://github.com/gepa-ai/gepa}
}
```

## Questions?

- See example scripts in `examples/evolve_anything/`
- Read the docstring in `src/gepa/optimize.py`
- Check main GEPA docs in `README.md`
- Open an issue on GitHub

