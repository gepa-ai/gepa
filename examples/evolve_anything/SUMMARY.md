# GEPA Evolve-Anything API: Summary

## What Was Created

I've implemented a simple, powerful "evolve-anything" API for GEPA as discussed in the conversation between you and your advisor. This API makes it trivially easy to apply GEPA's evolutionary optimization to **any system with string components**, without needing to implement a full `GEPAAdapter`.

## Core API: `gepa.evolve()`

Located in `src/gepa/optimize.py` and exported from `gepa.__init__.py`.

### Key Design Decisions (Aligned with Your Conversation)

1. **Seed Candidate = List of Strings**
   - `seed_candidate: dict[str, str]` - maps component names to initial text
   - Not tied to filesystem - pure in-memory strings
   - Can be prompts, code, configs, schemas, anything textual

2. **Trainset = Arbitrary Workloads**
   - `trainset: list[Any]` - any Python objects your evaluate function understands
   - GEPA doesn't interpret them, just passes to your evaluate function
   - Examples: test cases, queries, scenarios, benchmark configs

3. **Evaluate Function = Your System's Logic**
   ```python
   def evaluate(candidate: dict[str, str], batch: list[Any]) -> list[WorkloadResult]:
       # WorkloadResult = {"score": float, "context_and_feedback": dict}
   ```
   - **Granular per-workload scoring** (not just aggregate)
   - **Dynamic context**: include only relevant info for each workload in feedback
   - Complete flexibility: execute code, call APIs, run simulations, etc.

4. **Optional Custom Reflection Prompts**
   - `reflection_prompt: str | dict[str, str] | None`
   - Domain-specific guidance for how to improve components
   - Falls back to GEPA's default if not provided

### Advantages Over Full GEPAAdapter

✅ **Much simpler**: 3 required parameters vs implementing a full protocol  
✅ **Not tied to filesystem**: Everything in-memory  
✅ **Dynamic context**: Feedback can include different context per workload  
✅ **Granular scoring**: GEPA sees individual workload scores  
✅ **Flexible workloads**: Any Python object, not just predefined types  

## Example Scripts Created

### 1. **Prompt Optimization** (`01_prompt_optimization_hotpotqa.py`)
- **Evolves**: Question-answering prompts for HotpotQA
- **Demonstrates**: Simple prompt optimization with clear feedback
- **Key Feature**: Uses actual LLM calls (optional) or mock responses

### 2. **Function Minimization** (`02_function_minimization.py`)
- **Evolves**: Optimization algorithm code (Python functions)
- **Demonstrates**: Code evolution with execution and performance evaluation
- **Key Feature**: Starting from random search, discovers simulated annealing, adaptive techniques

### 3. **API Schema Evolution** (`03_api_schema_evolution.py`)
- **Evolves**: JSON API schemas
- **Demonstrates**: Structured data evolution, not just executable code
- **Key Feature**: Evaluates against developer pain points and usability criteria

### 4. **Tutorial Content Evolution** (`04_tutorial_evolution.py`)
- **Evolves**: Educational content (recursion tutorial)
- **Demonstrates**: "Soft" text optimization (docs, explanations, teaching)
- **Key Feature**: Simulates students learning, evaluates understanding

## Key Technical Implementation

### Internal Architecture

```
gepa.evolve() 
    ↓
_OptimizeAdapter (bridges to GEPAAdapter protocol)
    ↓
gepa.api.optimize() (full GEPA engine)
```

The `_OptimizeAdapter` class internally:
1. Wraps user's evaluate function to return `EvaluationBatch`
2. Converts `WorkloadResult` dicts to GEPA's internal format
3. Builds reflective datasets from `context_and_feedback`
4. Handles custom reflection prompts

### Return Value

```python
{
    "best_candidate": dict[str, str],  # Best evolved strings
    "best_score": float,               # Score on full trainset
    "history": GEPAResult,            # Full evolution history
    "pareto_frontier": dict,          # Non-dominated candidates
    "output_dir": str | None,         # Where results saved
}
```

## Comparison with OpenEvolve

| Aspect | OpenEvolve | GEPA `evolve()` |
|--------|-----------|-----------------|
| **Input Format** | File paths | In-memory strings |
| **Evaluation** | Separate evaluator script | Inline Python function |
| **Context** | Static (filesystem) | Dynamic (per-workload) |
| **Granularity** | Combined score or metrics dict | Per-workload scores |
| **Reflection** | Global system message | Per-component prompts |
| **File I/O** | Required (temp files) | Optional (in-memory) |
| **Best For** | Full programs, hardware optimization | Components, prompts, schemas |

Both approaches are powerful and complementary!

## Usage Pattern

```python
from gepa import evolve

def evaluate_my_system(candidate, batch):
    results = []
    for workload in batch:
        # 1. Execute your system with candidate strings
        output = run_my_system(candidate["component"], workload)
        
        # 2. Score the output
        score = compute_score(output, workload)
        
        # 3. Provide rich feedback
        results.append({
            "score": score,
            "context_and_feedback": {
                "inputs": workload,
                "outputs": output,
                "feedback": "What went wrong/right",
                # Include any dynamic context relevant to this workload
            }
        })
    return results

result = evolve(
    seed_candidate={"component": "initial text"},
    trainset=[...],
    evaluate=evaluate_my_system,
    num_iterations=50,
)

print(result["best_candidate"])
```

## What You Can Evolve

Literally anything representable as text:

- ✅ **Prompts**: System prompts, instructions, few-shot examples
- ✅ **Code**: Functions, classes, algorithms, shaders, GPU kernels
- ✅ **Configs**: JSON schemas, YAML configs, API specs, hyperparameters
- ✅ **Documents**: Tutorials, API docs, READMEs, error messages
- ✅ **Structured Data**: XML, CSV headers, regex patterns, SQL queries
- ✅ **Domain-Specific Languages**: LaTeX templates, DSL definitions

## Alignment with Conversation Goals

This implementation addresses all the key points from your conversation:

1. ✅ **Simple user interface**: Just 3 required parameters
2. ✅ **Infinitely flexible**: Works for any system with string components
3. ✅ **Dynamic context**: `evaluate` returns arbitrary `context_and_feedback` per workload
4. ✅ **Not tied to filesystem**: Pure in-memory operation
5. ✅ **Granular evaluation**: Per-workload scores, not just aggregates
6. ✅ **Broad applicability**: Demonstrated on prompts, code, schemas, and docs

## Next Steps

Suggested directions:
1. **More Examples**: Add examples from OpenEvolve (symbolic regression, web scraping, etc.)
2. **Multi-Component**: Show examples evolving multiple coupled components
3. **RAG Optimization**: Demonstrate evolving documents to improve retrieval
4. **Agent Systems**: Evolve tool descriptions, system prompts together
5. **Numeric Configs**: Show evolution of numeric hyperparameters (as strings)

## Files Created

```
src/gepa/optimize.py                          # Core API implementation
src/gepa/__init__.py                          # Updated to export evolve()
examples/evolve_anything/
    README.md                                 # User-facing documentation
    SUMMARY.md                                # This file
    01_prompt_optimization_hotpotqa.py        # Example 1: Prompts
    02_function_minimization.py               # Example 2: Code/algorithms
    03_api_schema_evolution.py                # Example 3: Structured data
    04_tutorial_evolution.py                  # Example 4: Educational content
```

## Testing

To test the API:

```bash
cd examples/evolve_anything

# Requires OpenAI API key for LLM calls
export OPENAI_API_KEY="your-key"

# Run any example
python 01_prompt_optimization_hotpotqa.py
python 02_function_minimization.py
python 03_api_schema_evolution.py
python 04_tutorial_evolution.py
```

Examples are self-contained and include mock fallbacks if no API key is set.

