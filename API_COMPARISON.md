# GEPA "Evolve Anything" API Comparison

## Executive Summary

**Recommendation: The `evolve()` API in `optimize.py` is significantly better** for the following reasons:

1. ✅ **Simpler, more intuitive interface** - fewer parameters, easier to learn
2. ✅ **Better documentation** - comprehensive examples in docstring
3. ✅ **Per-component customization** - `reflection_prompt` can be dict mapping component → prompt
4. ✅ **Cleaner return value** - simple dict vs complex GEPAResult object
5. ✅ **Better naming** - "evolve" is more descriptive than "optimize_anything"
6. ✅ **Structured evaluation format** - clear WorkloadResult pattern guides users
7. ✅ **More focused** - hides complexity while exposing power via **kwargs

However, **both APIs are superior to OpenEvolve/ShinkaEvolve** because they provide:
- ✅ **Workload-level granularity** - scores per individual workload, not just aggregate
- ✅ **No filesystem coupling** - work with in-memory strings
- ✅ **Dynamic context** - include different context per workload
- ✅ **Full evaluation flexibility** - not restricted to file-based evaluators
- ✅ **Arbitrary trajectories** - handle any system, not just code execution

---

## Detailed API Comparison

### 1. `evolve()` API (optimize.py) ⭐ RECOMMENDED

```python
from gepa import evolve

result = evolve(
    seed_candidate={"prompt": "Initial prompt..."},
    trainset=[...],
    evaluate=evaluate_fn,
    reflection_prompt="Custom instruction...",  # or dict for per-component
    num_iterations=50,
    minibatch_size=25,
    teacher_lm="openai/gpt-4o",
    output_dir="./results",
    verbose=True,
)
```

#### Strengths:

**1. Simple, focused interface:**
- Only essential parameters exposed by default
- Power users can access advanced features via **kwargs
- Clear separation of concerns

**2. Flexible reflection prompts:**
```python
# Single prompt for all components
reflection_prompt="Improve this prompt for math tutoring."

# Or per-component prompts
reflection_prompt={
    "system_prompt": "Improve the agent's reasoning...",
    "tool_descriptions": "Make tool descriptions clearer...",
}
```

**3. Structured evaluation format:**
```python
def evaluate(candidate, batch):
    results = []
    for workload in batch:
        # Your system logic here...
        results.append({
            "score": 0.85,  # Higher is better
            "context_and_feedback": {
                "inputs": ...,
                "outputs": ...,
                "feedback": "What went wrong...",
                # ... any other context for reflection
            }
        })
    return results
```

**4. Excellent documentation:**
- Comprehensive docstring with 3 detailed examples
- Covers prompt optimization, code optimization, multi-component systems
- Clear explanation of workload-level granularity

**5. Clean return value:**
```python
{
    "best_candidate": {"prompt": "Optimized prompt..."},
    "best_score": 0.92,
    "history": [...],  # Full GEPAResult for inspection
    "pareto_frontier": [...],
    "output_dir": "./results"
}
```

#### Weaknesses:

1. Less explicit about all configuration options (hidden in **kwargs)
2. Requires understanding of structured return format
3. `_OptimizeAdapter` is an internal implementation detail

---

### 2. `optimize_anything()` API (api.py)

```python
from gepa import optimize_anything

result = optimize_anything(
    seed_candidate={"prompt": "Initial prompt..."},
    trainset=[...],
    evaluate=evaluate_fn,
    context_to_feedback=feedback_fn,  # Optional converter
    reflection_lm="gpt-4",
    reflection_minibatch_size=25,
    max_metric_calls=100,
    # Many more parameters...
    candidate_selection_strategy="pareto",
    batch_sampler="epoch_shuffled",
    module_selector="round_robin",
    use_merge=False,
    # ... etc
)
```

#### Strengths:

**1. Explicit control:**
- All GEPA configuration parameters visible
- No hidden abstractions
- Direct access to internal mechanisms

**2. Flexible evaluation:**
```python
# Separate context extraction
def evaluate(candidate, batch):
    # Return (score, context) tuples
    return [(0.85, {"inputs": ..., "outputs": ...}), ...]

def context_to_feedback(candidate, trace, workload, score):
    # Custom feedback generation
    return {"feedback": "..."}
```

**3. Full return value:**
- Returns complete `GEPAResult` object with all internal state

#### Weaknesses:

1. **Too many parameters** - overwhelming for new users
2. **Poor naming** - "optimize_anything" is generic, "evolve" is clearer
3. **Complex return type** - GEPAResult object requires understanding internals
4. **Less flexible reflection** - single `reflection_prompt_template` string
5. **Separate feedback function** - adds complexity for simple use cases

---

## Comparison to Competing Frameworks

### OpenEvolve Pattern:

```python
# OpenEvolve requires:
# 1. Write initial program to file
# 2. Create separate evaluator.py with evaluate(program_path) -> dict
# 3. Configure via YAML
# 4. Run evolution on filesystem

# evaluator.py
def evaluate(program_path):
    # Load and execute program from file
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run and return single aggregate score
    result = program.run_packing()
    return {"combined_score": calculate_score(result)}
```

**Limitations:**
- ❌ **Filesystem-coupled** - must write/read files
- ❌ **Single aggregate score** - no workload-level feedback
- ❌ **Code-focused** - designed for program evolution, not flexible strings
- ❌ **Config-heavy** - YAML configuration required
- ❌ **No dynamic context** - same evaluation for all candidates

### ShinkaEvolve Pattern:

Similar to OpenEvolve but with additional complexity:
- Islands and migration
- Archive management  
- Meta-learning with scratchpads
- Even more YAML configuration

**Same core limitations as OpenEvolve.**

---

## Why GEPA is Better

Both GEPA APIs provide **fundamental advantages** over OpenEvolve/ShinkaEvolve:

### 1. Workload-Level Granularity

**GEPA:**
```python
def evaluate(candidate, batch):
    results = []
    for workload in batch:
        score = run_on_workload(candidate, workload)
        context = get_workload_context(workload)
        results.append((score, context))
    return results  # Individual scores + context per workload
```

**OpenEvolve:**
```python
def evaluate(program_path):
    # Run on ALL workloads, return single aggregate
    return {"combined_score": 0.85}
```

GEPA sees performance on **each individual workload**, enabling:
- Better reflection with workload-specific feedback
- Pareto frontier across different workload performance
- Dynamic context per workload

### 2. No Filesystem Coupling

**GEPA:**
```python
# Work with in-memory strings
seed_candidate = {"prompt": "...", "code": "def foo(): ..."}
```

**OpenEvolve:**
```python
# Must write to files
with open("initial_program.py", "w") as f:
    f.write(code)
```

GEPA allows:
- Faster iteration (no disk I/O)
- Better testability
- Easier integration with existing systems
- No temporary file management

### 3. Dynamic Context

**GEPA:**
```python
def evaluate(candidate, batch):
    for workload in batch:
        # Include only relevant context for THIS workload
        relevant_docs = get_docs_for_workload(workload)
        context = {
            "relevant_docs": relevant_docs,  # Dynamic!
            "workload_type": workload.type,
            # ...
        }
        results.append((score, context))
```

**OpenEvolve:**
```python
# Static evaluation, same context for everything
def evaluate(program_path):
    return {"combined_score": ...}
```

### 4. Arbitrary Systems

**GEPA:**
```python
# Evolve anything with string components:
# - Prompts
# - Code
# - Configurations
# - Documents
# - API schemas
# - Even numeric configs (as strings)
seed_candidate = {
    "system_prompt": "...",
    "tool_schema": '{"name": ...}',
    "hyperparams": "learning_rate=0.01,batch_size=32"
}
```

**OpenEvolve:**
```python
# Designed for code evolution
# initial_program.py must define run_search() or similar
```

---

## Recommendations for Improvement

While `evolve()` is the better API, here are suggestions to make it even better:

### 1. Support Direct Metric Dicts (like OpenEvolve)

Some users may want to return metrics directly without the structured format:

```python
def evaluate(candidate, batch):
    # Option 1: Current structured format (preferred)
    return [{"score": 0.85, "context_and_feedback": {...}}, ...]
    
    # Option 2: Simple scores (for quick prototyping)
    return [0.85, 0.92, 0.78]  # Just scores
    
    # Option 3: Multiple metrics (like OpenEvolve)
    return [{
        "primary_metric": 0.85,
        "validity": 1.0,
        "efficiency": 0.92,
        # GEPA uses primary_metric as score
    }, ...]
```

### 2. Optional Aggregation Control

For workloads like circle_packing where there's one "run" but multiple evaluation criteria:

```python
def evaluate(candidate, batch):
    # batch might be [{"n_circles": 26}] - single workload
    # but evaluation returns multiple metrics
    return [{
        "score": 0.85,  # Primary score
        "context_and_feedback": {
            "sum_radii": 2.635,
            "validity": 1.0,
            "eval_time": 45.2,
            # Multiple metrics, single workload
        }
    }]
```

### 3. Add Helper for File-Based Systems

For users evolving actual code files (like OpenEvolve users):

```python
from gepa import evolve_code_file

result = evolve_code_file(
    initial_code_file="initial.py",
    evaluator_file="evaluator.py",  # Has evaluate(code_str) -> score
    num_iterations=50,
    # ... same as evolve()
)

# Under the hood, converts to evolve() format:
# - Reads initial code as seed_candidate={"code": read(initial_code_file)}
# - Wraps evaluator to work with candidate dict
# - Returns results
```

### 4. Better Examples in README

Add diverse examples showing:
1. ✅ Prompt optimization (already have)
2. ⚠️ Code optimization (need better example)
3. ⚠️ Circle packing / function minimization (need to add)
4. ⚠️ Multi-component system (need to add)
5. ⚠️ Document evolution for RAG (need to add)

### 5. Clearer Error Messages

When user's evaluate function returns wrong format:

```python
# Bad return format:
return [0.85, 0.92]  # Just floats

# Error should say:
# "evaluate() must return List[Dict] with 'score' and 'context_and_feedback' keys.
# Got: List[float]. Did you mean: [{'score': 0.85, 'context_and_feedback': {}}, ...]?"
```

---

## Specific Use Cases

### Circle Packing (from OpenEvolve examples)

**With `evolve()`:**

```python
from gepa import evolve

seed_candidate = {
    "packing_algorithm": """
def run_packing():
    # Initial simple algorithm
    ...
    return centers, radii, sum_radii
"""
}

trainset = [{"n_circles": 26}]  # Could have multiple configs

def evaluate(candidate, batch):
    results = []
    for config in batch:
        # Execute the code
        exec_globals = {}
        exec(candidate["packing_algorithm"], exec_globals)
        
        try:
            centers, radii, sum_radii = exec_globals["run_packing"]()
            
            # Validate
            valid = validate_packing(centers, radii)
            target = 2.635
            score = (sum_radii / target) if valid else 0.0
            
            results.append({
                "score": score,
                "context_and_feedback": {
                    "sum_radii": sum_radii,
                    "target": target,
                    "valid": valid,
                    "feedback": f"Sum: {sum_radii:.3f}, Target: {target}, Valid: {valid}"
                }
            })
        except Exception as e:
            results.append({
                "score": 0.0,
                "context_and_feedback": {
                    "error": str(e),
                    "feedback": f"Execution failed: {str(e)}"
                }
            })
    
    return results

result = evolve(
    seed_candidate=seed_candidate,
    trainset=trainset,
    evaluate=evaluate,
    reflection_prompt="Improve the circle packing algorithm to maximize sum of radii.",
    num_iterations=100,
    teacher_lm="gpt-4o",
)
```

**Comparison to OpenEvolve:**
- ✅ More flexible - no file I/O required
- ✅ Easier to customize evaluation
- ✅ Can include rich context per iteration
- ✅ Can evolve multiple functions simultaneously

### Attention Optimization (from OpenEvolve examples)

```python
from gepa import evolve

seed_candidate = {
    "attention_kernel": """
def attention_forward(Q, K, V, mask=None):
    # Initial implementation
    scores = Q @ K.T
    if mask is not None:
        scores = scores + mask
    attn = softmax(scores, dim=-1)
    return attn @ V
"""
}

trainset = [
    {"seq_len": 128, "batch": 32, "dim": 512},
    {"seq_len": 512, "batch": 16, "dim": 512},
    {"seq_len": 1024, "batch": 8, "dim": 512},
]

def evaluate(candidate, batch):
    results = []
    for config in batch:
        # Benchmark the kernel
        exec_globals = {"torch": torch, "softmax": F.softmax}
        exec(candidate["attention_kernel"], exec_globals)
        
        latency = benchmark_kernel(
            exec_globals["attention_forward"],
            config
        )
        
        # Lower latency is better, convert to score
        baseline_latency = 10.0  # ms
        score = baseline_latency / latency  # >1 means better than baseline
        
        results.append({
            "score": score,
            "context_and_feedback": {
                "latency_ms": latency,
                "config": config,
                "speedup": f"{score:.2f}x vs baseline",
                "feedback": f"Latency: {latency:.2f}ms on {config}"
            }
        })
    
    return results

result = evolve(
    seed_candidate=seed_candidate,
    trainset=trainset,
    evaluate=evaluate,
    reflection_prompt="Optimize the attention kernel for lower latency while maintaining correctness.",
    num_iterations=200,
)
```

---

## Conclusion

**The `evolve()` API in `optimize.py` is the clear winner** because it:

1. **Balances simplicity and power** - easy for beginners, powerful for experts
2. **Better matches user mental models** - "evolve" is intuitive
3. **Cleaner abstractions** - hides complexity appropriately
4. **More flexible** - per-component reflection, structured feedback
5. **Better documentation** - comprehensive examples

**Both GEPA APIs are vastly superior to OpenEvolve/ShinkaEvolve** because:

1. **Workload-level granularity** vs aggregate scores
2. **No filesystem coupling** vs file-based evolution
3. **Dynamic context** vs static evaluation
4. **Arbitrary systems** vs code-focused

### Final Recommendation:

1. ✅ **Use `evolve()` as the primary API**
2. ✅ **Keep `optimize_anything()` for power users** who need explicit control
3. ✅ **Add the suggested improvements** to make it even better
4. ✅ **Add diverse examples** (circle packing, function minimization, etc.)
5. ✅ **Create migration guide** for OpenEvolve users

