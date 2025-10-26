# GEPA Evolve-Anything Examples

This directory contains examples demonstrating GEPA's `evolve()` API - a simple, flexible interface for evolving **anything with string components**.

## Philosophy

The `evolve()` API is designed on a simple principle: **if it can be represented as text and evaluated, GEPA can evolve it**.

```python
from gepa import evolve

result = evolve(
    seed_candidate={"component": "your initial text"},
    trainset=[...],  # your workloads
    evaluate=your_eval_function,  # scores candidates
)
```

That's it. No adapters to implement, no complex protocols - just provide your strings, workloads, and evaluation logic.

## Examples

### 1. Prompt Optimization (`01_prompt_optimization_hotpotqa.py`)

**What it evolves**: Question-answering prompts  
**Workload**: HotpotQA multi-hop reasoning questions  
**Learns to**: Extract and combine information from context, handle comparison questions

```bash
python 01_prompt_optimization_hotpotqa.py
```

**Key insight**: Starting from "Answer the question based on context", GEPA discovers prompt improvements that emphasize careful context analysis and concise answers.

---

### 2. Function Minimization Algorithms (`02_function_minimization.py`)

**What it evolves**: Optimization algorithm code  
**Workload**: Benchmark functions (Rastrigin, Rosenbrock, Ackley, Sphere)  
**Learns to**: Escape local minima, adapt step sizes, balance exploration vs exploitation

```bash
python 02_function_minimization.py
```

**Key insight**: Starting from random search, GEPA evolves sophisticated techniques like simulated annealing, momentum-based exploration, and adaptive cooling schedules.

---

### 3. API Schema Evolution (`03_api_schema_evolution.py`)

**What it evolves**: API request/response schemas (JSON)  
**Workload**: Developer usage scenarios with pain points  
**Learns to**: Add validation rules, mark required fields, provide examples, document constraints

```bash
python 03_api_schema_evolution.py
```

**Key insight**: Starting from a bare-bones schema, GEPA discovers improvements that make APIs more intuitive - adding type information, examples, validation rules, and clear documentation.

---

### 4. Tutorial Content Evolution (`04_tutorial_evolution.py`)

**What it evolves**: Educational content (recursion tutorial)  
**Workload**: Simulated students at different skill levels  
**Learns to**: Adapt explanations to audience, use better examples, address common misconceptions

```bash
python 04_tutorial_evolution.py
```

**Key insight**: Starting from a technical tutorial, GEPA learns to use relatable analogies, provide step-by-step breakdowns, and address practical debugging concerns.

---

### 5. Circle Packing Optimization (`05_circle_packing.py`)

**What it evolves**: Circle packing algorithms (from OpenEvolve example)  
**Workload**: Single task - pack 26 circles to maximize radii sum  
**Learns to**: Discover sophisticated algorithms from simple concentric rings to mathematical optimization

```bash
python 05_circle_packing.py
```

**Key insight**: Starting from basic geometric construction, GEPA discovers advanced techniques like scipy.optimize with SLSQP, hexagonal patterns, and constrained optimization. Demonstrates rich artifacts (execution errors, validation details, performance metrics). Target: 2.635 sum of radii (AlphaEvolve paper result).

---

## Requirements

```bash
# Install GEPA
pip install -e .

# Install LiteLLM for LLM calls (optional for examples 1 and 4)
pip install litellm

# Set your OpenAI API key (or use any OpenAI-compatible provider)
export OPENAI_API_KEY="your-key-here"
```

## The Evolve API in Detail

### Minimal Example

```python
from gepa import evolve

def evaluate(candidate, batch):
    """Your evaluation function."""
    results = []
    for workload in batch:
        # Run your system with candidate["component"]
        score = ...  # Higher is better
        results.append({
            "score": score,
            "context_and_feedback": {
                "inputs": ...,
                "outputs": ...,
                "feedback": "What went wrong/right"
            }
        })
    return results

result = evolve(
    seed_candidate={"component": "initial text"},
    trainset=[...],
    evaluate=evaluate,
)

print(result["best_candidate"])
```

### Key Advantages

1. **Not Tied to Filesystem**: Code can be compiled in-memory, APIs called directly, no file I/O required
2. **Dynamic Context**: Include only relevant information for each workload in feedback
3. **Granular Scoring**: GEPA sees per-workload scores, not just aggregates
4. **Flexible Workloads**: Any Python object - test cases, queries, scenarios, configurations
5. **Custom Reflection**: Optional custom reflection prompts for domain-specific guidance

### What Can You Evolve?

Anything representable as text:

- **Prompts**: System prompts, few-shot examples, instruction templates
- **Code**: Functions, algorithms, GPU kernels, shaders
- **Configs**: JSON schemas, API specs, database schemas, hyperparameters
- **Documents**: Tutorials, API docs, error messages, help text
- **Structured Data**: XML, YAML, Markdown, CSV headers
- **Domain-Specific Languages**: SQL queries, regex patterns, LaTeX templates

## Tips for Creating Your Own Examples

### 1. Design Good Evaluation Functions

```python
def evaluate(candidate, batch):
    results = []
    for workload in batch:
        try:
            # Execute your system
            output = run_system(candidate["component"], workload)
            score = compute_score(output, workload["expected"])
            
            # Provide rich feedback
            feedback = f"Expected: {workload['expected']}\n"
            feedback += f"Got: {output}\n"
            if score < 1.0:
                feedback += "Issue: ..." # Explain what went wrong
            
            results.append({
                "score": score,
                "context_and_feedback": {
                    "inputs": workload["input"],
                    "outputs": output,
                    "feedback": feedback,
                }
            })
        except Exception as e:
            # Handle failures gracefully
            results.append({
                "score": 0.0,
                "context_and_feedback": {
                    "inputs": workload["input"],
                    "outputs": f"Error: {str(e)}",
                    "feedback": "Execution failed. Check syntax/logic.",
                }
            })
    return results
```

### 2. Craft Effective Reflection Prompts

```python
reflection_prompt = """You are optimizing [WHAT] for [GOAL].

Current issues from feedback:
- [Common failure pattern 1]
- [Common failure pattern 2]

Improvement strategies:
1. [Strategy 1]
2. [Strategy 2]

Constraints:
- [What must not change]
- [Required format/structure]

Provide only the improved [WHAT] without explanation."""
```

### 3. Start Simple, Then Evolve

- Begin with a deliberately simple/suboptimal seed
- Use 20-50 iterations for initial exploration
- Adjust `minibatch_size` based on evaluation cost
- Use `random_seed` for reproducibility

### 4. Monitor Evolution

```python
result = evolve(
    ...,
    output_dir="./my_evolution",  # Saves checkpoints
    verbose=True,  # Show progress
)

# Access history
for iteration in result["history"]:
    print(f"Iteration {iteration['i']}: Score {iteration['score']}")
```

## Comparison with OpenEvolve

| Aspect | OpenEvolve | GEPA Evolve |
|--------|-----------|-------------|
| **Input** | Python file paths | In-memory strings/objects |
| **Evaluation** | Separate evaluator file/function | Inline evaluate function |
| **System Messages** | Global system message in config | Per-component reflection prompts |
| **Score Granularity** | Combined score or metrics dict | Per-workload scores |
| **Context** | Static (filesystem) | Dynamic (per-workload) |
| **File I/O** | Required (writes temp files) | Optional (in-memory) |
| **Artifacts** | Separate EvaluationResult object | Embedded in context_and_feedback |
| **Best For** | Full programs, hardware optimization | Components, quick experimentation |
| **Examples** | Circle packing, GPU kernels, etc. | Same circle packing + prompts, schemas |

Both are powerful and complementary! OpenEvolve excels at complete programs with complex execution, while `evolve()` excels at flexible, in-memory optimization. The circle packing example (`05_circle_packing.py`) demonstrates that GEPA's evolve API can handle the same sophisticated algorithm evolution as OpenEvolve, with a simpler interface.

## Next Steps

- **Combine Components**: Evolve multiple components together (e.g., prompt + schema)
- **Custom Metrics**: Add domain-specific scoring beyond simple accuracy
- **Multi-Objective**: Track multiple objectives in feedback
- **Production Integration**: Use evolved components in your applications
- **Share Results**: Open PRs with interesting examples!

## Questions?

See the main [GEPA documentation](../../README.md) or open an issue!

