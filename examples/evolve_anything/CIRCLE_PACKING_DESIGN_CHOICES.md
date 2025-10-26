# Circle Packing: Design Choices and Configuration Analysis

This document explains the design choices made for the GEPA circle packing implementation, based on analysis of OpenEvolve and ShinkaEvolve implementations.

## Implementation Comparison

### Configuration Analysis

| Aspect | OpenEvolve | ShinkaEvolve | GEPA Multi-N | Rationale |
|--------|------------|--------------|--------------|-----------|
| **LLM Model** | Gemini 2.0 Flash (80%) + Claude Sonnet (20%) | Gemini 2.5 Pro/Flash, Claude Sonnet 4, GPT-5/o4-mini | Claude Sonnet 3.7 | Claude best for math reasoning |
| **Iterations** | 200 (100+100 phases) | 400 | 100 | Balanced - can extend if needed |
| **Problem Size** | n=26 only | n=26 only | n=[7..992] | **Novel multi-workload approach** |
| **Population** | 60-70 programs | 2 islands, archive 40 | N/A (reflection-based) | Different paradigm |
| **Timeout** | 600s | 600s | 60-180s (adaptive) | Faster feedback, adaptive scaling |
| **Evaluation** | Subprocess + pickle | Shinka eval framework | Subprocess + pickle | Reliable, proven approach |
| **Evolution Style** | Population + islands | Multi-model + patches | Reflection + minibatch | GEPA's strength |
| **Best Result** | 2.634/2.635 (99.97%) | Unknown | TBD | Goal: match or exceed |

## Key Design Decisions

### 1. LLM Selection: Claude Sonnet 3.7

**OpenEvolve used**:
```yaml
primary_model: "google/gemini-2.0-flash-001"  # 80%
secondary_model: "anthropic/claude-3.7-sonnet"  # 20%
```

**ShinkaEvolve used**:
```python
llm_models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash", 
    "bedrock/us.anthropic.claude-sonnet-4",
    "o4-mini",
    "gpt-5",
]
```

**Our choice**: Claude Sonnet 3.7 (100%)

**Rationale**:
- OpenEvolve's 80/20 split suggests Gemini for speed, Claude for quality
- For mathematical reasoning, Claude excels
- Reflection-based approach needs high-quality reasoning
- Can switch to Gemini for production speed after prototyping
- UCB1 model selection (from Shinka) is overkill for initial implementation

### 2. System Prompt: Geometric Insights

**OpenEvolve Phase 1**:
```yaml
system_message: |
  You are an expert mathematician specializing in circle packing problems...
  
  Key geometric insights:
  - Circle packings often follow hexagonal patterns
  - Maximum density for infinite packing is pi/(2*sqrt(3))
  - Edge effects make square container packing harder
  - Circles can be placed in layers or shells
  - Similar radius circles often form regular patterns
  - Variable radii allow better space utilization
  
  Focus on designing an explicit constructor...
```

**OpenEvolve Phase 2**:
```yaml
system_message: |
  We're trying to reach 2.635 for n=26. 
  Current implementation has plateaued at 2.377...
  
  Key insights to explore:
  1. Variable-sized circles
  2. Hybrid arrangements (not pure hexagonal)
  3. Physics-based optimization with tuned parameters
  4. Strategic placement at corners and edges
  5. Larger circles at center, smaller at edges
  6. Special arrangements for specific n values
  
  Focus on breaking through the plateau...
```

**ShinkaEvolve**:
```python
search_task_sys_msg = """
You are an expert mathematician specializing in circle packing...
Best known result: 2.635

Key directions:
1. Variable-sized circles
2. Hybrid approach (not pure hexagonal)
3. Physics-based optimization
4. Strategic corner/edge placement
5. Pattern adjustments (larger center, smaller edges)
6. Special arrangements for specific n
7. Use scipy optimize (LP or SLSQP)

Make sure all circles are disjoint and inside unit square.
Be creative and try to find better than 2.635.
"""
```

**Our choice**: Combined insights + multi-n emphasis

```python
REFLECTION_PROMPT = """
You are an expert mathematician and computational geometry specialist...

Available packages:
- numpy (imported as np)
- scipy (all submodules)
- matplotlib (for visualization, if needed)

Key insights for improvement:
1. Algorithmic approaches: constructor, optimization, physics, hybrid
2. Geometric patterns: hexagonal packing is dense for infinite planes; adapt for squares
3. Circle sizes: variable-sized circles may utilize space better than uniform sizes
4. Edge effects: corners/edges constrain placement differently than interior
5. Scalability: must work for n<30 AND n>100  # ← NEW!
6. Optimization: if using optimization-based approaches, good initial placement helps

Based on examples above, improve construct_packing...
"""
```

**Rationale**:
- Synthesizes insights from both implementations
- Adds scalability requirement (for multi-n)
- Keeps general algorithmic directions without prescribing specific solutions
- Shorter and more focused than OpenEvolve's prompts
- Emphasizes learning from examples (cross-workload)

### 3. Timeout Strategy: Adaptive

**OpenEvolve**:
```python
timeout_seconds=600  # 10 minutes for all n=26 evaluations
```

**ShinkaEvolve**:
```python
# Not explicitly shown, but eval framework handles it
```

**Our choice**: Adaptive based on n
```python
if n <= 33:
    timeout = 60      # 1 minute for small n
elif n <= 100:
    timeout = 120     # 2 minutes for medium n
else:
    timeout = 180     # 3 minutes for large n
```

**Rationale**:
- 600s is too long for quick iterations
- Small n problems should be fast
- Large n legitimately needs more time
- Faster feedback → faster evolution
- Can still discover complex algorithms in 60s for small n

### 4. Validation: More Detailed

**OpenEvolve**:
```python
def validate_packing(centers, radii):
    # Check NaN
    # Check negative radii
    # Check boundaries
    # Check overlaps
    return True/False
```

**ShinkaEvolve**:
```python
def adapted_validate_packing(run_output):
    # Check shapes
    # Check negative radii
    # Check reported_sum matches calculated
    # Check boundaries  
    # Check overlaps
    return (is_valid, error_message)
```

**Our choice**: Detailed diagnostics
```python
def validate_packing(n, centers, radii):
    validation_details = {
        "expected_circles": n,
        "boundary_violations": [],  # List of specific violations
        "overlaps": [],             # List of specific pairs
        "nan_detected": False,
        "negative_radii": [],
        "shape_errors": [],
        "min_radius": ...,          # Statistics
        "max_radius": ...,
        "avg_radius": ...,
    }
    return (is_valid, validation_details)
```

**Rationale**:
- More details → better LLM feedback
- Lists specific violations, not just count
- Statistics help understand solution quality
- Separates different error types
- Essential for debugging during evolution

### 5. Multi-N Workloads: Novel Contribution

**OpenEvolve**: Single n=26

**ShinkaEvolve**: Single n=26

**Our choice**: Multiple n values [7, 10, 21, ..., 992]

**Rationale**:
- **Transfer learning**: Small n insights help large n
- **Generalization**: Prevents overfitting to n=26
- **Robustness**: Tests algorithm across scales
- **Efficiency**: Learn from easy problems first
- **Novel**: Neither implementation does this

**Workload selection rationale**:
```python
trainset = [
    7, 10,           # Small - easy to solve, fast feedback
    21, 22, 26,      # Medium - n=26 is main target
    28, 29, 31, 32, 33,  # Medium-large - test scaling
    52, 68, 99,      # Large - require efficient algorithms
    143, 216,        # Very large - test true scalability
    446, 992         # Huge - stress test (optional)
]
```

### 6. Evaluation Strategy: Minibatch Sampling

**OpenEvolve**: Full evaluation every time (but single n)

**ShinkaEvolve**: Full evaluation every time (but single n)

**Our choice**: Minibatch with periodic full eval
```python
minibatch_size=5,                    # Evaluate 5 random n values
minibatch_full_eval_steps=10,        # Full eval every 10 iterations
```

**Rationale**:
- 17 workloads × 60-180s = too slow for every iteration
- Random sampling gives good coverage
- Periodic full eval tracks true progress
- Balances speed and accuracy
- GEPA's strength: efficient minibatching

### 7. Evolution Parameters

**OpenEvolve Phase 1**:
```yaml
max_iterations: 100
population_size: 60
num_islands: 4
elite_selection_ratio: 0.3
exploitation_ratio: 0.7
```

**OpenEvolve Phase 2**:
```yaml
max_iterations: 100
population_size: 70
num_islands: 5
elite_selection_ratio: 0.3
exploitation_ratio: 0.6  # More exploration
```

**ShinkaEvolve**:
```python
num_generations = 400
num_islands = 2
archive_size = 40
parent_selection_strategy = "weighted"
parent_selection_lambda = 10.0
```

**Our choice**:
```python
num_iterations = 100        # Can extend if needed
minibatch_size = 5          # Sample 5 workloads per iteration
num_threads = 3             # Parallel evaluation
```

**Rationale**:
- Different paradigm (reflection vs. population)
- 100 iterations = good starting point
- Fewer parameters to tune (simplicity)
- Parallelism for speed
- Can increase if not converged

### 8. Progress Tracking

**OpenEvolve**:
- Checkpoints every 10 iterations
- Saves best program
- Tracks combined_score
- Generates visualizations (manual)

**ShinkaEvolve**:
- Saves results to SQLite database
- Tracks all programs and scores
- Island-based organization
- Meta-learning metrics

**Our choice**:
```python
output_dir="./circle_packing_output"
verbose=True
minibatch_full_eval_steps=10
```

**Rationale**:
- GEPA handles checkpointing automatically
- Verbose mode shows progress
- Full eval every 10 steps gives clear metrics
- Simpler than database (for this use case)
- Can add custom logging if needed

## Configuration Recommendations

### For Quick Testing (< 30 min, < $5)
```python
trainset = [7, 10, 26]
num_iterations = 20
minibatch_size = 2
teacher_lm = "openai/gpt-4o-mini"
```

### For Good Results (2-3 hours, $20-30)
```python
trainset = [7, 10, 21, 22, 26, 28, 29, 31, 32, 33]
num_iterations = 100
minibatch_size = 5
teacher_lm = "anthropic/claude-3.7-sonnet"
```

### For Publication-Quality (1 day, $50-100)
```python
trainset = [7, 10, 21, 22, 26, 28, 29, 31, 32, 33, 52, 68, 99, 143, 216]
num_iterations = 200
minibatch_size = 5
teacher_lm = "anthropic/claude-3.7-sonnet"
# Run multiple times with different seeds
```

## Novel Contributions

1. **Multi-workload training**: First implementation to train on multiple n simultaneously
2. **Adaptive timeout**: Scales timeout based on problem size
3. **Detailed validation**: Returns comprehensive diagnostics for better feedback
4. **Transfer learning**: Explicitly designed to learn from small n to help large n
5. **Simplified API**: Single `evolve()` call vs. complex configuration files

## Expected Improvements Over Baselines

| Metric | Expected Improvement | Reason |
|--------|---------------------|--------|
| **Generalization** | ✅ Much better | Multi-n training |
| **Scalability** | ✅ Better | Tested on n up to 992 |
| **Simplicity** | ✅ Much simpler | Fewer hyperparameters |
| **Speed per iteration** | ⚠️ Slower | Multiple workloads |
| **Best n=26 result** | 🎯 ~Same | Should match OpenEvolve (2.634) |
| **Overall efficiency** | ✅ Better | Learns general algorithm |

## Potential Weaknesses

1. **Slower iterations**: 17 workloads vs. 1
   - Mitigation: Minibatch sampling
   
2. **Target value uncertainty**: Some n values use estimates
   - Mitigation: Use conservative estimates, focus on n=26
   
3. **No population diversity**: Single candidate at a time
   - Mitigation: GEPA's reflection handles exploration
   
4. **Cold start**: Initial code is simple
   - Mitigation: Could warm-start with OpenEvolve's best

## Conclusion

Our GEPA implementation synthesizes the best practices from both OpenEvolve and ShinkaEvolve:

**From OpenEvolve**:
- ✅ Geometric insights in prompt
- ✅ Subprocess isolation
- ✅ Comprehensive validation
- ✅ Two-phase evolution strategy (exploration then exploitation)

**From ShinkaEvolve**:
- ✅ Multi-model awareness (chose best one)
- ✅ Scalability emphasis
- ✅ Multiple patch types concept (GEPA's reflection)
- ✅ Detailed evaluation framework

**Our Novel Additions**:
- 🌟 Multi-workload training (n=7 to n=992)
- 🌟 Adaptive timeouts
- 🌟 Enhanced validation diagnostics
- 🌟 Simplified API (single `evolve()` call)
- 🌟 Transfer learning across problem sizes

**Result**: A simpler, more general, more robust implementation that should match OpenEvolve's n=26 results while also discovering algorithms that scale well to much larger problem sizes.

