# Circle Packing Multi-N with GEPA

This example demonstrates using GEPA's `evolve` API to solve the circle packing problem across multiple problem sizes simultaneously. This is a powerful approach that allows the system to learn general strategies that work across different values of n.

## Problem Description

**Circle Packing Problem**: Pack n non-overlapping circles into a unit square (0,0) to (1,1), maximizing the sum of their radii.

**Constraints**:
1. All circles must be fully inside the unit square
2. No circles may overlap
3. All radii must be positive
4. Must return exactly n circles

## Multi-Workload Approach

Unlike traditional approaches that optimize for a single n value, this implementation uses **multiple n values as different workloads**:

```python
trainset = [7, 10, 21, 22, 26, 28, 29, 31, 32, 33, 52, 68, 99, 143, 216, 446, 992]
```

This allows GEPA to:
- **Learn general strategies** that work across different problem sizes
- **Transfer insights** from easier problems (small n) to harder ones (large n)
- **Discover scalable algorithms** rather than problem-specific hacks
- **Improve robustness** by testing on diverse workloads

## Implementation Overview

### 1. Seed Candidate

The initial implementation uses a simple grid-based placement:

```python
def construct_packing(n):
    # Place circles in a grid
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)
    
    # Compute maximum radii without overlaps
    # (Details in the code)
    
    return centers, radii, sum_radii
```

This achieves ~45-50% of optimal for small n values.

### 2. Evaluation Function

The `evaluate_circle_packing` function:
- Runs code in isolated subprocess with timeout
- Validates all geometric constraints
- Provides detailed feedback on failures
- Handles errors gracefully

**Validation checks**:
- Correct array shapes (n, 2) for centers and (n,) for radii
- No NaN values
- All radii positive
- All circles inside unit square
- No overlapping circles (with numerical tolerance)

**Scoring**:
```python
score = actual_sum / target_value
```
Where target values come from literature (AlphaEvolve paper and estimates).

### 3. Reflection Prompt

Custom prompt guides the LLM to improve the packing function:

```python
REFLECTION_PROMPT = """You are an expert mathematician and computational geometry specialist...

Available packages:
- numpy (imported as np)
- scipy (all submodules)
- matplotlib (for visualization, if needed)

Key insights for improvement:
1. Algorithmic approaches: constructor-based, optimization-based, 
   physics-based (force simulation), or hybrid
2. Geometric patterns: hexagonal packing is known to be dense for infinite planes; 
   consider edge effects for squares
3. Circle sizes: variable-sized circles may allow better space utilization than uniform sizes
4. Edge effects: corners and edges constrain circle placement differently than the interior
5. Scalability: solution should work efficiently for both small (n<30) and large (n>100)
6. Optimization: if using optimization-based approaches, good initial placement 
   can significantly improve results

{examples}

Based on the examples above, improve the construct_packing function...
"""
```

### 4. GEPA Configuration

```python
result = evolve(
    seed_candidate={"packing_function": INITIAL_PACKING_CODE},
    trainset=[7, 10, 21, 22, 26, ...],  # Multiple n values
    evaluate=evaluate_circle_packing,
    reflection_prompt=REFLECTION_PROMPT,
    num_iterations=100,
    minibatch_size=5,  # Evaluate on 5 random n values per iteration
    teacher_lm="anthropic/claude-3.7-sonnet",
    num_threads=3,  # Parallel evaluation
    minibatch_full_eval_steps=10,  # Full eval every 10 steps
)
```

## Running the Example

### Prerequisites

```bash
# Install dependencies (same as OpenEvolve/ShinkaEvolve)
pip install -r examples/evolve_anything/circle_packing_requirements.txt

# Or install manually
pip install numpy scipy matplotlib

# Make sure GEPA is installed
pip install -e .
```

**Available packages in evolved code:**
- `numpy` - Array operations, mathematical functions
- `scipy` - Scientific computing, optimization, etc.
- `matplotlib` - Visualization (optional, not used in evaluation)

### Quick Test

Test the implementation without running full evolution:

```bash
python examples/evolve_anything/06_circle_packing_test.py
```

Expected output:
```
Testing validation function...
Valid packing test: True
...
Testing initial packing code...
n=7:
  Score: 0.4844
  Valid: True
  Sum radii: 0.831250
  Target: 1.716000
...
All tests completed!
```

### Full Evolution

Run the full optimization (this will take significant time and API credits):

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Run evolution
python examples/evolve_anything/06_circle_packing_multi_n.py
```

**Note**: This will make many API calls and may take hours to complete. Consider:
- Starting with fewer iterations: modify `num_iterations=100` to `num_iterations=20`
- Using fewer workloads: modify the trainset to `[7, 10, 26]`
- Using a cheaper model: change to `teacher_lm="openai/gpt-4o-mini"`

### Output

The evolution will create an `circle_packing_output/` directory containing:
- Checkpoint files at regular intervals
- Best candidate found so far
- Detailed logs and metrics
- Evolution history

Final output shows results for all n values:
```
Final evaluation on all n values:

   n |    Score | Sum Radii |     Target | Status
------------------------------------------------------------
   7 |   0.9823 |   1.686541 |      1.716 | ✓ Valid
  10 |   0.9765 |   2.071248 |      2.121 | ✓ Valid
  26 |   0.9997 |   2.634292 |      2.635 | ✓ Valid
...
```

## Expected Evolution Trajectory

Based on OpenEvolve and ShinkaEvolve results, the evolution typically progresses:

1. **Generation 0-10**: Grid-based improvements
   - Better spacing calculations
   - Corner optimization
   - Score: ~50-60% of target

2. **Generation 10-30**: Geometric patterns
   - Hexagonal arrangements
   - Ring-based patterns
   - Variable circle sizes
   - Score: ~70-80% of target

3. **Generation 30-60**: Optimization-based approaches
   - May discover mathematical optimization libraries
   - Constrained optimization formulation
   - Score: ~85-95% of target

4. **Generation 60-100**: Fine-tuning and scaling
   - Better initial guesses
   - Adaptive strategies for different n
   - Score: ~95-99% of target

## Key Design Decisions

### 1. Multi-N Training

**Why multiple n values?**
- Prevents overfitting to a single problem size
- Encourages discovery of general algorithms
- Allows learning from easier problems

**Workload selection**:
- Mix of small (n<30), medium (30<n<100), and large (n>100)
- Includes n=26 from AlphaEvolve paper for comparison
- Known optimal values for scoring

### 2. Subprocess Execution

**Why subprocess?**
- Timeout protection (some algorithms may hang)
- Memory isolation (prevents memory leaks)
- Clean error handling
- Prevents infinite loops from crashing main process

**Timeouts**:
- Small n (≤33): 60 seconds
- Medium n (34-100): 120 seconds
- Large n (>100): 180 seconds

### 3. Detailed Validation

**Why so thorough?**
- Invalid solutions should score 0.0 immediately
- Detailed feedback helps LLM understand failures
- Catches subtle bugs (numerical precision, off-by-one, etc.)

**Validation levels**:
1. Shape validation (correct array dimensions)
2. Value validation (no NaN, no negative radii)
3. Boundary validation (circles inside square)
4. Overlap validation (no collisions)
5. Statistics (min/max/avg radii for feedback)

### 4. Flexible API Enhancement

The `optimize.py` file was enhanced to support:
- Per-component reflection prompts via dict
- Dynamic prompt selection based on component name
- Better error messages from evaluate function

## Comparison with Other Approaches

### OpenEvolve
- Uses two-phase approach (exploration → exploitation)
- Single n value (n=26)
- Population-based with islands
- Achieved 2.634/2.635 (99.97%)

### ShinkaEvolve
- Multi-model LLM selection with UCB1
- Diff, full, and cross patch types
- Meta-learning for self-improvement
- 400 generations

### GEPA Multi-N (This Implementation)
- **Unique**: Multi-workload learning
- Simpler API (just define evaluate function)
- Minibatch sampling from workloads
- Transfer learning across problem sizes
- Fewer hyperparameters to tune

## Troubleshooting

### "Module not found" errors
Make sure GEPA is installed: `pip install -e .` from the repo root.

### Timeout errors
Some optimization algorithms can be slow for large n:
- Start with small n values (7, 10, 26)
- Reduce iteration counts if using iterative optimization
- Use faster algorithms (heuristics instead of optimization)

### Low scores
Initial code is intentionally simple (~45% of optimal):
- Give it time to evolve (50+ iterations)
- Check that reflection prompt is being used
- Verify API key is set correctly
- Look at evolution logs for progress

### Memory issues
Subprocess isolation helps, but very large n may still cause issues:
- Remove largest n values (446, 992) from trainset
- Reduce batch size
- Add memory monitoring

## Next Steps

1. **Run full evolution**: See how close you can get to AlphaEvolve results
2. **Analyze progression**: Study how the algorithm evolves over time
3. **Try different models**: Compare Claude, GPT-4, Gemini
4. **Extend workloads**: Add more n values or different container shapes
5. **Multi-objective**: Optimize for both sum of radii and computation time

## References

- AlphaEvolve paper: Circle packing n=26 baseline (2.635)
- OpenEvolve example: Two-phase evolution approach
- ShinkaEvolve example: Multi-model LLM selection
- GEPA paper: Genetic Evolution via Prompt Adaptation

## Files

- `06_circle_packing_multi_n.py`: Main implementation
- `06_circle_packing_test.py`: Unit tests and validation
- `CIRCLE_PACKING_README.md`: This file
- `../src/gepa/optimize.py`: Enhanced evolve API

---

**Happy evolving!** 🔬🎯

