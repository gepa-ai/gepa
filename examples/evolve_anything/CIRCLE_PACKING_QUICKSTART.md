# Circle Packing Multi-N: Quick Start Guide

This guide will help you get started with the circle packing optimization in under 5 minutes.

## What This Does

Evolves a Python function that packs circles into a unit square, optimizing across **17 different problem sizes** simultaneously (n = 7, 10, 21, 22, 26, 28, 29, 31, 32, 33, 52, 68, 99, 143, 216, 446, 992).

**Goal**: Match or exceed AlphaEvolve's result of 2.635 for n=26.

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/lakshyaaagrawal/Documents/Repos/gepa_dev/gepa_3
pip install -r examples/evolve_anything/circle_packing_requirements.txt

# Or manually: pip install numpy scipy matplotlib
```

**Note**: These are the same packages available in OpenEvolve and ShinkaEvolve.

### 2. Run Tests (30 seconds)

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
...
All tests completed!
```

### 3. Run Evolution (Hours, costs API credits)

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run full evolution
python examples/evolve_anything/06_circle_packing_multi_n.py
```

## Faster Testing Options

### Option A: Fewer Iterations
Edit `06_circle_packing_multi_n.py`, change line:
```python
num_iterations=100,  # Change to 20 for quick test
```

### Option B: Fewer Workloads
Edit `06_circle_packing_multi_n.py`, change:
```python
trainset = [7, 10, 21, 22, 26, 28, 29, ...]  # Change to [7, 10, 26]
```

### Option C: Cheaper Model
Edit `06_circle_packing_multi_n.py`, change:
```python
teacher_lm="anthropic/claude-3.7-sonnet",  # Change to "openai/gpt-4o-mini"
```

## Understanding the Output

### During Evolution
```
Iteration 1/100
  Evaluating on minibatch: [7, 22, 31, 52, 143]
  Current best score: 0.4844
  
Iteration 2/100
  Evaluating on minibatch: [10, 26, 29, 68, 216]
  Current best score: 0.5123  ← Improving!
...
```

### Final Results
```
Final evaluation on all n values:

   n |    Score | Sum Radii |     Target | Status
------------------------------------------------------------
   7 |   0.9823 |   1.686541 |      1.716 | ✓ Valid
  10 |   0.9765 |   2.071248 |      2.121 | ✓ Valid
  26 |   0.9997 |   2.634292 |      2.635 | ✓ Valid  ← This is the main target!
  ...
```

### Output Directory
```
circle_packing_output/
  ├── best_candidate.json          # Best solution found
  ├── evolution_history.json       # Full history
  ├── checkpoints/
  │   ├── iteration_010/
  │   ├── iteration_020/
  │   └── ...
  └── logs/
      └── evolution.log
```

## What to Expect

### Initial Solution (Grid-based)
```python
def construct_packing(n):
    # Simple grid placement
    grid_size = int(np.ceil(np.sqrt(n)))
    ...
```
**Score for n=26**: ~0.36 (36% of target)

### After 20 Iterations
```python
def construct_packing(n):
    # Hexagonal pattern with optimization
    ...
```
**Score for n=26**: ~0.65 (65% of target)

### After 50 Iterations
```python
def construct_packing(n):
    # More sophisticated optimization approach
    # May use iterative refinement or mathematical optimization
    ...
```
**Score for n=26**: ~0.85 (85% of target)

### After 100+ Iterations
```python
def construct_packing(n):
    # Advanced optimization with good initial guess
    # Scales well across all n values
    ...
```
**Score for n=26**: ~0.95-0.99 (95-99% of target)

## Troubleshooting

### "No module named 'gepa'"
```bash
cd /Users/lakshyaaagrawal/Documents/Repos/gepa_dev/gepa_3
pip install -e .
```

### "TimeoutError" during evaluation
Normal! The code is trying complex algorithms. The subprocess will terminate and try again.

### Very low scores after many iterations
Check:
1. API key is set correctly
2. Reflection prompt is being used (check logs)
3. Model has enough context (Claude Sonnet recommended)

### "Process exited with code 1"
The generated code has an error. This is normal during evolution - GEPA learns from failures.

## Key Files

```
examples/evolve_anything/
  ├── 06_circle_packing_multi_n.py     ← Main file to run
  ├── 06_circle_packing_test.py        ← Test file
  ├── CIRCLE_PACKING_README.md         ← Detailed docs
  ├── CIRCLE_PACKING_SUMMARY.md        ← Technical summary
  └── CIRCLE_PACKING_QUICKSTART.md     ← This file

src/gepa/
  └── optimize.py                      ← Enhanced evolve API
```

## Next Steps

1. ✅ **Run tests** to verify everything works
2. 🚀 **Start evolution** with reduced settings for testing
3. 📊 **Monitor progress** - check scores improving over iterations
4. 🎯 **Full run** with all workloads and 100+ iterations
5. 🔬 **Analyze results** - what algorithms did it discover?

## Tips for Best Results

1. **Start small**: Test with [7, 10, 26] first
2. **Use good model**: Claude Sonnet > GPT-4 > GPT-4-mini
3. **Be patient**: Good results take 50+ iterations
4. **Check logs**: Look for error patterns
5. **Iterate**: Run multiple times with different seeds

## Cost Estimation

**Quick test** (20 iterations, 3 workloads):
- ~100 API calls
- ~$2-5 depending on model

**Full run** (100 iterations, 17 workloads):
- ~1000-2000 API calls
- ~$20-50 depending on model

Tips to reduce cost:
- Use GPT-4o-mini instead of Claude Sonnet
- Reduce minibatch_size from 5 to 3
- Use fewer workloads initially
- Stop early if plateau detected

## Support

For issues or questions:
1. Check `CIRCLE_PACKING_README.md` for detailed docs
2. Review test output for validation
3. Check GEPA documentation for API details
4. Examine logs in output directory

---

**Ready to start?** Run the test first, then the evolution! 🎯

