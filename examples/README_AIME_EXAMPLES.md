# AIME Examples - Quick Reference

This directory contains AIME examples demonstrating TurboGEPA features at different complexity levels.

## Quick Start - Choose Your Example

### 1. 🔬 Minimal Smoke Test (< 1 minute)
**File:** `minimal_aime_test.py`

The absolute fastest way to verify TurboGEPA is working:
- Uses 2 AIME problems
- Runs 3 optimization rounds
- Should complete in under 1 minute

```bash
python examples/minimal_aime_test.py
```

**Use this when:**
- You want to quickly verify your installation
- You're testing after making code changes
- You want to check LLM connectivity

---

### 2. 📝 Simple Demo (3-5 minutes)
**File:** `simple_aime_demo.py`

A more thorough test with detailed output:
- Uses 5 AIME problems
- Runs 10 optimization rounds
- Shows detailed metrics and the evolved prompt

```bash
python examples/simple_aime_demo.py
```

**Use this when:**
- You want to see TurboGEPA actually improve prompts
- You're learning how the system works
- You want to verify the full pipeline end-to-end

---

### 3. 🏆 Full Benchmark (10-30 minutes)
**File:** `aime_benchmark.py`

Complete performance benchmark comparing GEPA vs TurboGEPA:
- Configurable dataset size (default: 10 problems)
- Runs original GEPA and/or TurboGEPA
- Full performance metrics and comparison

```bash
# Run just TurboGEPA (faster)
python examples/aime_benchmark.py --run turbo

# Run just original GEPA
python examples/aime_benchmark.py --run gepa

# Run both for comparison
python examples/aime_benchmark.py --run both
```

**Use this when:**
- You want to measure actual performance improvements
- You're comparing different configurations
- You need reproducible benchmark results

---

### 4. ⚡ Dynamic Sharding Demo (5-10 minutes)
**File:** `demo_dynamic_sharding.py`

Demonstrates ASHA (Asynchronous Successive Halving) with 3-rung sharding:
- Uses 20 AIME problems
- Evaluates candidates progressively (20% → 50% → 100%)
- Shows rung activity and promotion between shards
- Calculates efficiency savings from early pruning

```bash
python examples/demo_dynamic_sharding.py
```

**Use this when:**
- You want to see ASHA pruning in action
- Learning how dynamic sharding saves compute
- Understanding rung progression and promotions

---

### 5. 🌴 Multi-Island Demo (5-10 minutes)
**File:** `demo_multi_island.py`

Demonstrates parallel island-based optimization:
- Uses 2-4 islands running in parallel processes
- Shows periodic migration between islands
- Ring topology for candidate exchange
- Demonstrates speedup from parallelism

```bash
python examples/demo_multi_island.py
```

**Use this when:**
- You want to see multi-process parallelism
- Understanding island migration patterns
- Measuring performance gains from parallelism

---

## Configuration

All examples use these models by default:
```python
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"
```

You can edit the examples to use different models if needed.

## Requirements

Make sure you have:
1. Installed the package: `uv sync` or `pip install -e .`
2. Set your API key: `export OPENROUTER_API_KEY=your-key-here`

## Expected Results

### Minimal Test
- Quality: 0-50% (only 2 problems, very noisy)
- Pareto size: 1-3 candidates
- Mutations: 4-8 generated
- **Success criterion:** No errors, some mutations generated

### Simple Demo
- Quality: 20-60% (5 problems)
- Pareto size: 3-8 candidates
- Mutations: 10-20 generated
- **Success criterion:** Quality > 0%, best prompt differs from seed

### Full Benchmark
- Quality: 40-70% (10+ problems)
- Pareto size: 5-15 candidates
- Speedup: 2-3x faster than original GEPA
- **Success criterion:** Reaches target quality, shows performance gains

## Troubleshooting

### Quality is 0%
- Check `OPENROUTER_API_KEY` is set correctly
- Verify models are available on OpenRouter
- Check `.turbo_gepa/logs/` for error messages

### No mutations generated
- Check LLM connectivity
- Verify reflection_lm is accessible
- Try reducing eval_concurrency

### Connection errors
- Check your internet connection
- Verify OpenRouter API is accessible
- Try increasing timeout settings

### Out of memory
- Reduce `eval_concurrency` in the config
- Use fewer problems in the dataset
- Run minimal test instead of full benchmark

## Next Steps

After verifying everything works:
1. Try modifying the seed prompt
2. Experiment with different model combinations
3. Adjust config parameters (concurrency, mutations, etc.)
4. Create your own adapter for different tasks

For more details, see the main documentation in `CLAUDE.md`.
