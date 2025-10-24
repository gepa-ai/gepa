#!/usr/bin/env python3
"""
Quick Benchmark - Show TurboGEPA improving 25% on AIME

Lightweight test to verify:
1. Optimization loop works
2. Quality improves over time
3. Real LLM calls succeed
4. Mutations and reflection work

Target: Improve from baseline to 25% accuracy on 3 AIME problems
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
os.environ["LITELLM_LOG"] = "ERROR"
warnings.filterwarnings("ignore")

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Model configuration
TASK_LM = "openrouter/google/gemini-2.0-flash-001"
REFLECTION_LM = "openrouter/x-ai/grok-4-fast"

# Seed prompt
SEED_PROMPT = (
    "You are a helpful assistant. You are given a question and you need to answer it. "
    "The answer should be given at the end of your response in exactly the format '### <final answer>'"
)


def check_api_key():
    """Verify OpenRouter API key is set."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nSet your API key:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        sys.exit(1)
    print("✓ OpenRouter API key found")


def load_dataset():
    """Load 3 AIME problems for quick testing."""
    print("\n📊 Loading AIME dataset (3 problems)...")
    trainset, valset, _ = gepa.examples.aime.init_dataset()

    # Use just 3 problems for speed
    train = trainset[:3]

    # Convert to TurboGEPA format
    dataset = [
        DefaultDataInst(
            input=ex["input"],
            answer=ex["answer"],
            id=f"aime_{i}",
            additional_context=ex.get("additional_context"),
        )
        for i, ex in enumerate(train)
    ]

    print(f"✓ Loaded {len(dataset)} AIME problems")
    return dataset


def run_benchmark():
    """Run quick benchmark targeting 25% improvement."""

    print("\n" + "="*70)
    print("  QUICK TURBOGEPA BENCHMARK")
    print("="*70)
    print(f"\n📝 Configuration:")
    print(f"   Dataset: 3 AIME problems")
    print(f"   Task LM: {TASK_LM}")
    print(f"   Reflection LM: {REFLECTION_LM}")
    print(f"   Target: 33% accuracy (1 of 3 problems correct)")
    print(f"   Mode: Fast optimization with minimal resources")
    print("\n" + "="*70 + "\n")

    # Load data
    dataset = load_dataset()

    # Create ultra-lightweight config for speed
    config = Config(
        shards=(1.0,),  # Single shard - no ASHA, full eval only
        cohort_quantile=0.6,
        batch_size=4,  # Very small batch
        max_mutations_per_round=4,  # Few mutations
        eval_concurrency=6,  # Low concurrency
        max_total_inflight=6,
        n_islands=1,  # Single island
        reflection_batch_size=2,  # Small reflection batch
        target_quality=0.34,  # Stop at 33% accuracy (1 of 3 correct)
    )

    print(f"🔧 Config: {config.batch_size} batch size, {config.eval_concurrency}x concurrency")
    print(f"   Target quality: {config.target_quality:.0%}\n")

    # Create adapter
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        auto_config=False,
        cache_dir=".turbo_gepa/cache_quick_benchmark",
    )
    adapter.config = config

    print("🚀 Starting optimization...\n")

    start_time = time.time()

    try:
        result = adapter.optimize(
            seeds=[SEED_PROMPT],
            max_rounds=5,  # Max 5 rounds
            max_evaluations=50,  # Stop after 50 evals
            enable_auto_stop=True,  # Stop at target quality
            display_progress=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        elapsed = time.time() - start_time
        print(f"   Ran for {elapsed:.1f} seconds before interruption")
        return

    elapsed = time.time() - start_time

    # Get best result
    if "pareto" in result and result["pareto"]:
        best = max(result["pareto"], key=lambda c: c.meta.get("quality", 0))
        best_quality = best.meta.get("quality", 0)
        best_prompt = best.text
    else:
        best_quality = 0.0
        best_prompt = None

    # Print results
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS")
    print("="*70 + "\n")

    print(f"📊 Performance:")
    print(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
    print(f"   Best quality: {best_quality:.1%}")
    print(f"   Target: 33%")

    if best_quality >= 0.33:
        print(f"\n✅ SUCCESS! Reached target quality of 33%")
        print(f"   Achieved: {best_quality:.1%} in {elapsed:.1f} seconds")
    else:
        print(f"\n⚠️  Did not reach 25% target")
        print(f"   Best achieved: {best_quality:.1%}")
        print(f"   Consider running longer or adjusting config")

    if best_prompt:
        print(f"\n🏆 Best prompt (quality: {best_quality:.1%}):")
        print("   " + "="*66)
        for line in best_prompt.split("\n")[:5]:  # Show first 5 lines
            print(f"   {line}")
        if len(best_prompt.split("\n")) > 5:
            print("   ...")
        print("   " + "="*66)

    print()


def main():
    print("\n🏁 Quick TurboGEPA Benchmark")
    print("   Goal: Improve to 33% accuracy on 3 AIME problems (ultra-fast test)\n")

    check_api_key()

    # Clear cache for fresh run
    import shutil
    cache_dir = Path(".turbo_gepa/cache_quick_benchmark")
    if cache_dir.exists():
        print(f"🗑️  Clearing cache...")
        shutil.rmtree(cache_dir)
        print("✓ Cache cleared\n")

    run_benchmark()

    print("✨ Benchmark complete!\n")


if __name__ == "__main__":
    main()
