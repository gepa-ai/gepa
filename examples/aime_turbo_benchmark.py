#!/usr/bin/env python3
"""
AIME TurboGEPA Speed Benchmark - Race to Target Score

Quick test mode (default):
    - 10 examples, target=0.7, 1 island
    - Fast validation that algorithms are working
    - Takes ~30 seconds

Full benchmark mode:
    - 90 examples, target=0.8, 4 islands
    - Compare speed vs OG GEPA (8 minutes to 0.8)
    - Use --full flag

Models:
- Task LM: gpt-oss-120b:nitro (fast, good quality)
- Reflection LM: grok-4-fast (fast reasoning)

Usage:
    # Quick test (default)
    export OPENROUTER_API_KEY=your_key_here
    python examples/aime_turbo_benchmark.py

    # Full benchmark
    python examples/aime_turbo_benchmark.py --full

    # Custom settings
    python examples/aime_turbo_benchmark.py --limit 20 --target-score 0.75 --islands 2
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress LiteLLM warnings and info messages
os.environ["LITELLM_LOG"] = "ERROR"  # Only show errors, not warnings/info
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*litellm.*")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Suppress litellm logging after import
try:
    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    # Disable litellm's internal logger
    import logging

    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
except ImportError:
    pass


# ============================================================================
# CONFIG
# ============================================================================

# Model configuration
TASK_LM = "openrouter/google/gemini-2.0-flash-001"  # Fast and capable
REFLECTION_LM = "openrouter/x-ai/grok-4-fast"  # Fast reflection model

# Use the same seed prompt as OG benchmark
SEED_PROMPT = (
    "You are a helpful assistant. You are given a question and you need to answer it. "
    "The answer should be given at the end of your response in exactly the format '### <final answer>'"
)


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def load_aime_dataset(limit: int | None = None):
    """Load AIME dataset and convert to TurboGEPA format."""
    print("📊 Loading AIME dataset...")

    trainset, valset, testset = gepa.examples.aime.init_dataset()

    if limit:
        trainset = trainset[:limit]
        valset = valset[: limit // 2] if limit > 1 else valset[:1]
        print(
            f"   Limited to {len(trainset)} training, {len(valset)} validation examples"
        )
    else:
        print(f"   Loaded {len(trainset)} training, {len(valset)} validation examples")

    # Convert to DefaultDataInst format
    turbo_trainset = []
    for i, example in enumerate(trainset):
        turbo_trainset.append(
            DefaultDataInst(
                input=example["input"],
                answer=example["answer"],
                id=f"aime_train_{i}",
                additional_context=example.get("additional_context"),
            )
        )

    turbo_valset = []
    for i, example in enumerate(valset):
        turbo_valset.append(
            DefaultDataInst(
                input=example["input"],
                answer=example["answer"],
                id=f"aime_val_{i}",
                additional_context=example.get("additional_context"),
            )
        )

    return turbo_trainset, turbo_valset


def check_api_key():
    """Verify OpenRouter API key is set."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nSet your API key:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        print("\nGet a key at: https://openrouter.ai/keys")
        sys.exit(1)
    print("✓ OpenRouter API key found")


def print_banner(
    target_score: float, trainset_size: int, valset_size: int, n_islands: int
):
    """Print startup banner."""
    total = trainset_size + valset_size

    print("\n" + "=" * 70)
    if total <= 20:
        print("TurboGEPA - QUICK TEST")
    else:
        print("TurboGEPA - SPEED BENCHMARK")
    print("=" * 70)
    print(f"\n📝 Configuration:")
    print(
        f"   Dataset: AIME ({trainset_size} train + {valset_size} val = {total} total)"
    )
    print(f"   Task LM: {TASK_LM}")
    print(f"   Reflection LM: {REFLECTION_LM}")
    print(f"   Islands: {n_islands}")
    if n_islands == 1:
        print(f"   💡 Single island mode - prompts will display in real-time!")
    print(f"   Target quality: {target_score}")

    if total <= 20:
        print(f"\n🧪 Quick test mode - verifying algorithms work")
    else:
        print(f"\n🎯 Goal: Reach {target_score} quality ASAP")
        print(f"   OG GEPA baseline: 8.0 minutes to 0.8")

    print("\n" + "=" * 70 + "\n")


def create_speed_config(
    n_islands: int, target_score: float, dataset_size: int
) -> Config:
    """
    Create a TurboGEPA config optimized for speed.

    Key insight: In streaming mode, batch_size should match eval_concurrency
    to fully utilize parallel evaluation. All candidates in a batch are
    evaluated concurrently, so larger batches = more throughput.

    Scales settings based on dataset size.
    """
    # Scale config based on dataset size
    if dataset_size <= 20:
        concurrency = 24
        batch_size = concurrency
        mutations = max(12, batch_size // 2)
        reflection_batch = 3
        shards = (0.3, 1.0)
    else:
        concurrency = 48
        batch_size = concurrency
        mutations = max(24, batch_size // 2)
        reflection_batch = 4
        shards = (0.3, 1.0)

    print(f"🔧 Configuration:")
    print(f"   Islands: {n_islands}")
    print(f"   Batch size: {batch_size} candidates/batch")
    print(f"   Concurrency: {concurrency} evals/batch (fully utilized!)")
    print(f"   Mutations/round: {mutations}")
    print(f"   ASHA shards: {list(shards)}")
    print(f"   Target quality: {target_score}")
    print()

    config = Config(
        shards=shards,
        cohort_quantile=0.6,
        batch_size=batch_size,
        max_mutations_per_round=mutations,
        eval_concurrency=concurrency,
        max_total_inflight=concurrency,
        n_islands=n_islands,
        migration_period=2 if n_islands > 1 else 999,
        migration_k=min(3, n_islands),
        reflection_batch_size=reflection_batch,
        target_quality=target_score,
    )

    return config


def run_optimization(trainset, valset, target_score: float, n_islands: int):
    """Run TurboGEPA optimization and track time to target score."""

    # Create speed-optimized config with target quality
    full_dataset = trainset + valset
    config = create_speed_config(n_islands, target_score, len(full_dataset))

    # Create adapter with full dataset
    print(f"🚀 Initializing DefaultAdapter...")
    print(
        f"   Dataset size: {len(full_dataset)} examples ({len(trainset)} train + {len(valset)} val)"
    )
    print(f"   Cache dir: .turbo_gepa/cache_aime_speed_benchmark")
    adapter = DefaultAdapter(
        dataset=full_dataset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        auto_config=False,
        cache_dir=".turbo_gepa/cache_aime_speed_benchmark",
    )

    # Apply speed config
    adapter.config = config

    print(f"✓ Adapter initialized\n")

    print(f"🏁 Starting optimization (will stop at {target_score} quality)...\n")

    start_time = time.time()

    try:
        # Just run optimize() normally - let it do its thing!
        result = adapter.optimize(
            seeds=[SEED_PROMPT],
            max_rounds=100,  # Safety limit
            max_evaluations=None,
            enable_auto_stop=False,  # Let it run
            display_progress=True,
            optimize_temperature_after_convergence=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        elapsed = time.time() - start_time
        print(f"   Ran for {elapsed/60:.1f} minutes before interruption")
        sys.exit(0)

    elapsed = time.time() - start_time

    # Check if we hit target
    if "pareto" in result and result["pareto"]:
        best_candidate = max(result["pareto"], key=lambda c: c.meta.get("quality", 0))
        best_quality = best_candidate.meta.get("quality", 0)
        best_temperature = best_candidate.meta.get("temperature")
        best_prompt = best_candidate.text
    else:
        best_quality = 0.0
        best_temperature = None
        best_prompt = None

    # Print final summary
    print("\n" + "=" * 70)
    print("✅ Optimization Complete!")
    print("=" * 70)

    print(f"\n📊 Final Results:")
    print(f"   Best quality achieved: {best_quality:.4f}")
    print(f"   Target: {target_score}")
    print(f"   Runtime: {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")
    if best_temperature is not None:
        print(f"   Best temperature: {best_temperature:.2f}")

    if best_quality >= target_score:
        print(f"\n🎉 TARGET REACHED!")
        print(f"   ✓ Achieved {best_quality:.4f} (target: {target_score})")
        print(f"   ✓ Time: {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")
        if best_temperature is not None:
            print(f"   ✓ Temperature: {best_temperature:.2f}")
        print()
        print(f"   ⚡ SPEED COMPARISON:")
        print(f"      OG GEPA:    8.00 minutes to 0.8")
        print(f"      TurboGEPA:  {elapsed/60:.2f} minutes to {best_quality:.2f}")

        if elapsed < 480:
            speedup = 480 / elapsed
            time_saved = 480 - elapsed
            print(f"\n   🚀 SPEEDUP: {speedup:.2f}× FASTER!")
            print(
                f"   💰 TIME SAVED: {time_saved/60:.2f} minutes ({time_saved:.0f} seconds)"
            )
            print(f"   📈 EFFICIENCY GAIN: {(1 - elapsed/480)*100:.1f}%")

        if best_prompt:
            print(f"\n🏆 Winning prompt (quality: {best_quality:.4f}):")
            print("   " + "=" * 66)
            for line in best_prompt.split("\n"):
                print(f"   {line}")
            print("   " + "=" * 66)
    else:
        print(f"\n⚠️  Did not reach target {target_score}")
        print(f"   Best achieved: {best_quality:.4f}")
        print(f"   Consider running longer or adjusting config")

    # Save results
    output_dir = Path(".turbo_gepa/results_aime_speed_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    output_data = {
        "mode": "turbo_gepa_speed_benchmark",
        "goal": f"Race to {target_score} quality as fast as possible",
        "dataset_size_train": len(trainset),
        "dataset_size_val": len(valset),
        "dataset_size_total": len(full_dataset),
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
        "target_score": target_score,
        "og_gepa_baseline_minutes": 8.0,
        "og_gepa_baseline_seconds": 480,
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60,
        "target_reached": best_quality >= target_score,
        "best_quality": best_quality,
        "best_temperature": best_temperature,
        "speedup_vs_og_gepa": (
            480 / elapsed if best_quality >= target_score and elapsed > 0 else None
        ),
        "time_saved_seconds": (480 - elapsed if best_quality >= target_score else None),
        "config": {
            "shards": list(config.shards),
            "batch_size": config.batch_size,
            "max_mutations_per_round": config.max_mutations_per_round,
            "n_islands": config.n_islands,
            "eval_concurrency_per_island": config.eval_concurrency,
            "total_concurrent_evals": config.eval_concurrency * config.n_islands,
            "cohort_quantile": config.cohort_quantile,
            "reflection_batch_size": config.reflection_batch_size,
            "migration_period": config.migration_period,
            "migration_k": config.migration_k,
        },
        "results": {
            "best_prompt": best_prompt or "",
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print()

    return best_quality, best_temperature, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run TurboGEPA speed benchmark - race to target score"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,  # Small default for quick testing
        help="Limit dataset size for testing (default: 10 for quick test)",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=0.8,  # Lower target for quick success
        help="Target quality score (default: 0.7 for quick test)",
    )
    parser.add_argument(
        "--islands",
        type=int,
        default=1,  # Single island for viewing prompts in real-time
        help="Number of islands (parallel processes) (default: 1)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark (90 examples, target=0.8, 4 islands)",
    )

    args = parser.parse_args()

    # Override defaults for full benchmark mode
    if args.full:
        args.limit = None  # Full dataset
        args.target_score = 0.8
        args.islands = 4
        print("🚀 Running FULL benchmark mode (90 examples, target=0.8, 4 islands)\n")
    else:
        print(
            f"🧪 Running QUICK test mode ({args.limit} examples, target={args.target_score}, {args.islands} island)\n"
        )
        print("   Use --full for complete benchmark\n")

    # Check API key
    check_api_key()

    # Wipe cache for fresh benchmark
    import shutil

    cache_dir = Path(".turbo_gepa/cache_aime_speed_benchmark")
    if cache_dir.exists():
        print(f"🗑️  Clearing cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("✓ Cache cleared\n")

    # Load dataset
    trainset, valset = load_aime_dataset(limit=args.limit)

    # Print banner
    print_banner(args.target_score, len(trainset), len(valset), args.islands)

    # Run speed benchmark
    best_quality, best_temperature, elapsed = run_optimization(
        trainset, valset, args.target_score, args.islands
    )

    print("\n✨ Speed benchmark complete!")
    print("\n📊 FINAL SUMMARY:")
    print(f"   Target quality: {args.target_score}")
    print(f"   Achieved quality: {best_quality:.4f}")
    print(f"   Time: {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")
    if best_temperature is not None:
        print(f"   Best temperature: {best_temperature:.2f}")
    if best_quality >= args.target_score:
        speedup = 480 / elapsed if elapsed > 0 else 0
        print(f"   Speedup vs OG GEPA: {speedup:.2f}×")
        print(f"   OG GEPA baseline: 8.00 minutes to 0.8")
    print(f"\n💾 Results: .turbo_gepa/results_aime_speed_benchmark/\n")


if __name__ == "__main__":
    main()
