#!/usr/bin/env python3
"""
AIME Original GEPA Benchmark - Baseline Comparison

This script runs the original GEPA on the AIME dataset using the same models
as the TurboGEPA blitz benchmark for direct comparison.

Models:
- Task LM: gpt-oss-120b:nitro (fast, good quality)
- Reflection LM: grok-4-fast (fast reasoning)

Expected runtime: ~2-4 hours (standard GEPA without optimizations)

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python examples/aime_og_gepa_benchmark.py

Options:
    python examples/aime_og_gepa_benchmark.py --limit 50  # Test on 50 examples first
    python examples/aime_og_gepa_benchmark.py --max-calls 300  # Increase budget
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gepa


# ============================================================================
# CONFIG
# ============================================================================

TASK_LM = "openrouter/openai/gpt-oss-120b:nitro"  # Same as blitz benchmark
REFLECTION_LM = "openrouter/x-ai/grok-4-fast"  # Faster reflection model

SEED_PROMPT = {
    "system_prompt": (
        "You are a helpful assistant. You are given a question and you need to answer it. "
        "The answer should be given at the end of your response in exactly the format '### <final answer>'"
    )
}


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def check_api_key():
    """Verify OpenRouter API key is set."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nSet your API key:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        print("\nGet a key at: https://openrouter.ai/keys")
        sys.exit(1)
    print("✓ OpenRouter API key found")


def print_banner(max_calls: int, dataset_size: int):
    """Print startup banner."""
    print("\n" + "=" * 70)
    print("Original GEPA - AIME Benchmark")
    print("=" * 70)
    print(f"\n📝 Configuration:")
    print(f"   Mode: Original GEPA (baseline)")
    print(f"   Dataset: AIME validation set ({dataset_size} examples)")
    print(f"   Task LM: {TASK_LM}")
    print(f"   Reflection LM: {REFLECTION_LM}")
    print(f"   Max metric calls: {max_calls}")
    print(f"\n⏱️  Expected runtime: ~2-4 hours")
    print(f"   Quality: 100% (baseline for comparison)")
    print("\n" + "=" * 70 + "\n")


def run_optimization(trainset, valset, max_calls: int):
    """Run original GEPA optimization."""

    print(f"🚀 Starting GEPA optimization...")
    print(f"   Seed prompt: {SEED_PROMPT['system_prompt'][:80]}...")
    print(f"   Training set: {len(trainset)} examples")
    print(f"   Validation set: {len(valset)} examples")
    print()

    start_time = time.time()

    try:
        gepa_result = gepa.optimize(
            seed_candidate=SEED_PROMPT,
            trainset=trainset,
            valset=valset,
            task_lm=TASK_LM,
            max_metric_calls=max_calls,
            reflection_lm=REFLECTION_LM,
            display_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        elapsed = time.time() - start_time
        print(f"   Ran for {elapsed/60:.1f} minutes before interruption")
        sys.exit(0)

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("✅ Optimization Complete!")
    print("=" * 70)
    print(f"\n⏱️  Runtime: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")

    print(f"\n🏆 Best candidate:")
    print(f"   Full prompt:")
    print("   " + "-" * 66)
    for line in gepa_result.best_candidate["system_prompt"].split("\n"):
        print(f"   {line}")
    print("   " + "-" * 66)

    # Show all candidates if available
    if hasattr(gepa_result, "candidates") and gepa_result.candidates:
        print(f"\n📈 Results:")
        print(f"   Total candidates evaluated: {len(gepa_result.candidates)}")

    # Save results
    output_dir = Path(".turbo_gepa/results_aime_og_gepa")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    output_data = {
        "mode": "original_gepa",
        "dataset_size_train": len(trainset),
        "dataset_size_val": len(valset),
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
        "max_metric_calls": max_calls,
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60,
        "results": {
            "best_prompt": gepa_result.best_candidate["system_prompt"],
            "total_candidates": (
                len(gepa_result.candidates)
                if hasattr(gepa_result, "candidates")
                else "unknown"
            ),
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print()

    return gepa_result


def main():
    parser = argparse.ArgumentParser(
        description="Run original GEPA on AIME dataset for baseline comparison"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset size for testing (default: None = full dataset)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=40,
        help="Maximum metric calls budget (default: 40 for quick runs)",
    )

    args = parser.parse_args()

    # Check API key
    check_api_key()

    # Load dataset
    print("📊 Loading AIME dataset...")
    trainset, valset, _ = gepa.examples.aime.init_dataset()

    if args.limit:
        trainset = trainset[: args.limit]
        valset = valset[: args.limit // 2] if args.limit > 1 else valset[:1]
        print(
            f"   Limited to {len(trainset)} training, {len(valset)} validation examples"
        )
    else:
        print(
            f"   Loaded {len(trainset)} training, {len(valset)} validation examples (full dataset)"
        )

    # Print banner
    print_banner(args.max_calls, len(valset))

    # Run optimization
    result = run_optimization(trainset, valset, args.max_calls)

    print("\n✨ Done! Check results above or in .turbo_gepa/results_aime_og_gepa/\n")
    print(
        "💡 Compare with TurboGEPA blitz results in .turbo_gepa/results_aime_blitz/\n"
    )


if __name__ == "__main__":
    main()
