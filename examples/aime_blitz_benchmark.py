#!/usr/bin/env python3
"""
AIME Blitz Mode Benchmark - 10× Faster Optimization

This script runs TurboGEPA in BLITZ mode on the full AIME dataset for rapid
exploration and benchmarking.

Models:
- Task LM: gpt-oss-120b:nitro (fast, good quality)
- Reflection LM: grok-4-fast (fast reasoning)

Expected runtime: ~30-60 minutes (vs 5-10 hours in balanced mode)
Expected quality: ~70% of optimal (good for exploration)

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python examples/aime_blitz_benchmark.py

Options:
    python examples/aime_blitz_benchmark.py --limit 50  # Test on 50 examples first
    python examples/aime_blitz_benchmark.py --mode lightning  # Use lightning instead of blitz
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
from turbo_gepa import blitz_config, get_lightning_config
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst


# ============================================================================
# CONFIG
# ============================================================================

TASK_LM = "openrouter/google/gemini-2.0-flash-001"  # Good balance of speed/quality
REFLECTION_LM = "openrouter/google/x-ai/grok-4-fast"  # Fast reasoning model

SEED_PROMPTS = [
    "You are a helpful assistant. You are given a question and you need to answer it. "
    "The answer should be given at the end of your response in exactly the format '### <final answer>'",
    "You are an expert mathematician. Solve the problem step by step. "
    "Provide your final answer at the end in the format: ### <answer>",
    "Break down the problem carefully. Show your reasoning. "
    "End with: ### <your final answer>",
]


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def load_aime_dataset(limit: int | None = None):
    """Load AIME dataset and convert to TurboGEPA format."""
    print("📊 Loading AIME dataset...")

    trainset, valset, testset = gepa.examples.aime.init_dataset()

    # Use validation set for this benchmark
    dataset = valset

    if limit:
        dataset = dataset[:limit]
        print(f"   Limited to {len(dataset)} examples for testing")
    else:
        print(f"   Loaded {len(dataset)} examples (full validation set)")

    # Convert to DefaultDataInst format
    turbo_dataset = []
    for i, example in enumerate(dataset):
        turbo_dataset.append(
            DefaultDataInst(
                input=example["input"],
                answer=example["answer"],  # Already includes "### " prefix
                id=f"aime_{i}",
            )
        )

    return turbo_dataset


def check_api_key():
    """Verify OpenRouter API key is set."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nSet your API key:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        print("\nGet a key at: https://openrouter.ai/keys")
        sys.exit(1)
    print("✓ OpenRouter API key found")


def print_banner(mode: str, dataset_size: int):
    """Print startup banner."""
    print("\n" + "=" * 70)
    print(f"TurboGEPA {mode.upper()} Mode - AIME Benchmark")
    print("=" * 70)
    print(f"\n📝 Configuration:")
    print(f"   Mode: {mode}")
    print(f"   Dataset: AIME validation set ({dataset_size} examples)")
    print(f"   Task LM: {TASK_LM}")
    print(f"   Reflection LM: {REFLECTION_LM}")

    if mode == "blitz":
        print(f"\n⚡ Blitz Mode: 10× faster, ~70% quality")
        print(f"   Expected runtime: ~30-60 minutes")
        print(f"   Good for: rapid exploration, initial testing")
    elif mode == "lightning":
        print(f"\n⚡ Lightning Mode: 5× faster, ~85% quality")
        print(f"   Expected runtime: ~1-2 hours")
        print(f"   Good for: iterative development, debugging")
    elif mode == "sprint":
        print(f"\n⚡ Sprint Mode: 3× faster, ~90% quality")
        print(f"   Expected runtime: ~2-3 hours")
        print(f"   Good for: fast production runs")

    print("\n" + "=" * 70 + "\n")


def run_optimization(dataset, mode: str, max_rounds: int | None = None):
    """Run TurboGEPA optimization with lightning mode."""

    # Get lightning config
    config = get_lightning_config(mode, len(dataset))

    print(f"🔧 Configuration Details:")
    print(f"   Shards: {config.shards}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max mutations/round: {config.max_mutations_per_round}")
    print(f"   Islands: {config.n_islands}")
    print(
        f"   Cohort quantile: {config.cohort_quantile} (top {int((1-config.cohort_quantile)*100)}%)"
    )
    print(f"   Eval concurrency: {config.eval_concurrency}")
    print(f"   Reflection batch size: {config.reflection_batch_size}")
    print()

    # Create adapter
    print("🚀 Initializing DefaultAdapter...")
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        auto_config=False,  # We're using manual lightning config
        cache_dir=f".turbo_gepa/cache_aime_{mode}",
    )

    # Apply lightning config
    adapter.config = config

    print(f"✓ Adapter initialized with {mode} mode\n")

    # Run optimization
    print(f"🏃 Starting optimization...")
    print(f"   Seeds: {len(SEED_PROMPTS)} initial prompts")
    if max_rounds:
        print(f"   Max rounds: {max_rounds}")
    else:
        print(f"   Max rounds: Unlimited (will run until converged)")
    print()

    start_time = time.time()

    try:
        result = adapter.optimize(
            seeds=SEED_PROMPTS,
            max_rounds=max_rounds or 999,
            max_evaluations=None,  # Let ASHA control
            enable_auto_stop=True if not max_rounds else False,
            display_progress=True,
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

    if "pareto" in result and result["pareto"]:
        print(f"\n📈 Results:")
        print(f"   Pareto frontier: {len(result['pareto'])} candidates")

        # Show top candidates
        pareto = sorted(
            result["pareto"], key=lambda c: c.meta.get("quality", 0), reverse=True
        )

        print(f"\n   Top 3 candidates:")
        for i, candidate in enumerate(pareto[:3], 1):
            quality = candidate.meta.get("quality", 0)
            tokens = len(candidate.text.split())
            temp = candidate.meta.get("temperature", "N/A")

            print(f"\n   {i}. Quality: {quality:.3f}, Tokens: {tokens}, Temp: {temp}")
            print(f"      Prompt preview: {candidate.text[:100]}...")

        # Best candidate
        best = pareto[0]
        print(f"\n🏆 Best candidate:")
        print(f"   Quality: {best.meta.get('quality', 0):.3f}")
        print(f"   Tokens: {len(best.text.split())}")
        print(f"\n   Full prompt:")
        print("   " + "-" * 66)
        for line in best.text.split("\n"):
            print(f"   {line}")
        print("   " + "-" * 66)

    # Save results
    output_dir = Path(f".turbo_gepa/results_aime_{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    output_data = {
        "mode": mode,
        "dataset_size": len(dataset),
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60,
        "config": {
            "shards": list(config.shards),
            "batch_size": config.batch_size,
            "max_mutations_per_round": config.max_mutations_per_round,
            "n_islands": config.n_islands,
            "cohort_quantile": config.cohort_quantile,
            "eval_concurrency": config.eval_concurrency,
        },
        "results": {
            "pareto_size": len(result.get("pareto", [])),
            "best_quality": (
                best.meta.get("quality", 0)
                if "pareto" in result and result["pareto"]
                else 0
            ),
            "best_prompt": best.text if "pareto" in result and result["pareto"] else "",
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run TurboGEPA in blitz mode on AIME dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset size for testing (default: None = full dataset)",
    )
    parser.add_argument(
        "--mode",
        choices=["blitz", "lightning", "sprint"],
        default="blitz",
        help="Lightning mode to use (default: blitz)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum optimization rounds (default: None = auto-stop)",
    )

    args = parser.parse_args()

    # Check API key
    check_api_key()

    # Load dataset
    dataset = load_aime_dataset(limit=args.limit)

    # Print banner
    print_banner(args.mode, len(dataset))

    # Run optimization
    result = run_optimization(dataset, args.mode, args.max_rounds)

    print("\n✨ Done! Check results above or in .turbo_gepa/results_aime_{mode}/\n")


if __name__ == "__main__":
    main()
