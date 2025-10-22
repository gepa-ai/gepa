#!/usr/bin/env python3
"""
Demo: Lightning Mode Speed Optimization

This script demonstrates TurboGEPA's lightning modes for 3-10× faster optimization.

Usage:
    python examples/demo_lightning_mode.py --mode lightning
    python examples/demo_lightning_mode.py --mode sprint
    python examples/demo_lightning_mode.py --mode blitz
"""

import argparse
import time

from turbo_gepa import blitz_config, get_lightning_config, lightning_config, sprint_config


def demo_lightning_modes():
    """Compare different lightning mode configurations."""
    dataset_size = 500

    print("=" * 70)
    print("TurboGEPA Lightning Modes Comparison")
    print("=" * 70)
    print(f"\nDataset size: {dataset_size} examples\n")

    modes = {
        "blitz": (blitz_config, "10× faster, ~70% quality"),
        "lightning": (lightning_config, "5× faster, ~85% quality"),
        "sprint": (sprint_config, "3× faster, ~90% quality"),
    }

    print(f"{'Mode':<12} {'Speedup':<10} {'Shards':<25} {'Batch':<8} {'Islands':<10} {'Pruning':<12}")
    print("-" * 70)

    for mode_name, (config_func, description) in modes.items():
        config = config_func(dataset_size)
        speedup = "10×" if mode_name == "blitz" else "5×" if mode_name == "lightning" else "3×"
        pruning = f"Top {int((1-config.cohort_quantile)*100)}%"

        print(
            f"{mode_name:<12} {speedup:<10} {str(config.shards):<25} "
            f"{config.batch_size:<8} {config.n_islands:<10} {pruning:<12}"
        )

    print("\n" + "=" * 70)
    print("Using get_lightning_config() helper:")
    print("=" * 70 + "\n")

    for mode in ["blitz", "lightning", "sprint", "balanced"]:
        config = get_lightning_config(mode, dataset_size)
        description = modes.get(mode, (None, "Default balanced mode"))[1]
        print(f"  {mode:12} -> {description}")
        print(f"               Shards: {config.shards}")
        print(f"               Batch: {config.batch_size}, Islands: {config.n_islands}")
        print()


def run_with_mode(mode: str, dataset_size: int = 100):
    """
    Example: Run optimization with specific lightning mode.

    This is a minimal example showing config only - replace with actual
    optimization logic from your use case.
    """
    print(f"\n{'='*70}")
    print(f"Running with {mode.upper()} mode")
    print(f"{'='*70}\n")

    # Get config
    config = get_lightning_config(mode, dataset_size)

    print(f"Config parameters:")
    print(f"  Shards: {config.shards}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max mutations/round: {config.max_mutations_per_round}")
    print(f"  Islands: {config.n_islands}")
    print(f"  Cohort quantile: {config.cohort_quantile} (top {int((1-config.cohort_quantile)*100)}%)")
    print(f"  Eval concurrency: {config.eval_concurrency}")
    print(f"  Reflection batch size: {config.reflection_batch_size}")
    print()

    # Example: Estimate speedup
    baseline_evals = dataset_size * 8  # 8 candidates × 100% shard
    lightning_evals = estimate_evaluations(config, dataset_size)
    speedup = baseline_evals / lightning_evals

    print(f"Estimated evaluations:")
    print(f"  Baseline (balanced): ~{baseline_evals} evals")
    print(f"  Lightning ({mode}): ~{lightning_evals} evals")
    print(f"  Speedup: ~{speedup:.1f}×")
    print()

    print("✓ Configuration ready - integrate with your optimization code")
    print()


def estimate_evaluations(config, dataset_size: int) -> int:
    """Rough estimate of total evaluations with ASHA pruning."""
    # Assume 60% pruning at first shard, 40% at second (if exists)
    batch = config.batch_size
    shards = config.shards
    prune_rate = config.cohort_quantile

    total = 0
    candidates = batch

    for shard_frac in shards:
        evals = candidates * int(dataset_size * shard_frac)
        total += evals
        candidates = int(candidates * (1 - prune_rate))  # Prune

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Demo TurboGEPA lightning modes for speed optimization"
    )
    parser.add_argument(
        "--mode",
        choices=["blitz", "lightning", "sprint", "balanced", "compare"],
        default="compare",
        help="Lightning mode to demonstrate (default: compare all)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=100,
        help="Dataset size for configuration (default: 100)",
    )

    args = parser.parse_args()

    if args.mode == "compare":
        demo_lightning_modes()
    else:
        run_with_mode(args.mode, args.dataset_size)


if __name__ == "__main__":
    main()
