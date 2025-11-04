#!/usr/bin/env python3
"""
Test different rung configurations to find optimal ASHA strategies.
"""

import os
import shutil
import time
from pathlib import Path

os.environ["LITELLM_LOG"] = "ERROR"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Test configurations
RUNG_CONFIGS = [
    {
        "name": "Current (2-rung, large gap)",
        "shards": (0.6, 1.0),
        "description": "60% then 100% - large gap, aggressive pruning",
    },
    {
        "name": "3-rung gradual",
        "shards": (0.2, 0.5, 1.0),
        "description": "20% → 50% → 100% - gradual progression",
    },
    {
        "name": "4-rung small steps",
        "shards": (0.1, 0.3, 0.6, 1.0),
        "description": "Small incremental steps for careful evaluation",
    },
    {
        "name": "5-rung ladder",
        "shards": (0.05, 0.2, 0.5, 0.8, 1.0),
        "description": "Very gradual ladder - maximum early filtering",
    },
    {
        "name": "2-rung smaller start",
        "shards": (0.3, 1.0),
        "description": "30% then 100% - smaller initial evaluation",
    },
    {
        "name": "3-rung tight spacing",
        "shards": (0.4, 0.7, 1.0),
        "description": "Tight spacing for late-stage differentiation",
    },
]


def run_test(config_dict: dict, dataset: list) -> dict:
    """Run optimization with specific rung configuration."""
    # Clear cache
    cache_path = Path(".turbo_gepa/")
    if cache_path.exists():
        shutil.rmtree(cache_path)

    shards = config_dict["shards"]

    config = Config(
        eval_concurrency=32,
        n_islands=1,
        shards=shards,
        max_mutations_per_round=8,
        max_optimization_time_seconds=90,  # 90s per test
        log_level="WARNING",
    )

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="openrouter/openai/gpt-oss-20b:nitro",
        reflection_lm="openrouter/x-ai/grok-4-fast",
        config=config,
        auto_config=False,
    )

    start = time.time()
    result = adapter.optimize(
        seeds=["You are a helpful math assistant. Solve step by step and put final answer after ###."],
        max_rounds=3,
        display_progress=False,
    )
    elapsed = time.time() - start

    # Extract results
    pareto_entries = result.get("pareto_entries", [])
    stats = result.get("evolution_stats", {})
    metrics = adapter._metrics

    # Group candidates by rung
    by_rung = {}
    for entry in pareto_entries:
        rung = entry.result.shard_fraction
        if rung not in by_rung:
            by_rung[rung] = []
        by_rung[rung].append(entry)

    # Calculate metrics
    total_candidates = 1 + stats.get("mutations_generated", 0)
    candidates_at_final_rung = len(by_rung.get(1.0, []))
    promotion_rate = candidates_at_final_rung / total_candidates if total_candidates > 0 else 0

    total_api_time = metrics.eval_time_sum + metrics.mutation_latency_sum
    parallelism = total_api_time / elapsed if elapsed > 0 else 0

    # Best quality
    final_rung_entries = by_rung.get(1.0, [])
    best_quality = 0.0
    if final_rung_entries:
        best_quality = max(e.result.objectives.get("quality", 0) for e in final_rung_entries)

    return {
        "elapsed": elapsed,
        "total_evals": stats.get("total_evaluations", 0),
        "mutations_generated": stats.get("mutations_generated", 0),
        "mutations_promoted": stats.get("mutations_promoted", 0),
        "candidates_at_final": candidates_at_final_rung,
        "promotion_rate": promotion_rate,
        "pareto_size": len(pareto_entries),
        "rungs_reached": len(by_rung),
        "best_quality": best_quality,
        "eval_time": metrics.eval_time_sum,
        "mutation_time": metrics.mutation_latency_sum,
        "parallelism": parallelism,
        "evals_per_rung": dict(metrics.evaluations_by_shard),
        "by_rung": {k: len(v) for k, v in by_rung.items()},
    }


def main():
    # Load dataset
    trainset, _, _ = gepa.examples.aime.init_dataset()
    data = [
        DefaultDataInst(input=ex["input"], answer=ex["answer"], id=f"aime_{i}")
        for i, ex in enumerate(trainset[:8])
    ]

    print("\n" + "="*100)
    print("RUNG STRATEGY COMPARISON")
    print("="*100)
    print(f"\nDataset: {len(data)} AIME problems")
    print("Config: 32 concurrency, 3 rounds, 90s timeout\n")

    results = []

    for i, config_dict in enumerate(RUNG_CONFIGS, 1):
        print(f"\n[{i}/{len(RUNG_CONFIGS)}] Testing: {config_dict['name']}")
        print(f"    Shards: {config_dict['shards']}")
        print(f"    {config_dict['description']}")
        print(f"    Problems per rung: {[int(len(data) * s) for s in config_dict['shards']]}")

        try:
            result = run_test(config_dict, data)
            results.append((config_dict, result))

            print(f"    ✅ Complete in {result['elapsed']:.1f}s")
            print(f"       Evaluations: {result['total_evals']}")
            print(f"       Mutations: {result['mutations_generated']}")
            print(f"       Final rung: {result['candidates_at_final']}/{result['mutations_generated']+1} ({result['promotion_rate']:.1%})")
            print(f"       Best quality: {result['best_quality']:.1%}")
            print(f"       Parallelism: {result['parallelism']:.2f}x")

        except Exception as e:
            print(f"    ❌ Failed: {e}")

    # Print comparison table
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"\n{'Strategy':<25} {'Time':>6} {'Evals':>6} {'Muts':>5} {'Final':>6} {'Rate':>6} {'Quality':>8} {'Par':>5}")
    print("-" * 100)

    for config_dict, result in results:
        name = config_dict['name']
        print(f"{name:<25} {result['elapsed']:>6.1f}s {result['total_evals']:>6} "
              f"{result['mutations_generated']:>5} {result['candidates_at_final']:>6} "
              f"{result['promotion_rate']:>6.1%} {result['best_quality']:>8.1%} {result['parallelism']:>5.2f}x")

    # Find best by different metrics
    print("\n" + "="*100)
    print("WINNERS BY METRIC")
    print("="*100)

    # Best promotion rate (most candidates reaching final rung)
    best_promotion = max(results, key=lambda x: x[1]['promotion_rate'])
    print(f"\n🏆 Best Promotion Rate: {best_promotion[0]['name']}")
    print(f"   {best_promotion[1]['promotion_rate']:.1%} of candidates reached final rung")
    print(f"   Shards: {best_promotion[0]['shards']}")

    # Best quality
    best_quality = max(results, key=lambda x: x[1]['best_quality'])
    print(f"\n🏆 Best Quality: {best_quality[0]['name']}")
    print(f"   {best_quality[1]['best_quality']:.1%} accuracy")
    print(f"   Shards: {best_quality[0]['shards']}")

    # Most efficient (evals per second)
    best_throughput = max(results, key=lambda x: x[1]['total_evals'] / x[1]['elapsed'])
    throughput = best_throughput[1]['total_evals'] / best_throughput[1]['elapsed']
    print(f"\n🏆 Best Throughput: {best_throughput[0]['name']}")
    print(f"   {throughput:.2f} evaluations/sec")
    print(f"   Shards: {best_throughput[0]['shards']}")

    # Best parallelism
    best_parallel = max(results, key=lambda x: x[1]['parallelism'])
    print(f"\n🏆 Best Parallelism: {best_parallel[0]['name']}")
    print(f"   {best_parallel[1]['parallelism']:.2f}x concurrent operations")
    print(f"   Shards: {best_parallel[0]['shards']}")

    # Most mutations generated
    best_exploration = max(results, key=lambda x: x[1]['mutations_generated'])
    print(f"\n🏆 Most Exploration: {best_exploration[0]['name']}")
    print(f"   {best_exploration[1]['mutations_generated']} mutations generated")
    print(f"   Shards: {best_exploration[0]['shards']}")

    # Detailed rung analysis for best promotion rate
    print("\n" + "="*100)
    print(f"DETAILED RUNG ANALYSIS: {best_promotion[0]['name']}")
    print("="*100)

    result = best_promotion[1]
    shards = best_promotion[0]['shards']

    print(f"\nCandidates by rung:")
    for rung in sorted(result['by_rung'].keys()):
        count = result['by_rung'][rung]
        problems = int(len(data) * rung)
        print(f"   Rung {rung:.1%} ({problems} problems): {count} candidates")

    print(f"\nEvaluations by rung:")
    for rung in sorted(result['evals_per_rung'].keys()):
        count = result['evals_per_rung'][rung]
        problems = int(len(data) * rung)
        print(f"   Rung {rung:.1%} ({problems} problems): {count} evaluations")

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    # Calculate variance tolerance for each config
    from turbo_gepa.config import _default_variance_tolerance

    print("\nVariance tolerances by strategy:")
    for config_dict, result in results:
        shards = config_dict['shards']
        tolerance = _default_variance_tolerance(shards)
        print(f"\n{config_dict['name']}:")
        for rung, tol in sorted(tolerance.items()):
            problems = int(len(data) * rung)
            print(f"   Rung {rung:.1%} ({problems} problems): ±{tol:.1%} tolerance")

    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
