#!/usr/bin/env python3
"""
Systematic Test Suite - Isolate Variables

This test suite incrementally tests different configurations to identify
what causes issues. Each test isolates one variable at a time.

Tests:
1. Dataset size: 2, 5, 10, 20 problems (1 island, 1 shard)
2. Islands: 1, 2, 4 islands (5 problems, 1 shard)
3. Shards: 1, 2, 3 shards (5 problems, 1 island)
4. Combined: Various combinations

Run with: python examples/test_suite_systematic.py
"""

import sys
import time
from pathlib import Path
import os

# Disable litellm's async logging worker to avoid event loop issues between tests
os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

# Models
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Load full dataset once
print("Loading AIME dataset...")
trainset, _, _ = gepa.examples.aime.init_dataset()
print(f"✓ Loaded {len(trainset)} problems\n")


def create_dataset(size: int):
    """Create dataset of specified size."""
    return [
        DefaultDataInst(
            input=ex["input"],
            answer=ex["answer"],
            id=f"aime_{i}",
            additional_context=ex.get("additional_context"),
        )
        for i, ex in enumerate(trainset[:size])
    ]


def run_test(
    test_name: str,
    dataset_size: int,
    n_islands: int,
    shards: tuple,
    max_rounds: int = 3,
    timeout: int = 120,
):
    """
    Run a single test configuration.

    Returns: (success: bool, time: float, error: str | None)
    """
    print(f"\n{'=' * 80}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 80}")
    print(f"  Dataset: {dataset_size} problems")
    print(f"  Islands: {n_islands}")
    print(f"  Shards: {shards}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Timeout: {timeout}s")
    print()

    # Clean cache between tests
    import shutil
    cache_dir = Path(".turbo_gepa/")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    dataset = create_dataset(dataset_size)

    config = Config(
        shards=shards,
        eval_concurrency=8,
        max_mutations_per_round=4,
        mutation_buffer_min=2,
        queue_limit=64,
        batch_size=dataset_size,
        n_islands=n_islands,
    )
    config.cohort_quantile = 0.5
    config.eps_improve = 0.0
    config.migration_period = 2
    config.migration_k = 1

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
    )

    seed = "You are a helpful assistant. Answer in the format '### <final answer>'"

    start_time = time.time()
    result = None
    try:
        result = adapter.optimize(
            seeds=[seed],
            enable_auto_stop=False,
            max_rounds=max_rounds,
            display_progress=False,  # Quiet mode for batch testing
            optimize_temperature_after_convergence=False,
        )
        elapsed = time.time() - start_time

        # Check results
        pareto = result.get("pareto_entries", [])
        stats = result.get("evolution_stats", {})

        if pareto:
            best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0.0))
            quality = best.result.objectives.get("quality", 0.0)
            evals = stats.get('total_evaluations', 0)
            mutations = stats.get('mutations_generated', 0)

            print(f"✅ SUCCESS ({elapsed:.1f}s)")
            print(f"   Quality: {quality:.1%}")
            print(f"   Evaluations: {evals}")
            print(f"   Mutations: {mutations}")
            print(f"   Pareto size: {len(pareto)}")
            return True, elapsed, None
        else:
            print(f"❌ FAILED: No results generated ({elapsed:.1f}s)")
            return False, elapsed, "No results"

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED: Exception ({elapsed:.1f}s)")
        print(f"   Error: {str(e)}")
        return False, elapsed, str(e)
    finally:
        cleanup_litellm()


def main():
    """Run systematic test suite."""
    print("=" * 80)
    print("SYSTEMATIC TEST SUITE")
    print("=" * 80)
    print()
    print("This will test different configurations to identify issues.")
    print("Each test has a 120-second timeout.")
    print()

    results = []

    # ============================================================================
    # PHASE 1: Dataset Size (1 island, 1 shard)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: DATASET SIZE (single island, single shard)")
    print("=" * 80)

    for size in [2, 5, 10, 20]:
        success, elapsed, error = run_test(
            f"Dataset size: {size}",
            dataset_size=size,
            n_islands=1,
            shards=(1.0,),
        )
        results.append({
            "phase": "Dataset Size",
            "config": f"{size} problems",
            "success": success,
            "time": elapsed,
            "error": error,
        })

    # ============================================================================
    # PHASE 2: Number of Islands (5 problems, 1 shard)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: NUMBER OF ISLANDS (5 problems, single shard)")
    print("=" * 80)

    for n_islands in [1, 2, 4]:
        success, elapsed, error = run_test(
            f"Islands: {n_islands}",
            dataset_size=5,
            n_islands=n_islands,
            shards=(1.0,),
        )
        results.append({
            "phase": "Islands",
            "config": f"{n_islands} islands",
            "success": success,
            "time": elapsed,
            "error": error,
        })

    # ============================================================================
    # PHASE 3: Number of Shards (5 problems, 1 island)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: NUMBER OF SHARDS (5 problems, single island)")
    print("=" * 80)

    shard_configs = [
        (1.0,),
        (0.5, 1.0),
        (0.20, 0.50, 1.0),
    ]

    for shards in shard_configs:
        success, elapsed, error = run_test(
            f"Shards: {len(shards)} rungs",
            dataset_size=5,
            n_islands=1,
            shards=shards,
        )
        results.append({
            "phase": "Shards",
            "config": f"{len(shards)} shards {shards}",
            "success": success,
            "time": elapsed,
            "error": error,
        })

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    for phase in ["Dataset Size", "Islands", "Shards"]:
        phase_results = [r for r in results if r["phase"] == phase]
        print(f"\n{phase}:")
        print("-" * 80)
        for r in phase_results:
            status = "✅" if r["success"] else "❌"
            error_msg = f" ({r['error']})" if r['error'] else ""
            print(f"  {status} {r['config']:<30} {r['time']:>6.1f}s{error_msg}")

    # Identify issues
    print("\n" + "=" * 80)
    print("ISSUES IDENTIFIED")
    print("=" * 80)

    failures = [r for r in results if not r["success"]]
    if failures:
        print()
        for f in failures:
            print(f"❌ {f['phase']}: {f['config']}")
            if f['error']:
                print(f"   Error: {f['error']}")
    else:
        print("\n✅ All tests passed!")

    print("\n" + "=" * 80)

    return len(failures) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
