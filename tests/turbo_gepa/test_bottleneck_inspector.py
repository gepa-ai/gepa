#!/usr/bin/env python3
"""Bottleneck inspector: short real run that reports timing mix."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

# Ensure src/ on path
SRC_ROOT = Path(__file__).parent.parent.parent / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

import gepa  # type: ignore
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


REQUIRED_ENV = "OPENROUTER_API_KEY"


@pytest.mark.integration
@pytest.mark.skipif(REQUIRED_ENV not in os.environ, reason="OPENROUTER_API_KEY required")
def test_bottleneck_inspector(capfd):
    """Run a lightweight optimization and report bottleneck analysis."""

    # Clear cache to ensure accurate timing measurements
    cache_path = Path(".turbo_gepa/")
    if cache_path.exists():
        shutil.rmtree(cache_path)

    print("\n" + "="*80)
    print("TURBOGEPA BOTTLENECK ANALYSIS")
    print("="*80)

    trainset, _, _ = gepa.examples.aime.init_dataset()
    data = [
        DefaultDataInst(input=ex["input"], answer=ex["answer"], id=f"aime_{i}")
        for i, ex in enumerate(trainset[:8])
    ]

    config = Config(
        eval_concurrency=32,
        n_islands=1,
        shards=(0.6, 1.0),
        max_mutations_per_round=8,
        max_optimization_time_seconds=120,
        log_level="WARNING",
    )

    adapter = DefaultAdapter(
        dataset=data,
        task_lm="openrouter/openai/gpt-oss-20b:nitro",
        reflection_lm="openrouter/x-ai/grok-4-fast",
        config=config,
        auto_config=False,
    )

    print(f"\n📊 Dataset: {len(data)} problems")
    print(f"🔥 Concurrency: {config.eval_concurrency}")
    print(f"🎯 Shards: {config.shards}")
    print(f"⏱️  Max time: {config.max_optimization_time_seconds}s")
    print(f"🔄 Max rounds: 3\n")

    start = time.time()
    result = adapter.optimize(
        seeds=[
            "You are a helpful math assistant. Solve step by step and put final answer after ###."
        ],
        max_rounds=3,
        display_progress=False,
    )
    elapsed = time.time() - start

    print(f"\n✅ Optimization complete in {elapsed:.1f}s")

    stats = result.get("evolution_stats", {})
    print(f"\n📈 Performance Metrics:")
    print(f"   Total evaluations: {stats.get('total_evaluations', 0)}")
    print(f"   Mutations generated: {stats.get('mutations_generated', 0)}")
    print(f"   Mutations promoted: {stats.get('mutations_promoted', 0)}")
    print(f"   Unique candidates: {stats.get('unique_children', 0)}")

    if elapsed > 0:
        evals_per_sec = stats.get("total_evaluations", 0) / elapsed
        print(f"\n⚡ Throughput: {evals_per_sec:.2f} evaluations/sec")
        print(f"   ({elapsed / max(1, stats.get('total_evaluations', 1)):.2f}s per evaluation)")

    # Access metrics from adapter (correct timing approach)
    metrics = adapter._metrics

    # Get cumulative API times (these account for concurrency correctly)
    eval_api_time = metrics.eval_time_sum
    mutation_api_time = metrics.mutation_latency_sum
    total_api_time = eval_api_time + mutation_api_time

    # Calculate API time distribution
    eval_pct = (eval_api_time / total_api_time * 100) if total_api_time > 0 else 0
    mut_pct = (mutation_api_time / total_api_time * 100) if total_api_time > 0 else 0

    # Calculate effective parallelism (how much concurrency we achieved)
    parallelism = total_api_time / elapsed if elapsed > 0 else 0

    print(f"\n⏱️  Actual Time Breakdown (API Time):")
    print(f"   Evaluation API time: {eval_api_time:.1f}s ({eval_pct:.1f}%)")
    print(f"   Mutation API time: {mutation_api_time:.1f}s ({mut_pct:.1f}%)")
    print(f"   Total API time: {total_api_time:.1f}s")
    print(f"   Wall clock time: {elapsed:.1f}s")
    print(f"   Effective parallelism: {parallelism:.2f}x")

    print("\n" + "="*80)
    print("BOTTLENECK SUMMARY:")
    if mut_pct > 66:
        print("🔴 MUTATION GENERATION is the bottleneck (>66% of API time)")
        print("   → Consider using a faster reflection LLM (e.g., claude-3-haiku, gpt-4o-mini)")
    elif eval_pct > 66:
        print("🟢 EVALUATION is the bottleneck (>66% of API time)")
        print("   → This is expected - evaluation should dominate in production")
    else:
        print("🟡 BALANCED: Time split relatively evenly between mutation and evaluation")

    if parallelism < 2.0:
        print("\n⚠️  LOW PARALLELISM detected:")
        print(f"   Only {parallelism:.2f}x parallelism achieved")
        print("   → Consider increasing eval_concurrency or batch_size")
    elif parallelism > 4.0:
        print(f"\n✅ GOOD PARALLELISM: {parallelism:.2f}x concurrent operations")

    print("="*80 + "\n")

    # Basic assertion - just check that it ran
    assert stats.get("total_evaluations", 0) > 0, "Should have run some evaluations"

