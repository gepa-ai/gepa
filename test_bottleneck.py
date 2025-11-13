#!/usr/bin/env python3
"""Bottleneck analysis script to measure utilization and throughput."""

import time
import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst

# Test with 5 problems
trainset, _, _ = gepa.examples.aime.init_dataset()
data = [
    DefaultDataInst(input=ex["input"], answer=ex["answer"], id=f"aime_{i}")
    for i, ex in enumerate(trainset[:50])
]

print("=" * 80)
print("BOTTLENECK ANALYSIS")
print("=" * 80)

# Test with your preferred model
adapter = DefaultAdapter(
    dataset=data,
    task_lm="openrouter/openai/gpt-oss-20b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    auto_config=True,
)

print(f"\nConfig:")
print(f"  Task model: openai/gpt-oss-20b:nitro")
print(f"  Concurrency: {adapter.config.eval_concurrency}")
print(f"  Shards: {adapter.config.shards}")
print(f"  Batch size: {adapter.config.batch_size}")
print(f"  Max mutations/round: {adapter.config.max_mutations_per_round}")

start = time.time()
result = adapter.optimize(
    seeds=["Solve step by step and put answer after ###"],
    max_rounds=5,
    display_progress=True,
)
elapsed = time.time() - start

stats = result.get("evolution_stats", {})
evals = stats.get("total_evaluations", 0)
mutations = stats.get("mutations_generated", 0)

print(f"\n⏱️  Results:")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Evaluations: {evals}")
print(f"  Mutations: {mutations}")
if evals > 0:
    print(f"  Time per eval: {elapsed/evals:.2f}s")
    print(f"  Throughput: {evals/elapsed:.1f} evals/s")

# Calculate theoretical max throughput with no rate limits
print(f"\n📊 Analysis:")
if evals > 0:
    avg_latency = elapsed / evals
    print(f"  Avg eval latency: {avg_latency:.2f}s")
    print(
        f"  With concurrency={adapter.config.eval_concurrency}, theoretical max: {adapter.config.eval_concurrency/avg_latency:.1f} evals/s"
    )
    print(f"  Actual: {evals/elapsed:.1f} evals/s")
    utilization = (
        (evals / elapsed) / (adapter.config.eval_concurrency / avg_latency) * 100
    )
    print(f"  Utilization: {utilization:.0f}%")

    if utilization < 50:
        print(f"\n⚠️  WARNING: Low utilization ({utilization:.0f}%)")
        print(f"  System is not keeping API saturated with requests")
        print(f"  Recommendations:")
        print(
            f"    - Increase eval_concurrency from {adapter.config.eval_concurrency} to 16-32"
        )
        print(f"    - Increase max_mutations_per_round to keep queue full")
        print(f"    - Consider reducing shards (fewer rungs = less pruning)")
    elif utilization < 80:
        print(f"\n⚠️  Moderate utilization ({utilization:.0f}%)")
        print(f"  Room for improvement")
    else:
        print(f"\n✅ Good utilization ({utilization:.0f}%)")

print("=" * 80)
