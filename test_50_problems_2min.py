#!/usr/bin/env python3
"""Test with 50 AIME problems, 2 minute budget."""

import os
import shutil
import time
from pathlib import Path

os.environ["LITELLM_LOG"] = "ERROR"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Clear cache
cache_path = Path(".turbo_gepa/")
if cache_path.exists():
    shutil.rmtree(cache_path)

trainset, _, _ = gepa.examples.aime.init_dataset()
data = [
    DefaultDataInst(input=ex["input"], answer=ex["answer"], id=f"aime_{i}")
    for i, ex in enumerate(trainset[:50])
]

print("\n" + "="*80)
print("50 AIME PROBLEMS - 2 MINUTE TEST")
print("="*80)
print(f"\nDataset: {len(data)} problems")

# Don't pass any config - let auto_config do its magic!
# Just override the time budget and log level
adapter = DefaultAdapter(
    dataset=data,
    task_lm="openrouter/openai/gpt-oss-20b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    auto_config=True,  # This is the default, but explicit for clarity
)

# Override just the time budget after creation
adapter.config.max_optimization_time_seconds = 120
adapter.config.log_level = "WARNING"

print(f"Config:")
print(f"  Dynamic sharding: ENABLED (auto_config=True)")
print(f"  Actual shards: {adapter.config.shards}")
print(f"  Problems per rung: {[int(len(data) * s) for s in adapter.config.shards]}")
print(f"  Variance tolerance: {adapter.config.variance_tolerance}")
print(f"  Concurrency: {adapter.config.eval_concurrency}")
print(f"  Time budget: {adapter.config.max_optimization_time_seconds}s")
print(f"  Max mutations/round: {adapter.config.max_mutations_per_round}")

start = time.time()
result = adapter.optimize(
    seeds=["You are a helpful math assistant. Solve step by step and put final answer after ###."],
    max_rounds=10,  # High limit, time budget will stop us
    display_progress=False,
)
elapsed = time.time() - start

pareto = result.get("pareto_entries", [])
stats = result.get("evolution_stats", {})
metrics = adapter._metrics

# Group by rung
by_rung = {}
for entry in pareto:
    rung = entry.result.shard_fraction
    if rung not in by_rung:
        by_rung[rung] = []
    by_rung[rung].append(entry)

total_candidates = 1 + stats.get("mutations_generated", 0)
final_rung_count = len(by_rung.get(1.0, []))
promotion_rate = final_rung_count / total_candidates if total_candidates > 0 else 0

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\n⏱️  Time: {elapsed:.1f}s / {adapter.config.max_optimization_time_seconds}s budget")

print(f"\n📊 Evolution:")
print(f"  Rounds completed: {stats.get('rounds_completed', 0)}")
print(f"  Total evaluations: {stats.get('total_evaluations', 0)}")
print(f"  Mutations generated: {stats.get('mutations_generated', 0)}")
print(f"  Candidates at final rung: {final_rung_count}/{total_candidates} ({promotion_rate:.1%})")

print(f"\n🔍 Evaluations by rung:")
for rung in sorted(metrics.evaluations_by_shard.keys()):
    count = metrics.evaluations_by_shard[rung]
    problems = int(len(data) * rung)
    print(f"  Rung {rung:.1%} ({problems} problems): {count} evaluations")

print(f"\n🎯 Candidates by rung:")
for rung in sorted(by_rung.keys()):
    entries = by_rung[rung]
    problems = int(len(data) * rung)
    qualities = [e.result.objectives.get("quality", 0) for e in entries]
    avg_q = sum(qualities) / len(qualities) if qualities else 0
    max_q = max(qualities) if qualities else 0
    print(f"  Rung {rung:.1%} ({problems} problems): {len(entries)} candidates")
    print(f"    Quality: avg={avg_q:.1%}, max={max_q:.1%}")

# Best quality at final rung
best_quality = 0.0
if by_rung.get(1.0):
    best_quality = max(e.result.objectives.get("quality", 0) for e in by_rung[1.0])
    print(f"\n🏆 Best quality at final rung: {best_quality:.1%} ({int(best_quality * len(data))}/{len(data)} problems)")

# Timing breakdown
total_api_time = metrics.eval_time_sum + metrics.mutation_latency_sum
parallelism = total_api_time / elapsed if elapsed > 0 else 0
eval_pct = metrics.eval_time_sum / total_api_time * 100 if total_api_time > 0 else 0
mut_pct = metrics.mutation_latency_sum / total_api_time * 100 if total_api_time > 0 else 0

print(f"\n⚡ Performance:")
print(f"  Evaluation API time: {metrics.eval_time_sum:.1f}s ({eval_pct:.1f}%)")
print(f"  Mutation API time: {metrics.mutation_latency_sum:.1f}s ({mut_pct:.1f}%)")
print(f"  Total API time: {total_api_time:.1f}s")
print(f"  Effective parallelism: {parallelism:.2f}x")

# Throughput
evals_per_sec = stats.get('total_evaluations', 0) / elapsed if elapsed > 0 else 0
print(f"  Throughput: {evals_per_sec:.2f} evaluations/sec")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"In {elapsed:.1f}s with {len(data)} problems:")
print(f"  • {stats.get('mutations_generated', 0)} mutations generated")
print(f"  • {promotion_rate:.1%} promotion rate to final rung")
print(f"  • {best_quality:.1%} best quality achieved")
print(f"  • {parallelism:.2f}x parallelism")
print(f"  • {evals_per_sec:.2f} evals/sec throughput")
print("="*80 + "\n")
