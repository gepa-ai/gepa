#!/usr/bin/env python3
"""
Scheduler diagnostics: understand what's happening with rung promotions.
"""

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
    for i, ex in enumerate(trainset[:8])
]

config = Config(
    eval_concurrency=32,
    n_islands=1,
    shards=(0.6, 1.0),  # Two rungs: 60% then 100%
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

print("\n" + "="*80)
print("SCHEDULER DIAGNOSTICS")
print("="*80)
print(f"\n📊 Config:")
print(f"   Dataset: {len(data)} problems")
print(f"   Shards: {config.shards}")
print(f"   Problems per rung: {[int(len(data) * s) for s in config.shards]}")
print(f"   Concurrency: {config.eval_concurrency}")
print(f"   Max mutations/round: {config.max_mutations_per_round}")
print(f"   Max time: {config.max_optimization_time_seconds}s\n")

start = time.time()
result = adapter.optimize(
    seeds=["You are a helpful math assistant. Solve step by step and put final answer after ###."],
    max_rounds=3,
    display_progress=False,
)
elapsed = time.time() - start

print(f"\n✅ Complete in {elapsed:.1f}s\n")

# Analyze pareto entries by rung
pareto_entries = result.get("pareto_entries", [])
print(f"📈 Pareto Frontier: {len(pareto_entries)} candidates")

if pareto_entries:
    # Group by shard fraction (rung)
    by_rung = {}
    for entry in pareto_entries:
        rung = entry.result.shard_fraction
        if rung not in by_rung:
            by_rung[rung] = []
        by_rung[rung].append(entry)

    print(f"\n🎯 Candidates by Rung:")
    for rung in sorted(by_rung.keys()):
        entries = by_rung[rung]
        problems_evaluated = int(len(data) * rung)
        print(f"   Rung {rung:.1%} ({problems_evaluated} problems): {len(entries)} candidates")

        # Show quality distribution
        qualities = [e.result.objectives.get("quality", 0) for e in entries]
        if qualities:
            avg_quality = sum(qualities) / len(qualities)
            max_quality = max(qualities)
            print(f"      Quality: avg={avg_quality:.1%}, max={max_quality:.1%}")

# Metrics analysis
metrics = adapter._metrics
print(f"\n⏱️  Timing Breakdown:")
print(f"   Evaluation API time: {metrics.eval_time_sum:.1f}s")
print(f"   Mutation API time: {metrics.mutation_latency_sum:.1f}s")
print(f"   Total API time: {metrics.eval_time_sum + metrics.mutation_latency_sum:.1f}s")
print(f"   Wall clock time: {elapsed:.1f}s")
parallelism = (metrics.eval_time_sum + metrics.mutation_latency_sum) / elapsed if elapsed > 0 else 0
print(f"   Effective parallelism: {parallelism:.2f}x")

print(f"\n📊 Evolution Stats:")
stats = result.get("evolution_stats", {})
print(f"   Total evaluations: {stats.get('total_evaluations', 0)}")
print(f"   Mutations generated: {stats.get('mutations_generated', 0)}")
print(f"   Mutations promoted: {stats.get('mutations_promoted', 0)}")

# Evaluations per rung
print(f"\n🔍 Evaluations by Rung:")
for rung, count in sorted(metrics.evaluations_by_shard.items()):
    problems = int(len(data) * rung)
    print(f"   Rung {rung:.1%} ({problems} problems): {count} evaluations")

# Calculate expected vs actual
print(f"\n💡 Expected Behavior:")
seed_count = 1
mutation_count = stats.get('mutations_generated', 0)
total_candidates = seed_count + mutation_count
print(f"   Total candidates: {total_candidates} (1 seed + {mutation_count} mutations)")

# With 2-rung ASHA (0.6, 1.0), expected evaluations:
# - All candidates evaluated at rung 0.6
# - Top performers promoted to rung 1.0
rung1_expected = total_candidates * int(len(data) * 0.6)
rung2_promoted = len(by_rung.get(1.0, []))
rung2_expected = rung2_promoted * int(len(data) * (1.0 - 0.6))
total_expected = rung1_expected + rung2_expected

print(f"   Expected rung 0.6 evals: {total_candidates} candidates × {int(len(data) * 0.6)} problems = {rung1_expected}")
print(f"   Expected rung 1.0 evals: {rung2_promoted} promoted × {int(len(data) * (1.0 - 0.6))} problems = {rung2_expected}")
print(f"   Expected total evals: {total_expected}")
print(f"   Actual total evals: {stats.get('total_evaluations', 0)}")

efficiency = stats.get('total_evaluations', 0) / total_expected if total_expected > 0 else 0
print(f"   Efficiency: {efficiency:.1%} (cache hits reduce evals)")

print("\n" + "="*80)
print("🎯 BOTTLENECK DIAGNOSIS:")

eval_pct = metrics.eval_time_sum / (metrics.eval_time_sum + metrics.mutation_latency_sum) * 100 if (metrics.eval_time_sum + metrics.mutation_latency_sum) > 0 else 0
mut_pct = metrics.mutation_latency_sum / (metrics.eval_time_sum + metrics.mutation_latency_sum) * 100 if (metrics.eval_time_sum + metrics.mutation_latency_sum) > 0 else 0

print(f"   Evaluation: {eval_pct:.1f}% of API time")
print(f"   Mutation: {mut_pct:.1f}% of API time")

if mut_pct > 66:
    print("\n🔴 MUTATION BOTTLENECK:")
    print("   → Reflection LLM (grok-4-fast) is too slow")
    print("   → Consider faster model (claude-3-haiku, gpt-4o-mini)")
elif eval_pct > 66:
    print("\n🟢 EVALUATION BOTTLENECK (expected):")
    print("   → Task LLM dominates (good - this is production workload)")
    if parallelism < 2.0:
        print("   → Low parallelism - increase eval_concurrency")
else:
    print("\n🟡 BALANCED: Time split evenly")

if parallelism < 2.0:
    print(f"\n⚠️  LOW PARALLELISM: {parallelism:.2f}x")
    print("   → Increase eval_concurrency or batch_size")
elif parallelism > 4.0:
    print(f"\n✅ GOOD PARALLELISM: {parallelism:.2f}x")

print("="*80 + "\n")
