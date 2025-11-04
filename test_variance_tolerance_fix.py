#!/usr/bin/env python3
"""Test variance tolerance fix for binary scoring tasks."""

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
    for i, ex in enumerate(trainset[:20])
]

print("\n" + "="*80)
print("VARIANCE TOLERANCE FIX TEST")
print("="*80)

# Test 1: Default aggressive tolerance
print("\n[TEST 1] Default Tolerance (aggressive)")
print("-"*80)

config1 = Config(
    eval_concurrency=32,
    n_islands=1,
    shards=(0.5, 1.0),
    max_mutations_per_round=8,
    max_optimization_time_seconds=90,
    log_level="WARNING",
)

adapter1 = DefaultAdapter(
    dataset=data,
    task_lm="openrouter/openai/gpt-oss-20b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    config=config1,
    auto_config=False,
)

# Show default tolerance
from turbo_gepa.config import _default_variance_tolerance
default_tol = _default_variance_tolerance(config1.shards)
print(f"Variance tolerance: {default_tol}")
print(f"  Rung 50% (10 problems): ±{default_tol[0.5]:.1%}")
print(f"  Rung 100% (20 problems): ±{default_tol[1.0]:.1%}")

start1 = time.time()
result1 = adapter1.optimize(
    seeds=["You are a helpful math assistant. Solve step by step and put final answer after ###."],
    max_rounds=3,
    display_progress=False,
)
elapsed1 = time.time() - start1

pareto1 = result1.get("pareto_entries", [])
stats1 = result1.get("evolution_stats", {})

by_rung1 = {}
for entry in pareto1:
    rung = entry.result.shard_fraction
    if rung not in by_rung1:
        by_rung1[rung] = []
    by_rung1[rung].append(entry)

total_candidates1 = 1 + stats1.get("mutations_generated", 0)
final_rung_count1 = len(by_rung1.get(1.0, []))
promotion_rate1 = final_rung_count1 / total_candidates1 if total_candidates1 > 0 else 0

print(f"\nResults:")
print(f"  Time: {elapsed1:.1f}s")
print(f"  Mutations generated: {stats1.get('mutations_generated', 0)}")
print(f"  Final rung: {final_rung_count1}/{total_candidates1} ({promotion_rate1:.1%})")
print(f"  Candidates by rung: {[(f'{r:.0%}', len(c)) for r, c in sorted(by_rung1.items())]}")

# Test 2: Relaxed tolerance for binary scoring
print("\n[TEST 2] Relaxed Tolerance (for binary scoring)")
print("-"*80)

config2 = Config(
    eval_concurrency=32,
    n_islands=1,
    shards=(0.5, 1.0),
    max_mutations_per_round=8,
    max_optimization_time_seconds=90,
    log_level="WARNING",
    # Custom tolerance: ±12% at 50% rung, ±2% at 100% rung
    variance_tolerance={0.5: 0.12, 1.0: 0.02},
)

adapter2 = DefaultAdapter(
    dataset=data,
    task_lm="openrouter/openai/gpt-oss-20b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    config=config2,
    auto_config=False,
)

print(f"Variance tolerance: {config2.variance_tolerance}")
print(f"  Rung 50% (10 problems): ±{config2.variance_tolerance[0.5]:.1%}")
print(f"  Rung 100% (20 problems): ±{config2.variance_tolerance[1.0]:.1%}")

start2 = time.time()
result2 = adapter2.optimize(
    seeds=["You are a helpful math assistant. Solve step by step and put final answer after ###."],
    max_rounds=3,
    display_progress=False,
)
elapsed2 = time.time() - start2

pareto2 = result2.get("pareto_entries", [])
stats2 = result2.get("evolution_stats", {})

by_rung2 = {}
for entry in pareto2:
    rung = entry.result.shard_fraction
    if rung not in by_rung2:
        by_rung2[rung] = []
    by_rung2[rung].append(entry)

total_candidates2 = 1 + stats2.get("mutations_generated", 0)
final_rung_count2 = len(by_rung2.get(1.0, []))
promotion_rate2 = final_rung_count2 / total_candidates2 if total_candidates2 > 0 else 0

print(f"\nResults:")
print(f"  Time: {elapsed2:.1f}s")
print(f"  Mutations generated: {stats2.get('mutations_generated', 0)}")
print(f"  Final rung: {final_rung_count2}/{total_candidates2} ({promotion_rate2:.1%})")
print(f"  Candidates by rung: {[(f'{r:.0%}', len(c)) for r, c in sorted(by_rung2.items())]}")

# Compare quality
best_quality1 = 0.0
if by_rung1.get(1.0):
    best_quality1 = max(e.result.objectives.get("quality", 0) for e in by_rung1[1.0])

best_quality2 = 0.0
if by_rung2.get(1.0):
    best_quality2 = max(e.result.objectives.get("quality", 0) for e in by_rung2[1.0])

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"{'Metric':<30} {'Default':<20} {'Relaxed':<20}")
print("-"*80)
print(f"{'Tolerance @ 50%':<30} {'±2.8%':<20} {'±12.0%':<20}")
print(f"{'Promotion rate':<30} {promotion_rate1:<20.1%} {promotion_rate2:<20.1%}")
print(f"{'Candidates at final':<30} {f'{final_rung_count1}/{total_candidates1}':<20} {f'{final_rung_count2}/{total_candidates2}':<20}")
print(f"{'Best quality':<30} {best_quality1:<20.1%} {best_quality2:<20.1%}")
print(f"{'Time':<30} {f'{elapsed1:.1f}s':<20} {f'{elapsed2:.1f}s':<20}")

improvement = promotion_rate2 - promotion_rate1
print(f"\n{'Promotion improvement':<30} {improvement:+.1%}")

print("\n" + "="*80)
if promotion_rate2 > promotion_rate1:
    print("✅ CONFIRMED: Relaxed tolerance allows more promotions")
    print("   Binary scoring with small datasets needs higher tolerance at early rungs")
else:
    print("⚠️  Relaxed tolerance did not increase promotions - investigate further")
print("="*80 + "\n")
