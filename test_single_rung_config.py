#!/usr/bin/env python3
"""Test a single rung configuration."""

import os
import shutil
import sys
import time
from pathlib import Path

os.environ["LITELLM_LOG"] = "ERROR"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config, _default_variance_tolerance

# Get shards from command line, default to (0.6, 1.0)
if len(sys.argv) > 1:
    shards_str = sys.argv[1]
    shards = tuple(float(x) for x in shards_str.split(","))
else:
    shards = (0.6, 1.0)

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
    shards=shards,
    max_mutations_per_round=8,
    max_optimization_time_seconds=60,  # 1 minute timeout
    log_level="WARNING",
)

adapter = DefaultAdapter(
    dataset=data,
    task_lm="openrouter/openai/gpt-oss-20b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    config=config,
    auto_config=False,
)

print(f"\n{'='*80}")
print(f"Testing shards: {shards}")
print(f"{'='*80}")
print(f"Problems per rung: {[int(len(data) * s) for s in shards]}")

# Show variance tolerance
tolerance = _default_variance_tolerance(shards)
print(f"\nVariance tolerance:")
for rung, tol in sorted(tolerance.items()):
    problems = int(len(data) * rung)
    print(f"  Rung {rung:.1%} ({problems} problems): ±{tol:.1%}")

start = time.time()
result = adapter.optimize(
    seeds=["You are a helpful math assistant. Solve step by step and put final answer after ###."],
    max_rounds=2,
    display_progress=False,
)
elapsed = time.time() - start

pareto_entries = result.get("pareto_entries", [])
stats = result.get("evolution_stats", {})
metrics = adapter._metrics

# Group by rung
by_rung = {}
for entry in pareto_entries:
    rung = entry.result.shard_fraction
    if rung not in by_rung:
        by_rung[rung] = []
    by_rung[rung].append(entry)

total_candidates = 1 + stats.get("mutations_generated", 0)
final_rung_count = len(by_rung.get(1.0, []))
promotion_rate = final_rung_count / total_candidates if total_candidates > 0 else 0

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"Time: {elapsed:.1f}s")
print(f"Evaluations: {stats.get('total_evaluations', 0)}")
print(f"Mutations generated: {stats.get('mutations_generated', 0)}")
print(f"Candidates at final rung: {final_rung_count}/{total_candidates} ({promotion_rate:.1%})")

# Best quality
if by_rung.get(1.0):
    best_quality = max(e.result.objectives.get("quality", 0) for e in by_rung[1.0])
    print(f"Best quality: {best_quality:.1%}")
else:
    print(f"Best quality: N/A (no candidates reached final rung)")

# Parallelism
total_api_time = metrics.eval_time_sum + metrics.mutation_latency_sum
parallelism = total_api_time / elapsed if elapsed > 0 else 0
print(f"Parallelism: {parallelism:.2f}x")

print(f"\nCandidates by rung:")
for rung in sorted(by_rung.keys()):
    count = by_rung[rung]
    problems = int(len(data) * rung)
    print(f"  Rung {rung:.1%} ({problems} problems): {len(count)} candidates")

print(f"\nEvaluations by rung:")
for rung in sorted(metrics.evaluations_by_shard.keys()):
    count = metrics.evaluations_by_shard[rung]
    problems = int(len(data) * rung)
    print(f"  Rung {rung:.1%} ({problems} problems): {count} evaluations")

print(f"{'='*80}\n")

# Summary line for easy comparison
print(f"SUMMARY: {shards} | {elapsed:.1f}s | {stats.get('total_evaluations', 0)} evals | "
      f"{stats.get('mutations_generated', 0)} muts | {promotion_rate:.1%} promoted | "
      f"{best_quality if by_rung.get(1.0) else 0:.1%} quality")
