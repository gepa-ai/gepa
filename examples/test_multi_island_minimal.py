#!/usr/bin/env python3
"""
Minimal Multi-Island Test - Debug version

Uses only 2 problems and 2 islands to debug why multi-island gets stuck.
"""

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

print("=" * 80)
print("MINIMAL MULTI-ISLAND TEST (DEBUG)")
print("=" * 80)
print()

task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Use minimal dataset like minimal_aime_test
print("📊 Loading 2 AIME problems...")
trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset[:2])
]
print(f"✓ Loaded {len(dataset)} problems")
print()

# Configure 2 islands (minimal multi-island)
print("⚙️  Configuring 2-island optimization...")
config = Config(
    shards=(1.0,),
    eval_concurrency=8,
    max_mutations_per_round=4,
    mutation_buffer_min=2,
    queue_limit=64,
    batch_size=2,
    n_islands=2,  # Just 2 islands
)
config.migration_period = 2
config.migration_k = 1
config.cohort_quantile = 0.5
config.eps_improve = 0.0

print(f"   Islands: {config.n_islands}")
print(f"   Concurrency per island: {config.eval_concurrency}")
print(f"   Migration every {config.migration_period} rounds")
print()

adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

seed = "You are a helpful assistant. Answer in the format '### <final answer>'"
print("🚀 Starting 2-island optimization (3 rounds)...")
print()

result = adapter.optimize(
    seeds=[seed],
    enable_auto_stop=False,
    max_rounds=3,
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
cleanup_litellm()

print()
print("=" * 80)
pareto = result.get("pareto_entries", [])
stats = result.get("evolution_stats", {})

if pareto:
    best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0.0))
    quality = best.result.objectives.get("quality", 0.0)
    print(f"✅ Success! Quality: {quality:.1%}")
    print(f"   Pareto size: {len(pareto)}")
    print(f"   Evaluations: {stats.get('total_evaluations', 0)}")
    print(f"   Mutations: {stats.get('mutations_generated', 0)}")
else:
    print("❌ No results")

print("=" * 80)
