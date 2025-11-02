#!/usr/bin/env python3
"""
Simple Dynamic Sharding Test

Minimal test to verify ASHA (dynamic sharding) is working correctly.
Uses 5 problems with 2 rungs: 50% → 100%

Expected behavior:
- Candidates start on Rung 0 (50% of data = 2-3 problems)
- Top performers get promoted to Rung 1 (100% of data = 5 problems)
- Should see promotions in the logs
"""

import os

# Disable litellm's async logging worker to avoid event loop issues
os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

print("=" * 80)
print("SIMPLE DYNAMIC SHARDING TEST")
print("=" * 80)
print()

task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Load 5 problems
print("📊 Loading 5 AIME problems...")
trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset[:5])
]
print(f"✓ Loaded {len(dataset)} problems")
print()

# Configure with 2 shards (rungs): 50% → 100%
print("⚙️  Configuring 2-rung ASHA sharding...")
config = Config(
    shards=(0.5, 1.0),  # Two rungs
    eval_concurrency=8,
    max_mutations_per_round=6,
    mutation_buffer_min=3,
    queue_limit=64,
    batch_size=5,
    n_islands=1,  # Single island for simplicity
)
config.cohort_quantile = 0.5  # Top 50% advance to next rung
config.eps_improve = 0.0

print(f"   Shards: {config.shards}")
print(f"   Rung 0: {int(5 * 0.5)} problems (50%)")
print(f"   Rung 1: {int(5 * 1.0)} problems (100%)")
print(f"   Top {config.cohort_quantile:.0%} advance to next rung")
print()

adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

seed = "You are a helpful assistant. Answer in the format '### <final answer>'"
print("🚀 Starting optimization with 2-rung sharding (5 rounds)...")
print()

result = adapter.optimize(
    seeds=[seed],
    enable_auto_stop=False,
    max_rounds=5,
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

    # Show rung statistics if available
    rung_stats = stats.get('rung_stats', {})
    if rung_stats:
        print()
        print("   Rung Activity:")
        for rung_idx, rung_data in rung_stats.items():
            print(f"     Rung {rung_idx}: {rung_data.get('evaluations', 0)} evaluations, "
                  f"{rung_data.get('promotions', 0)} promotions")
else:
    print("❌ No results")

print("=" * 80)
