#!/usr/bin/env python3
"""
Adaptive Rung Adjustment Test

Demonstrates how TurboGEPA dynamically adjusts rung positions based on
promotion rates during optimization.

Starting rungs: (0.2, 0.5, 1.0)
- If a rung promotes >60% → shrink (make cheaper)
- If a rung promotes <40% → expand (make more expensive)

Watch for log messages like:
🔧 Adaptive shards: [0.20, 0.50, 1.0] → [0.18, 0.56, 1.0]

Run with: python examples/test_adaptive_rungs.py
"""

import os
os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

print("=" * 80)
print("ADAPTIVE RUNG ADJUSTMENT TEST")
print("=" * 80)
print()
print("This test demonstrates dynamic adjustment of rung positions.")
print("Starting with 3 rungs: (0.20, 0.50, 1.0)")
print("Rungs will automatically adjust based on promotion rates.")
print()

task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Use 5 problems - enough to show meaningful promotion patterns
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

# Configure 3 rungs with adaptive sharding enabled
print("⚙️  Configuring 3-rung ASHA with adaptive adjustment...")
config = Config(
    shards=(0.20, 0.50, 1.0),  # Three rungs: 20%, 50%, 100%
    eval_concurrency=8,
    max_mutations_per_round=8,
    mutation_buffer_min=4,
    queue_limit=64,
    batch_size=5,
    n_islands=1,
)

# Adaptive sharding is enabled by default
# config.adaptive_shards_enabled = True  # This is the default

config.cohort_quantile = 0.5  # Top 50% advance
config.eps_improve = 0.0

print(f"   Initial rungs: {config.shards}")
print(f"   Promotion threshold: top {config.cohort_quantile:.0%}")
print(f"   Adaptive adjustment: ENABLED")
print()
print("Rungs will adjust if:")
print("  - >60% promotions → shrink rung (make cheaper)")
print("  - <40% promotions → expand rung (need more data)")
print()

adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

seed = "You are a helpful assistant. Answer in the format '### <final answer>'"
print("🚀 Starting optimization with adaptive rungs (6 rounds)...")
print()
print("Watch for adaptive shard adjustments in the logs!")
print("=" * 80)
print()

result = adapter.optimize(
    seeds=[seed],
    enable_auto_stop=False,
    max_rounds=6,  # More rounds to see multiple adjustments
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
cleanup_litellm()

print()
print("=" * 80)
print("RESULTS")
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

    # Show final rung configuration
    final_shards = config.shards
    print()
    print(f"   Initial rungs: (0.20, 0.50, 1.0)")
    print(f"   Final rungs:   {final_shards}")

    if final_shards != (0.20, 0.50, 1.0):
        print()
        print("   ✨ Rungs were dynamically adjusted during optimization!")
    else:
        print()
        print("   ℹ️  Rungs remained stable (promotion rates were balanced)")
else:
    print("❌ No results")

print("=" * 80)
