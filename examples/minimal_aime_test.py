#!/usr/bin/env python3
"""
Minimal AIME Test - Ultra-quick smoke test

This is the absolute minimal test to verify TurboGEPA is working.
Uses just 2 problems and 3 rounds for fastest possible feedback.

Usage:
    python examples/minimal_aime_test.py

Expected runtime: < 1 minute
"""

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

print("🔬 Minimal AIME Smoke Test")
print("=" * 60)

# Configure LLMs
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Load minimal dataset
print("Loading 2 AIME problems...")
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
print(f"✓ Loaded {len(dataset)} problems\n")

# Minimal configuration - single island for fastest results
config = Config(
    shards=(1.0,),
    eval_concurrency=8,
    max_mutations_per_round=4,
    mutation_buffer_min=2,
    queue_limit=64,
    batch_size=2,
    n_islands=1,  # Single island - no multi-processing overhead
)
config.cohort_quantile = 0.5
config.eps_improve = 0.0

print("Creating adapter...")
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

seed = "You are a helpful assistant. Answer in the format '### <final answer>'"
print(f"Seed: {seed[:50]}...\n")

print("Running 3 rounds of optimization...")
print("-" * 60)

result = adapter.optimize(
    seeds=[seed],
    enable_auto_stop=False,
    max_rounds=3,
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
cleanup_litellm()

print("-" * 60)

# Check results
pareto = result.get("pareto_entries", [])
stats = result.get("evolution_stats", {})

if pareto:
    best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0.0))
    quality = best.result.objectives.get("quality", 0.0)

    print(f"\n✅ Success!")
    print(f"   Quality: {quality:.1%}")
    print(f"   Pareto size: {len(pareto)}")
    print(f"   Evaluations: {stats.get('total_evaluations', 0)}")
    print(f"   Mutations: {stats.get('mutations_generated', 0)}")

    if quality > 0:
        print(f"\n✅ ✅ TurboGEPA is working correctly!")
    else:
        print(f"\n⚠️  Quality is 0% - check LLM connectivity")
else:
    print("\n❌ No results - check configuration")

print("\n" + "=" * 60)
