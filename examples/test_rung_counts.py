#!/usr/bin/env python3
"""
Quick test to verify rung inflight counts don't go negative.
Uses 3 problems, 2 rounds - should complete in < 2 minutes.
"""

import os
os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

print("Quick Rung Count Test")
print("=" * 80)

task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset[:3])
]

config = Config(
    shards=(0.5, 1.0),
    eval_concurrency=4,
    max_mutations_per_round=3,
    mutation_buffer_min=2,
    queue_limit=32,
    batch_size=3,
    n_islands=1,
)
config.cohort_quantile = 0.5

adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

seed = "You are a helpful assistant. Answer in the format '### <final answer>'"

result = adapter.optimize(
    seeds=[seed],
    enable_auto_stop=False,
    max_rounds=2,
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
cleanup_litellm()

pareto = result.get("pareto_entries", [])
stats = result.get("evolution_stats", {})

print()
print("=" * 80)
if pareto:
    best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0.0))
    quality = best.result.objectives.get("quality", 0.0)
    print(f"✅ Success! Quality: {quality:.1%}")
    print(f"   Evaluations: {stats.get('total_evaluations', 0)}")
    print(f"   Mutations: {stats.get('mutations_generated', 0)}")
else:
    print("❌ No results")
print("=" * 80)
