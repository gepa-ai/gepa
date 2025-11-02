#!/usr/bin/env python3
"""
Dynamic Sharding Demo - ASHA Successive Halving

This example demonstrates TurboGEPA's dynamic sharding (ASHA) feature,
which progressively evaluates candidates on increasingly larger portions
of the dataset, pruning poor performers early.

ASHA (Asynchronous Successive Halving Algorithm) saves computation by:
1. Evaluating all candidates on a small shard first (e.g., 20% of data)
2. Promoting only top performers to the next shard (e.g., 50% of data)
3. Finally evaluating the best candidates on 100% of data

This can reduce evaluations by 60-80% compared to always using full dataset.

Usage:
    python examples/demo_dynamic_sharding.py

Expected output:
    - Shows candidates being evaluated at different shard levels
    - Demonstrates ASHA pruning in action
    - Tracks promotion rates between rungs
"""

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

print("=" * 80)
print("DYNAMIC SHARDING (ASHA) DEMO")
print("=" * 80)
print()

# Configure LLMs
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Load dataset - use enough data to make sharding worthwhile
print("📊 Loading AIME dataset...")
trainset, _, _ = gepa.examples.aime.init_dataset()
DATASET_SIZE = 20  # Use 20 problems to see sharding in action
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset[:DATASET_SIZE])
]
print(f"✓ Loaded {len(dataset)} problems")
print()

# Configure multi-shard ASHA
print("⚙️  Configuring ASHA with 3 rungs...")
print()

# Three-rung ASHA: 20% → 50% → 100%
config = Config(
    shards=(0.20, 0.50, 1.0),  # Three rungs
    eval_concurrency=16,
    max_mutations_per_round=12,
    mutation_buffer_min=6,
    queue_limit=128,
    batch_size=20,
    n_islands=1,  # Single island for clarity
)

# ASHA promotion settings
config.cohort_quantile = 0.6  # Top 40% advance (60th percentile cutoff)
config.eps_improve = 0.0  # Don't over-prune equal quality

print("📋 ASHA Configuration:")
print(f"   Shard 1 (Rung 0): {config.shards[0]:.0%} of data ({int(DATASET_SIZE * config.shards[0])} problems)")
print(f"   Shard 2 (Rung 1): {config.shards[1]:.0%} of data ({int(DATASET_SIZE * config.shards[1])} problems)")
print(f"   Shard 3 (Rung 2): {config.shards[2]:.0%} of data ({int(DATASET_SIZE * config.shards[2])} problems)")
print()
print(f"   Promotion rule: Top {int((1 - config.cohort_quantile) * 100)}% advance to next rung")
print(f"   Concurrency: {config.eval_concurrency}")
print(f"   Mutations/round: {config.max_mutations_per_round}")
print()
print("💡 What to watch for:")
print("   - Most candidates will be evaluated on 20% shard only")
print("   - Only best ~40% advance to 50% shard")
print("   - Only best ~16% reach 100% shard")
print("   - This saves ~60-70% of evaluations!")
print()

# Create adapter
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

seed = "You are a helpful assistant. Answer in the format '### <final answer>'"
print("🚀 Starting optimization with ASHA...")
print(f"   Seed: {seed[:50]}...")
print()
print("=" * 80)
print()

result = adapter.optimize(
    seeds=[seed],
    enable_auto_stop=False,
    max_rounds=8,
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
cleanup_litellm()

print()
print("=" * 80)
print("ASHA ANALYSIS")
print("=" * 80)
print()

# Analyze shard distribution
pareto = result.get("pareto_entries", [])
stats = result.get("evolution_stats", {})

if pareto:
    # Group by shard
    by_shard = {}
    for entry in pareto:
        shard = entry.result.shard_fraction or 0.0
        if shard not in by_shard:
            by_shard[shard] = []
        by_shard[shard].append(entry)

    print("📊 Candidates by Shard:")
    for shard in sorted(by_shard.keys()):
        count = len(by_shard[shard])
        pct = count / len(pareto) * 100
        best_quality = max(e.result.objectives.get("quality", 0.0) for e in by_shard[shard])
        print(f"   {shard:5.0%} shard: {count:2d} candidates ({pct:4.1f}%) - best quality: {best_quality:.1%}")

    print()
    print("✅ Results:")
    best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0.0))
    best_quality = best.result.objectives.get("quality", 0.0)
    best_shard = best.result.shard_fraction or 1.0

    print(f"   Best quality: {best_quality:.1%} @ {best_shard:.0%} shard")
    print(f"   Pareto size: {len(pareto)}")
    print(f"   Total evaluations: {stats.get('total_evaluations', 0)}")
    print(f"   Mutations generated: {stats.get('mutations_generated', 0)}")
    print(f"   Mutations promoted: {stats.get('mutations_promoted', 0)}")

    # Calculate savings
    total_evals = stats.get('total_evaluations', 0)
    if total_evals > 0:
        # If all candidates were evaluated on full dataset
        full_dataset_evals = len(pareto) * DATASET_SIZE
        actual_cost = sum(
            len(by_shard.get(s, [])) * int(DATASET_SIZE * s)
            for s in config.shards
        )
        savings = (1 - actual_cost / full_dataset_evals) * 100 if full_dataset_evals > 0 else 0

        print()
        print("💰 ASHA Efficiency:")
        print(f"   Actual evaluations: {total_evals}")
        print(f"   Without ASHA: ~{len(pareto)} candidates × {DATASET_SIZE} = {len(pareto) * DATASET_SIZE} evals")
        print(f"   Estimated savings: ~{savings:.0f}%")

    print()
    print("📝 Best prompt:")
    print("-" * 80)
    print(best.candidate.text[:300] + "..." if len(best.candidate.text) > 300 else best.candidate.text)
    print("-" * 80)
else:
    print("❌ No results generated")

print()
print("=" * 80)
print("🎓 Key Takeaways:")
print("   ✓ ASHA evaluates candidates progressively on larger shards")
print("   ✓ Poor candidates are pruned early, saving compute")
print("   ✓ Best candidates still get evaluated on full dataset")
print("   ✓ Typical savings: 60-80% fewer evaluations")
print()
print("💡 Tuning Tips:")
print("   - More shards = more aggressive pruning, but more risk")
print("   - cohort_quantile controls promotion rate (0.5 = top 50%)")
print("   - First shard should have ~15 examples for reliability")
print("=" * 80)
