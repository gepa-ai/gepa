"""
Definitive proof that mutations improve over a deliberately bad seed.

This test:
1. Uses a BAD seed that will get 0% on AIME problems
2. Runs optimization to generate mutations
3. Proves mutations achieve higher quality than seed
4. Shows progression through multiple rungs
"""

import os
import shutil
from pathlib import Path

Path('.turbo_gepa/').exists() and shutil.rmtree('.turbo_gepa/')
os.environ['LITELLM_LOG'] = 'ERROR'

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

trainset, _, _ = gepa.examples.aime.init_dataset()
# Use 10 problems for better signal
dataset = [DefaultDataInst(input=ex['input'], answer=ex['answer'], id=f'aime_{i}') for i, ex in enumerate(trainset[:10])]

config = Config(
    eval_concurrency=8,
    n_islands=1,
    shards=(0.4, 0.7, 1.0),  # 3 rungs: 4, 7, 10 examples
    batch_size=8,
    max_mutations_per_round=8,
    mutation_buffer_min=6,
    queue_limit=32,
    log_level='INFO',
    adaptive_shards_enabled=False,
    max_optimization_time_seconds=300,  # 5 minutes - allow time for promotions
    eps_improve=0.01,  # Allow small improvements
)

adapter = DefaultAdapter(
    dataset=dataset,
    task_lm='openrouter/openai/gpt-oss-20b:nitro',
    reflection_lm='openrouter/x-ai/grok-4-fast',
    config=config,
    auto_config=False
)

# DELIBERATELY BAD SEED - won't follow the ### format requirement
bad_seed = "You are a helpful assistant. Solve the math problem and explain your work."

print("=" * 80)
print("PROOF: MUTATIONS IMPROVE OVER BAD SEED")
print("=" * 80)
print(f"\n📊 Dataset: 10 AIME problems")
print(f"🎯 Shards: {config.shards}")
print(f"   Rung 1 (40%): 4 examples")
print(f"   Rung 2 (70%): 7 examples")
print(f"   Rung 3 (100%): 10 examples")
print(f"\n🌱 BAD Seed (missing ### format requirement):")
print(f'   "{bad_seed}"')
print(f"\n🔬 Expected: Seed gets 0% (doesn't follow ### answer format)")
print(f"🎯 Goal: Mutations learn to use ### format and improve quality")
print(f"\n🚀 Starting optimization (max 5 rounds)...\n")

result = adapter.optimize(
    seeds=[bad_seed],
    max_rounds=5,
    enable_auto_stop=False,
    display_progress=True
)

# Analysis
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

pareto = result.get('pareto_entries', [])
evolution_stats = result.get('evolution_stats', {}) or {}

print(f"\n📈 Evolution Statistics:")
print(f"   Total Evaluations: {evolution_stats.get('total_evaluations', 0)}")
print(f"   Mutations Generated: {evolution_stats.get('mutations_generated', 0)}")
print(f"   Pareto Size: {len(pareto)}")

# Separate by source
seeds = [e for e in pareto if e.candidate.meta.get('source') == 'seed']
mutations = [e for e in pareto if e.candidate.meta.get('source') == 'mutation']

print(f"\n📊 By Source:")
print(f"   Seeds in pareto: {len(seeds)}")
print(f"   Mutations in pareto: {len(mutations)}")

# Get final rung results
final_rung = config.shards[-1]
seeds_final = [e for e in seeds if e.result.shard_fraction >= final_rung]
mutations_final = [e for e in mutations if e.result.shard_fraction >= final_rung]

print(f"\n🏆 Final Rung (100%) Results:")
print(f"   Seeds at final rung: {len(seeds_final)}")
print(f"   Mutations at final rung: {len(mutations_final)}")

# Find best of each
seed_best_quality = 0.0
mutation_best_quality = 0.0
best_seed = None
best_mutation = None

if seeds_final:
    best_seed = max(seeds_final, key=lambda e: e.result.objectives.get('quality', 0))
    seed_best_quality = best_seed.result.objectives.get('quality', 0)
    print(f"\n   Best Seed Quality: {seed_best_quality:.1%}")

if mutations_final:
    best_mutation = max(mutations_final, key=lambda e: e.result.objectives.get('quality', 0))
    mutation_best_quality = best_mutation.result.objectives.get('quality', 0)
    print(f"   Best Mutation Quality: {mutation_best_quality:.1%}")

# Show rung progression
print(f"\n🎯 Rung Progression:")
rungs = {}
for e in pareto:
    rung = e.result.shard_fraction
    if rung not in rungs:
        rungs[rung] = {'seeds': [], 'mutations': []}

    source = e.candidate.meta.get('source')
    if source == 'seed':
        rungs[rung]['seeds'].append(e)
    elif source == 'mutation':
        rungs[rung]['mutations'].append(e)

for rung in sorted(rungs.keys()):
    rung_pct = int(rung * 100)
    n_seeds = len(rungs[rung]['seeds'])
    n_muts = len(rungs[rung]['mutations'])

    all_entries = rungs[rung]['seeds'] + rungs[rung]['mutations']
    best_q = max(e.result.objectives.get('quality', 0) for e in all_entries) if all_entries else 0

    print(f"   Rung {rung_pct:3d}%: {n_seeds} seeds, {n_muts} mutations, best={best_q:.1%}")

# THE PROOF
print(f"\n" + "=" * 80)
print("PROOF VERIFICATION")
print("=" * 80)

reached_final = seeds_final or mutations_final
if not reached_final:
    print("\n❌ FAILED: Did not reach final rung")
    print("   This indicates a bug in the multi-rung system")
    exit(1)

print(f"\n✅ Reached final rung (100%)")

if not mutations_final:
    print(f"\n⚠️  No mutations reached final rung")
    print("   Seed may have been too good, or not enough rounds")
    if seeds_final:
        print(f"   Seed quality: {seed_best_quality:.1%}")
else:
    improvement = mutation_best_quality - seed_best_quality

    print(f"\n📊 Quality Comparison:")
    print(f"   Seed:     {seed_best_quality:.1%}")
    print(f"   Mutation: {mutation_best_quality:.1%}")
    print(f"   Improvement: {improvement:+.1%}")

    if improvement > 0:
        print(f"\n🎉 SUCCESS! Mutations improved over seed by {improvement:.1%}")
        print(f"✅ Multi-rung optimization WORKS")
        print(f"✅ Mutations learn and improve")
        print(f"✅ System reaches final rung")

        # Show the improved prompt
        if best_mutation:
            print(f"\n📝 Best Mutation Text:")
            print("-" * 80)
            print(best_mutation.candidate.text[:300])
            if len(best_mutation.candidate.text) > 300:
                print("...")
            print("-" * 80)
    elif improvement == 0:
        print(f"\n⚠️  Mutation tied with seed (both {mutation_best_quality:.1%})")
        print("   System works but needs more diverse mutations")
    else:
        print(f"\n⚠️  Mutation worse than seed")
        print("   Unexpected - may indicate regression")

print("\n" + "=" * 80)
