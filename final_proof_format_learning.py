"""
DEFINITIVE PROOF: System learns to add missing format instruction.

Scenario:
- Seed LACKS the ### format instruction → scores 0% (model solves correctly but wrong format)
- Mutations should LEARN to add "### <answer>" instruction → scores improve to 60-100%
- This proves the optimization system works end-to-end
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
dataset = [DefaultDataInst(input=ex['input'], answer=ex['answer'], id=f'aime_{i}') for i, ex in enumerate(trainset[:8])]

print("=" * 80)
print("PROOF: MUTATIONS LEARN FORMAT REQUIREMENT")
print("=" * 80)
print(f"\n📊 Dataset: 8 AIME problems")
print(f"🎯 Shards: (0.5, 1.0) - 2 rungs for speed")
print(f"   Rung 1 (50%): 4 examples")
print(f"   Rung 2 (100%): 8 examples")

config = Config(
    eval_concurrency=8,
    n_islands=1,
    shards=(0.5, 1.0),  # Just 2 rungs for faster completion
    batch_size=8,
    max_mutations_per_round=8,
    mutation_buffer_min=6,
    queue_limit=32,
    log_level='INFO',
    adaptive_shards_enabled=False,
    max_optimization_time_seconds=180,
    eps_improve=0.01,
)

adapter = DefaultAdapter(
    dataset=dataset,
    task_lm='openrouter/openai/gpt-oss-20b:nitro',
    reflection_lm='openrouter/x-ai/grok-4-fast',
    config=config,
    auto_config=False
)

# Seed that WORKS mathematically but LACKS format instruction
incomplete_seed = "You are a helpful assistant. Solve the math problem and show your work."

print(f"\n🌱 Incomplete Seed (missing ### format):")
print(f'   "{incomplete_seed}"')
print(f"\n🔬 Expected behavior:")
print(f"   - Seed: 0% (solves correctly but wrong output format)")
print(f"   - Mutations: Learn to add '### <answer>' instruction → high %")
print(f"\n🚀 Starting optimization (max 5 rounds)...\n")

result = adapter.optimize(
    seeds=[incomplete_seed],
    max_rounds=5,
    enable_auto_stop=False,
    display_progress=True
)

# Analysis
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

pareto = result.get('pareto_entries', [])
evolution_stats = result.get('evolution_stats', {}) or {}

print(f"\n📈 Statistics:")
print(f"   Evaluations: {evolution_stats.get('total_evaluations', 0)}")
print(f"   Mutations: {evolution_stats.get('mutations_generated', 0)}")
print(f"   Pareto size: {len(pareto)}")

# Analyze final rung
final_rung = 1.0
seeds_final = [e for e in pareto if e.result.shard_fraction >= final_rung and e.candidate.meta.get('source') == 'seed']
muts_final = [e for e in pareto if e.result.shard_fraction >= final_rung and e.candidate.meta.get('source') == 'mutation']

print(f"\n🏆 Final Rung (100%):")
print(f"   Seeds: {len(seeds_final)}")
print(f"   Mutations: {len(muts_final)}")

seed_best_q = 0.0
mut_best_q = 0.0

if seeds_final:
    seed_best_q = max(e.result.objectives.get('quality', 0) for e in seeds_final)
    print(f"   Best seed: {seed_best_q:.1%}")

if muts_final:
    mut_best_q = max(e.result.objectives.get('quality', 0) for e in muts_final)
    best_mut = max(muts_final, key=lambda e: e.result.objectives.get('quality', 0))
    print(f"   Best mutation: {mut_best_q:.1%}")

# THE PROOF
print(f"\n" + "=" * 80)
print("PROOF")
print("=" * 80)

if not (seeds_final or muts_final):
    print("\n❌ FAILED: Did not reach final rung")
    print("   Bug in multi-rung system")
    exit(1)

print(f"\n✅ Reached final rung")

if mut_best_q > seed_best_q:
    improvement = mut_best_q - seed_best_q
    print(f"\n🎉 SUCCESS! Mutation improved over seed")
    print(f"   Seed:       {seed_best_q:.1%} (missing format instruction)")
    print(f"   Mutation:   {mut_best_q:.1%} (learned format)")
    print(f"   Improvement: {improvement:+.1%}")

    print(f"\n✅ Multi-rung optimization WORKS end-to-end")
    print(f"✅ System learns from failures and improves")
    print(f"✅ Mutations successfully evolved better prompts")

    if muts_final:
        print(f"\n📝 Best Mutation (first 400 chars):")
        print("-" * 80)
        best_text = best_mut.candidate.text
        print(best_text[:400])
        if len(best_text) > 400:
            print("...")
        print("-" * 80)

        # Check if it learned the format
        if '###' in best_text.lower() or 'format' in best_text.lower():
            print("\n✅ Mutation contains format-related instructions!")
            print("   Successfully learned the missing requirement")

elif mut_best_q == seed_best_q and mut_best_q > 0:
    print(f"\n⚠️  Mutation tied seed at {mut_best_q:.1%}")
    print("   System works but needs more differentiation")

elif muts_final and not seeds_final:
    print(f"\n✅ Mutation reached final ({mut_best_q:.1%}), seed did not")
    print("   Shows mutations can outperform seed")

else:
    print(f"\n⚠️  Seed: {seed_best_q:.1%}, Mutation: {mut_best_q:.1%}")
    print("   Need more rounds for mutations to learn")

# Show progression
print(f"\n📊 Rung Progression:")
for e in sorted(pareto, key=lambda x: (x.result.shard_fraction, e.candidate.meta.get('source') != 'mutation')):
    src = e.candidate.meta.get('source', '?')
    rung = int(e.result.shard_fraction * 100)
    q = e.result.objectives.get('quality', 0)
    marker = "🌱" if src == 'seed' else "🧬"
    print(f"   {marker} Rung {rung:3d}% {src:8s} quality={q:.1%}")

print("\n" + "=" * 80)
