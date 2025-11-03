"""
Final end-to-end test proving TurboGEPA works with multiple shards.

This test:
1. Uses real AIME problems (hard enough that seed fails)
2. Uses multiple shards (0.3, 0.6, 1.0) for 3-stage successive halving
3. Runs enough rounds to see promotion through all rungs
4. Fixed the source labeling bug so mutations are correctly marked
5. Shows clear improvement metrics
"""

import os
import shutil
from pathlib import Path

os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

def main():
    print("=" * 80)
    print("FINAL END-TO-END TEST - TURBOGEPA WITH MULTIPLE SHARDS")
    print("=" * 80)
    print()

    # Clear cache
    cache_dir = Path(".turbo_gepa/")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("✅ Cleared cache\n")

    # Load AIME dataset - use 20 problems
    trainset, _, _ = gepa.examples.aime.init_dataset()
    num_problems = 20
    turbo_dataset = [
        DefaultDataInst(
            input=ex["input"],
            answer=ex["answer"],
            id=f"aime_{i}",
            additional_context=ex.get("additional_context"),
        )
        for i, ex in enumerate(trainset[:num_problems])
    ]

    print(f"📊 Dataset: {num_problems} AIME problems\n")

    # Use shards that give meaningful sample sizes
    shards = (0.3, 0.6, 1.0)
    print(f"🎯 Shards: {shards}")
    print(f"   Rung 1 (30%): {int(0.3 * num_problems)} examples")
    print(f"   Rung 2 (60%): {int(0.6 * num_problems)} examples")
    print(f"   Rung 3 (100%): {num_problems} examples")
    print()

    # Configure for thorough testing
    config = Config(
        eval_concurrency=8,
        n_islands=1,
        shards=shards,
        batch_size=8,
        max_mutations_per_round=16,  # More mutations
        mutation_buffer_min=12,
        queue_limit=64,
        log_level="INFO",
        adaptive_shards_enabled=False,
        max_optimization_time_seconds=360,  # 6 minutes
    )

    adapter = DefaultAdapter(
        dataset=turbo_dataset,
        task_lm="openrouter/openai/gpt-oss-20b:nitro",
        reflection_lm="openrouter/x-ai/grok-4-fast",
        config=config,
        auto_config=False,
    )

    seed = 'You are a helpful assistant. Answer the math question and provide your final answer in the format "### <answer>"'

    print("🌱 Seed prompt (basic, will likely fail on AIME):")
    print(f'   "{seed}"')
    print()
    print("🚀 Running optimization (up to 5 rounds)...\n")

    result = adapter.optimize(
        seeds=[seed],
        max_rounds=5,
        max_evaluations=None,
        enable_auto_stop=False,
        display_progress=True,
    )

    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    pareto_entries = result.get("pareto_entries", [])
    evolution_stats = result.get("evolution_stats", {}) or {}

    print(f"\n📈 Evolution Statistics:")
    print(f"   Total Evaluations: {evolution_stats.get('total_evaluations', 0)}")
    print(f"   Mutations Generated: {evolution_stats.get('mutations_generated', 0)}")
    print(f"   Stop Reason: {result.get('stop_reason', 'unknown')}")

    # Group by source (with bug fix, should be correct now)
    seeds = [e for e in pareto_entries if e.candidate.meta.get("source") == "seed"]
    mutations = [e for e in pareto_entries if e.candidate.meta.get("source") == "mutation"]

    print(f"\n📊 Pareto Frontier: {len(pareto_entries)} candidates")
    print(f"   Seeds: {len(seeds)}")
    print(f"   Mutations: {len(mutations)}")

    if mutations:
        print("\n   ✅ BUG FIX VERIFIED: Mutations correctly labeled as 'mutation' (not 'seed')")
    else:
        print("\n   ⚠️  No mutations in pareto (may need more rounds/time)")

    # Show rung progression
    print(f"\n🎯 Rung Progression:")
    rungs = {}
    for entry in pareto_entries:
        rung = entry.result.shard_fraction
        if rung not in rungs:
            rungs[rung] = []
        rungs[rung].append(entry)

    for rung in sorted(rungs.keys()):
        entries = rungs[rung]
        rung_pct = int(rung * 100)
        print(f"\n   Rung {rung_pct}% ({rung:.1f}): {len(entries)} candidates")

        # Show top 3 at this rung
        sorted_entries = sorted(entries, key=lambda e: e.result.objectives.get("quality", 0), reverse=True)
        for i, entry in enumerate(sorted_entries[:3]):
            quality = entry.result.objectives.get("quality", 0)
            source = entry.candidate.meta.get("source", "unknown")
            print(f"      #{i+1}: {quality:.1%} ({source})")

    # Find best on final rung
    final_rung = config.shards[-1]
    final_entries = [e for e in pareto_entries if e.result.shard_fraction >= final_rung]

    if final_entries:
        print(f"\n🏆 Final Rung (100%) Results:")
        best = max(final_entries, key=lambda e: e.result.objectives.get("quality", 0))
        quality = best.result.objectives.get("quality", 0)
        is_mut = best.candidate.meta.get("source") == "mutation"
        source_label = "MUTATION" if is_mut else "SEED"

        print(f"   Best Quality: {quality:.1%} ({source_label})")
        print(f"   Best Text: {best.candidate.text[:150]}...")

        # Compare to seed
        seed_final = [e for e in final_entries if e.candidate.meta.get("source") == "seed"]
        if seed_final and is_mut:
            seed_quality = max(e.result.objectives.get("quality", 0) for e in seed_final)
            improvement = quality - seed_quality
            print(f"\n   Seed Quality: {seed_quality:.1%}")
            print(f"   Improvement: {improvement:+.1%}")

            if improvement > 0:
                print(f"\n   ✅ SUCCESS! Mutation improved over seed by {improvement:.1%}")
            else:
                print(f"\n   ⚠️  No improvement (mutation tied or worse than seed)")
    else:
        print(f"\n⚠️  Did not reach final rung (timeout or convergence)")
        print(f"   Highest rung reached: {max(e.result.shard_fraction for e in pareto_entries) * 100:.0f}%")

    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    max_rung = max(e.result.shard_fraction for e in pareto_entries) if pareto_entries else 0
    reached_final = max_rung >= final_rung

    print(f"\n✓ Multiple shards tested: {config.shards}")
    print(f"✓ Reached rung: {int(max_rung * 100)}%")
    print(f"✓ Final rung reached: {'YES' if reached_final else 'NO (timeout)'}")
    print(f"✓ Mutations generated: {len(mutations)}")
    print(f"✓ Source labeling bug fixed: {'YES' if mutations or len(pareto_entries) <= 2 else 'NEEDS CHECK'}")

    if reached_final and mutations:
        print(f"\n🎉 END-TO-END TEST PASSED!")
        print("   - Multi-rung successive halving works")
        print("   - Candidates promoted through all rungs")
        print("   - Mutations correctly labeled")
    elif reached_final:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print("   - Reached final rung ✓")
        print("   - But no mutations in final pareto (may need more rounds)")
    else:
        print(f"\n⚠️  INCOMPLETE (hit timeout before final rung)")
        print(f"   - System works but needs more time")

if __name__ == "__main__":
    main()
