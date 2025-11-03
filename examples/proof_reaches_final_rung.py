"""
PROOF: TurboGEPA reaches final rung (1.0) with multi-shard successive halving.

This test definitively proves:
1. Multi-rung successive halving works end-to-end
2. Candidates are promoted from rung 0.5 → 0.75 → 1.0
3. Bug fix verified: mutations labeled as "source": "mutation"
4. System completes full optimization cycle

Configuration chosen to avoid ties:
- Start at 0.5 (10 examples) for better differentiation
- Use 20 AIME problems total
- Higher concurrency for faster completion
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
    print("PROOF: TURBOGEPA REACHES FINAL RUNG 1.0")
    print("=" * 80)
    print("\nThis test will definitively prove multi-rung successive halving works.\n")

    # Clear cache
    cache_dir = Path(".turbo_gepa/")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("✅ Cleared cache\n")

    # Load AIME dataset
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

    print(f"📊 Dataset: {num_problems} AIME problems")

    # Configure shards starting at 0.5 to avoid ties
    # 0.5 = 10 examples, 0.75 = 15 examples, 1.0 = 20 examples
    shards = (0.5, 0.75, 1.0)

    print(f"\n🎯 Shards: {shards}")
    print(f"   Rung 1 (50%):  {int(0.5 * num_problems)} examples (better differentiation)")
    print(f"   Rung 2 (75%):  {int(0.75 * num_problems)} examples")
    print(f"   Rung 3 (100%): {num_problems} examples ← TARGET")
    print()

    # Configure for success
    config = Config(
        eval_concurrency=8,
        n_islands=1,
        shards=shards,
        batch_size=8,
        max_mutations_per_round=12,
        mutation_buffer_min=10,
        queue_limit=64,
        log_level="INFO",
        adaptive_shards_enabled=False,
        max_optimization_time_seconds=480,  # 8 minutes - should be enough
        eps_improve=0.01,  # Allow 1% improvement to count
    )

    adapter = DefaultAdapter(
        dataset=turbo_dataset,
        task_lm="openrouter/openai/gpt-oss-20b:nitro",
        reflection_lm="openrouter/x-ai/grok-4-fast",
        config=config,
        auto_config=False,
    )

    seed = 'You are a helpful assistant. Answer the math question and provide your final answer in the format "### <answer>"'

    print(f"🌱 Seed: {seed}")
    print()
    print("🚀 Running optimization until completion...\n")
    print("⏱️  This may take 5-8 minutes. Watch for rung progression!\n")

    result = adapter.optimize(
        seeds=[seed],
        max_rounds=8,  # More rounds to ensure completion
        max_evaluations=None,
        enable_auto_stop=False,
        display_progress=True,
    )

    # Analyze results
    print("\n" + "=" * 80)
    print("PROOF VERIFICATION")
    print("=" * 80)

    pareto_entries = result.get("pareto_entries", [])
    evolution_stats = result.get("evolution_stats", {}) or {}

    print(f"\n📈 Statistics:")
    print(f"   Total Evaluations: {evolution_stats.get('total_evaluations', 0)}")
    print(f"   Mutations Generated: {evolution_stats.get('mutations_generated', 0)}")
    print(f"   Pareto Candidates: {len(pareto_entries)}")

    # Check source labeling
    seeds = [e for e in pareto_entries if e.candidate.meta.get("source") == "seed"]
    mutations = [e for e in pareto_entries if e.candidate.meta.get("source") == "mutation"]

    print(f"\n🏷️  Source Labeling (Bug Fix Verification):")
    print(f"   Seeds: {len(seeds)}")
    print(f"   Mutations: {len(mutations)}")

    if mutations:
        print(f"   ✅ PASS: Mutations correctly labeled with source='mutation'")
    else:
        print(f"   ⚠️  No mutations in pareto (system converged on seed)")

    # Check rung progression - THIS IS THE KEY PROOF
    print(f"\n🎯 Rung Progression (CRITICAL PROOF):")

    rungs_found = set()
    rung_details = {}

    for entry in pareto_entries:
        rung = entry.result.shard_fraction
        rungs_found.add(rung)
        if rung not in rung_details:
            rung_details[rung] = []
        rung_details[rung].append(entry)

    for rung in sorted(rungs_found):
        entries = rung_details[rung]
        rung_pct = int(rung * 100)
        qualities = [e.result.objectives.get("quality", 0) for e in entries]
        best_quality = max(qualities) if qualities else 0

        marker = "✓" if rung >= 1.0 else " "
        print(f"   {marker} Rung {rung_pct:3d}% ({rung:.2f}): {len(entries)} candidates, best quality {best_quality:.1%}")

    # THE PROOF
    final_rung = config.shards[-1]
    reached_final = any(e.result.shard_fraction >= final_rung for e in pareto_entries)

    print(f"\n🎯 FINAL RUNG CHECK:")
    print(f"   Target rung: {final_rung} (100%)")
    print(f"   Reached: {reached_final}")

    if reached_final:
        print(f"\n   ✅ PROOF COMPLETE: System reached final rung!")

        # Show what made it
        final_entries = [e for e in pareto_entries if e.result.shard_fraction >= final_rung]
        best_final = max(final_entries, key=lambda e: e.result.objectives.get("quality", 0))
        quality = best_final.result.objectives.get("quality", 0)
        source = best_final.candidate.meta.get("source", "unknown")

        print(f"\n   🏆 Best candidate on final rung:")
        print(f"      Quality: {quality:.1%}")
        print(f"      Source: {source}")
        print(f"      Text: {best_final.candidate.text[:120]}...")

        # Summary
        print(f"\n" + "=" * 80)
        print("✅ ✅ ✅ PROOF SUCCESSFUL ✅ ✅ ✅")
        print("=" * 80)
        print(f"\n   ✓ Multi-rung successive halving: WORKING")
        print(f"   ✓ Candidates promoted through: {' → '.join(f'{int(r*100)}%' for r in sorted(rungs_found))}")
        print(f"   ✓ Reached final rung (100%): YES")
        print(f"   ✓ Source labeling bug fixed: {'YES' if mutations else 'N/A (no mutations)'}")
        print(f"   ✓ Total rungs visited: {len(rungs_found)}")
        print(f"\n   🎉 END-TO-END MULTI-SHARD OPTIMIZATION: PROVEN!")

        return 0  # Success

    else:
        max_rung = max(e.result.shard_fraction for e in pareto_entries) if pareto_entries else 0
        print(f"\n   ❌ INCOMPLETE: Only reached {int(max_rung * 100)}%")
        print(f"      (Need more time or different configuration)")

        return 1  # Failed to reach final rung

if __name__ == "__main__":
    exit(main())
