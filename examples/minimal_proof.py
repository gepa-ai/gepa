"""
MINIMAL PROOF: Reaches final rung with clean state.

This is the simplest possible test that MUST complete:
- Only 10 AIME problems
- Only 2 shards: 0.6 → 1.0 (6 examples → 10 examples)
- Cache wiped at start
- Forces completion in max 3 rounds
"""

import os
import shutil
from pathlib import Path
import time

# CRITICAL: Wipe cache BEFORE any imports
cache_dir = Path(".turbo_gepa/")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("🗑️  Cache wiped clean")
    time.sleep(0.5)  # Ensure filesystem sync

os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


def main():
    print("\n" + "=" * 80)
    print("MINIMAL PROOF: 2-RUNG SUCCESSIVE HALVING")
    print("=" * 80)

    # Load minimal dataset
    trainset, _, _ = gepa.examples.aime.init_dataset()
    num_problems = 10

    turbo_dataset = [
        DefaultDataInst(
            input=ex["input"],
            answer=ex["answer"],
            id=f"aime_{i}",
            additional_context=ex.get("additional_context"),
        )
        for i, ex in enumerate(trainset[:num_problems])
    ]

    print(f"\n📊 Dataset: {num_problems} AIME problems")

    # Simplest possible multi-shard config
    shards = (0.6, 1.0)  # Just 2 rungs

    print(f"\n🎯 Shards: {shards}")
    print(f"   Rung 1 (60%):  {int(0.6 * num_problems)} examples")
    print(f"   Rung 2 (100%): {num_problems} examples ← MUST REACH THIS")

    config = Config(
        eval_concurrency=8,
        n_islands=1,
        shards=shards,
        batch_size=8,
        max_mutations_per_round=8,
        mutation_buffer_min=6,
        queue_limit=32,
        log_level="INFO",
        adaptive_shards_enabled=False,
        max_optimization_time_seconds=300,  # 5 minutes max
        eps_improve=0.01,
    )

    adapter = DefaultAdapter(
        dataset=turbo_dataset,
        task_lm="openrouter/openai/gpt-oss-20b:nitro",
        reflection_lm="openrouter/x-ai/grok-4-fast",
        config=config,
        auto_config=False,
    )

    seed = 'You are a helpful assistant. Answer the math question and provide your final answer in the format "### <answer>"'

    print(f"\n🌱 Seed: {seed}")
    print(f"\n🚀 Starting optimization (max 3 rounds)...")
    print(f"⏱️  Should complete in ~2-4 minutes\n")

    start_time = time.time()

    result = adapter.optimize(
        seeds=[seed],
        max_rounds=3,  # Very limited rounds
        max_evaluations=None,
        enable_auto_stop=False,
        display_progress=True,
    )

    elapsed = time.time() - start_time

    # VERIFICATION
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    pareto_entries = result.get("pareto_entries", [])
    evolution_stats = result.get("evolution_stats", {}) or {}

    print(f"\n⏱️  Completed in {elapsed:.1f} seconds")
    print(f"📊 Total Evaluations: {evolution_stats.get('total_evaluations', 0)}")
    print(f"📊 Mutations Generated: {evolution_stats.get('mutations_generated', 0)}")
    print(f"📊 Pareto Candidates: {len(pareto_entries)}")

    # Check rungs reached
    rungs_reached = sorted(set(e.result.shard_fraction for e in pareto_entries))

    print(f"\n🎯 Rungs Reached: {[f'{int(r*100)}%' for r in rungs_reached]}")

    for rung in rungs_reached:
        entries_at_rung = [e for e in pareto_entries if e.result.shard_fraction == rung]
        qualities = [e.result.objectives.get("quality", 0) for e in entries_at_rung]
        best_q = max(qualities) if qualities else 0
        print(f"   Rung {int(rung*100):3d}%: {len(entries_at_rung)} candidates, best={best_q:.1%}")

    # THE CRITICAL CHECK
    target_rung = 1.0
    reached_target = any(e.result.shard_fraction >= target_rung for e in pareto_entries)

    print(f"\n" + "=" * 80)
    if reached_target:
        print("✅ ✅ ✅ SUCCESS ✅ ✅ ✅")
        print("=" * 80)
        print(f"\n✓ Reached final rung (100%): YES")
        print(f"✓ Multi-shard successive halving: WORKING")
        print(f"✓ Rungs traversed: {' → '.join(f'{int(r*100)}%' for r in rungs_reached)}")

        # Show best at final rung
        final_entries = [e for e in pareto_entries if e.result.shard_fraction >= target_rung]
        best = max(final_entries, key=lambda e: e.result.objectives.get("quality", 0))
        quality = best.result.objectives.get("quality", 0)
        source = best.candidate.meta.get("source", "unknown")

        print(f"\n🏆 Best on final rung:")
        print(f"   Quality: {quality:.1%}")
        print(f"   Source: {source}")
        print(f"   Text: {best.candidate.text[:100]}...")

        # Check bug fix
        mutations = [e for e in pareto_entries if e.candidate.meta.get("source") == "mutation"]
        if mutations:
            print(f"\n✓ Bug fix verified: {len(mutations)} mutations correctly labeled")

        print(f"\n🎉 PROOF COMPLETE: End-to-end multi-shard optimization WORKS!")
        return 0

    else:
        print("❌ ❌ ❌ FAILED ❌ ❌ ❌")
        print("=" * 80)
        max_rung = max(rungs_reached) if rungs_reached else 0
        print(f"\n✗ Only reached {int(max_rung*100)}%, not 100%")
        print(f"✗ This indicates a real bug in the system")

        # Debug info
        print(f"\n🔍 Debug Info:")
        print(f"   Rounds completed: {result.get('round', 'unknown')}")
        print(f"   Stop reason: {result.get('stop_reason', 'unknown')}")

        # Check if candidates are stuck
        if pareto_entries:
            print(f"\n   Candidates in pareto:")
            for i, entry in enumerate(pareto_entries[:5]):
                rung = entry.result.shard_fraction
                quality = entry.result.objectives.get("quality", 0)
                source = entry.candidate.meta.get("source", "?")
                print(f"     {i+1}. Rung {int(rung*100)}%, Q={quality:.1%}, source={source}")

        return 1


if __name__ == "__main__":
    exit(main())
