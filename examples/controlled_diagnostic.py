"""
Controlled diagnostic with simple problems where we KNOW the seed should fail
and mutations should improve performance.
"""

import os
import shutil
from pathlib import Path

os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Simple arithmetic problems where basic prompt will fail
# but a more specific prompt will succeed
simple_problems = [
    {
        "input": "What is 15 + 27? Provide ONLY the number.",
        "answer": "42",
        "id": "add_1"
    },
    {
        "input": "What is 100 - 37? Provide ONLY the number.",
        "answer": "63",
        "id": "sub_1"
    },
    {
        "input": "What is 8 × 9? Provide ONLY the number.",
        "answer": "72",
        "id": "mul_1"
    },
    {
        "input": "What is 144 ÷ 12? Provide ONLY the number.",
        "answer": "12",
        "id": "div_1"
    },
    {
        "input": "What is 25 + 38? Provide ONLY the number.",
        "answer": "63",
        "id": "add_2"
    },
    {
        "input": "What is 200 - 85? Provide ONLY the number.",
        "answer": "115",
        "id": "sub_2"
    },
    {
        "input": "What is 7 × 8? Provide ONLY the number.",
        "answer": "56",
        "id": "mul_2"
    },
    {
        "input": "What is 96 ÷ 8? Provide ONLY the number.",
        "answer": "12",
        "id": "div_2"
    },
    {
        "input": "What is 33 + 19? Provide ONLY the number.",
        "answer": "52",
        "id": "add_3"
    },
    {
        "input": "What is 150 - 67? Provide ONLY the number.",
        "answer": "83",
        "id": "sub_3"
    },
]

def main():
    print("=" * 80)
    print("CONTROLLED DIAGNOSTIC - Simple Arithmetic")
    print("=" * 80)
    print(f"\n📊 Using {len(simple_problems)} simple arithmetic problems\n")

    # Clear cache
    cache_dir = Path(".turbo_gepa/")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("✅ Cleared cache\n")

    # Convert to DefaultDataInst format
    dataset = [
        DefaultDataInst(
            input=p["input"],
            answer=p["answer"],
            id=p["id"],
            additional_context=None,
        )
        for p in simple_problems
    ]

    # Config with multiple shards
    config = Config(
        eval_concurrency=4,
        n_islands=1,
        shards=(0.2, 0.5, 1.0),  # More reasonable shards
        batch_size=4,
        max_mutations_per_round=8,
        mutation_buffer_min=8,
        queue_limit=64,
        log_level="INFO",
        adaptive_shards_enabled=False,
        max_optimization_time_seconds=120,
    )

    print("Configuration:")
    print(f"  eval_concurrency: {config.eval_concurrency}")
    print(f"  shards: {config.shards}")
    print(f"  max_mutations_per_round: {config.max_mutations_per_round}")
    print(f"  timeout: 120s")
    print()

    # Create adapter - use a weak model for task, strong for reflection
    task_lm = "openrouter/meta-llama/llama-3-8b-instruct"  # Intentionally weak 8B model
    reflection_lm = "openrouter/x-ai/grok-4-fast"  # Strong for mutations

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
    )

    # Use a DELIBERATELY BAD seed that will get things wrong
    # The LLM should learn to be more precise with formatting
    bad_seed = "Answer the question. Be helpful and explain your reasoning."

    print("🌱 Bad Seed Prompt (should fail on 'ONLY the number' requirement):")
    print(f"   '{bad_seed}'")
    print()
    print("🚀 Starting optimization...\n")

    result = adapter.optimize(
        seeds=[bad_seed],
        max_rounds=5,
        max_evaluations=None,
        enable_auto_stop=False,  # Disable auto-stop to force all rounds
        display_progress=True,
    )

    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    evolution_stats = result.get("evolution_stats", {}) or {}
    pareto_entries = result.get("pareto_entries", [])

    print(f"\nEvolution Stats:")
    print(f"  Total Evaluations: {evolution_stats.get('total_evaluations', 0)}")
    print(f"  Mutations Generated: {evolution_stats.get('mutations_generated', 0)}")
    print(f"  Stop Reason: {result.get('stop_reason', 'unknown')}")

    # Find seed and best
    seed_quality = None
    best_quality = None
    best_candidate = None

    for entry in pareto_entries:
        quality = entry.result.objectives.get(config.promote_objective, 0.0)
        is_seed = (isinstance(entry.candidate.meta, dict) and
                  entry.candidate.meta.get("source") == "seed")

        # Track seed quality on final rung
        if is_seed and entry.result.shard_fraction >= config.shards[-1]:
            if seed_quality is None:
                seed_quality = quality

        # Track best on final rung
        if entry.result.shard_fraction >= config.shards[-1]:
            if best_quality is None or quality > best_quality:
                best_quality = quality
                best_candidate = entry.candidate

    print(f"\n📊 Quality Comparison (on final rung):")
    if seed_quality is not None:
        print(f"  Seed Quality: {seed_quality:.1%}")
    if best_quality is not None:
        print(f"  Best Quality: {best_quality:.1%}")
    if seed_quality is not None and best_quality is not None:
        improvement = best_quality - seed_quality
        print(f"  Improvement: {improvement:+.1%}")

        if improvement > 0:
            print(f"\n✅ SUCCESS! Optimization improved quality by {improvement:.1%}")
        elif improvement < 0:
            print(f"\n⚠️  REGRESSION! Quality decreased by {abs(improvement):.1%}")
        else:
            print(f"\n⚠️  NO IMPROVEMENT! This suggests a potential bug.")

    # Show all pareto candidates
    if pareto_entries:
        print(f"\n📊 Pareto Frontier ({len(pareto_entries)} candidates):")
        print("-" * 80)

        # Group by fingerprint to see unique candidates
        by_fp = {}
        for entry in pareto_entries:
            fp = entry.candidate.fingerprint
            if fp not in by_fp:
                by_fp[fp] = []
            by_fp[fp].append(entry)

        print(f"  {len(by_fp)} unique candidates across rungs")

        for i, (fp, entries) in enumerate(sorted(by_fp.items(),
                                                  key=lambda x: max(e.result.objectives.get(config.promote_objective, 0) for e in x[1]),
                                                  reverse=True)):
            # Show best rung for this candidate
            best_entry = max(entries, key=lambda e: e.result.shard_fraction)
            quality = best_entry.result.objectives.get(config.promote_objective, 0.0)
            shard = best_entry.result.shard_fraction
            is_seed = (isinstance(best_entry.candidate.meta, dict) and
                      best_entry.candidate.meta.get("source") == "seed")

            rungs_reached = sorted([int(e.result.shard_fraction * 100) for e in entries])

            print(f"\n  #{i+1}: {'[SEED]' if is_seed else '[MUT] '} Quality={quality:.1%} @ Rung {int(shard*100)}%")
            print(f"       Rungs: {rungs_reached}")
            print(f"       Text: {best_entry.candidate.text[:100]}...")

    # Show best candidate
    if best_candidate:
        print(f"\n📝 Best Candidate Text:")
        print("-" * 80)
        print(best_candidate.text)
        print("-" * 80)

    # Show seed for comparison
    seed_candidates = [e for e in pareto_entries
                      if isinstance(e.candidate.meta, dict) and e.candidate.meta.get("source") == "seed"]
    if seed_candidates:
        print(f"\n📝 Seed Candidate Text (for comparison):")
        print("-" * 80)
        print(seed_candidates[0].candidate.text)
        print("-" * 80)

if __name__ == "__main__":
    main()
