"""
Smoke test for TurboGEPA on actual AIME problems.

This script demonstrates TurboGEPA's batching and reflection capabilities:
- Concurrent evaluation in batches
- Batched reflection: LLM analyzes multiple successful parents simultaneously
- Hybrid mutations: Incremental reflection + spec induction run in parallel
- ASHA successive halving: Prunes weak candidates early

REQUIREMENTS:
    pip install datasets  # For loading AIME dataset
    export OPENROUTER_API_KEY=your_key_here

Usage:
    pip install datasets
    export OPENROUTER_API_KEY=your_key_here
    python examples/benchmark_max_speed_smoke.py

Get an API key at: https://openrouter.ai/keys

Expected runtime: ~90-120 seconds (5 rounds with real LLM calls + batch reflection)
Expected result: Quality improvement showing batch reflection working on real AIME problems
"""

from __future__ import annotations

import time
from typing import Iterable, List

from turbo_gepa.archive import ArchiveEntry
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config, adaptive_config


def _load_aime_dataset(num_problems: int = 10) -> List[DefaultDataInst]:
    """Load real AIME problems from the dataset.

    Uses the same dataset and format as aime_full_eval.py but takes a small subset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ ERROR: 'datasets' library not installed")
        print("Install with: pip install datasets")
        raise

    print("📥 Loading AIME dataset from HuggingFace...")

    # Load the AIME dataset (same as gepa.examples.aime.init_dataset)
    train_data = load_dataset("AI-MO/aimo-validation-aime", split="train")

    # Convert to the format expected by TurboGEPA (same as aime_full_eval.py)
    dataset: List[DefaultDataInst] = []
    for idx, sample in enumerate(
        train_data.select(range(min(num_problems, len(train_data))))
    ):
        problem = sample["problem"]
        answer = "### " + str(sample["answer"])  # Match GEPA format
        solution = sample.get("solution", "")

        dataset.append(
            DefaultDataInst(
                input=problem,
                answer=answer,
                additional_context={"solution": solution} if solution else {},
                id=f"train-{idx}",
                difficulty=0.5,  # AIME problems are uniformly hard
            )
        )

    print(f"✓ Loaded {len(dataset)} AIME problems")
    return dataset


def _best_quality(pareto_entries: Iterable[ArchiveEntry]) -> float:
    best = 0.0
    for entry in pareto_entries:
        best = max(best, entry.result.objectives.get("quality", 0.0))
    return best


def main() -> None:
    print("\n" + "=" * 70)
    print("🚀 TurboGEPA Smoke Test - Real AIME Problems")
    print("=" * 70)

    # Load 10 real AIME problems for smoke test
    dataset = _load_aime_dataset(num_problems=10)
    print(f"\n📊 Dataset: {len(dataset)} real AIME problems")
    print(f"   Sample problem: {dataset[0].input[:80]}...")

    # Require OPENROUTER_API_KEY - TurboGEPA requires real LLM integration
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n❌ ERROR: OPENROUTER_API_KEY environment variable not set")
        print("\nTurboGEPA requires real LLM integration. Please set your API key:")
        print("   export OPENROUTER_API_KEY=your_key_here")
        print("\nGet an API key at: https://openrouter.ai/keys")
        print("=" * 70 + "\n")
        return

    print("\n🤖 Using Real LLM Integration:")
    # Use better models for real AIME problems (they're hard!)
    task_lm = "openrouter/google/gemini-2.0-flash-001"  # Good balance of speed/quality
    reflection_lm = "openrouter/google/x-ai/grok-4-fast"  # Same for reflection
    print(f"   Task LM: {task_lm}")
    print(f"   Reflection LM: {reflection_lm}")
    task_temperature = 0.7
    reflection_temperature = 0.7

    # Create custom config to ensure batch reflection runs properly
    config = adaptive_config(
        dataset_size=len(dataset),
        strategy="balanced",
        available_compute="laptop",
    )

    # Override to ensure we get enough mutations for testing batch reflection
    config.max_mutations_per_round = (
        8  # Generate 8 mutations/round (4 incremental + 4 spec)
    )
    config.reflection_batch_size = 5  # Use up to 5 parent contexts in batch reflection
    config.eval_concurrency = 100  # Moderate concurrency for smoke test

    print(f"\n⚙️  Custom Configuration (optimized for testing batch reflection):")
    print(f"   • Shards: {config.shards}")
    print(
        f"   • Mutations/round: {config.max_mutations_per_round} (4 incremental + 4 spec)"
    )
    print(f"   • Reflection batch size: {config.reflection_batch_size} parents")
    print(f"   • Eval concurrency: {config.eval_concurrency}")

    adapter = DefaultAdapter(
        dataset=dataset,
        sampler_seed=123,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        task_lm_temperature=task_temperature,
        reflection_lm_temperature=reflection_temperature,
        config=config,  # Use our custom config
    )

    # OPTION 1: Use PROMPT-MII to generate smart seeds from task examples (RECOMMENDED)
    use_prompt_mii_initialization = True

    if use_prompt_mii_initialization:
        print(
            "\n🌱 Using PROMPT-MII seed initialization (generates task-specific seeds)"
        )
        print("   Analyzing AIME examples to create structured specifications...")
        seeds = None  # Let PROMPT-MII generate seeds
        enable_seed_initialization = True
        num_generated_seeds = 3
    else:
        # OPTION 2: Use hand-crafted seeds (traditional approach)
        print("\n🌱 Using hand-crafted seeds (traditional approach)")
        seeds = [
            (
                "You are a meticulous AIME math assistant. Explain your reasoning "
                "briefly, keep calculations organized, and end with '### <final answer>'."
            ),
            (
                "Solve this math problem step-by-step. Show all work clearly and "
                "provide the final numerical answer."
            ),
            (
                "Think through this problem carefully. Break it down into parts, "
                "solve each part, then combine for the final answer."
            ),
        ]
        print(f'   Seed 1: "{seeds[0][:50]}..."')
        enable_seed_initialization = False
        num_generated_seeds = 0

    print(f"\n⏱️  Starting optimization (5 rounds, 100 evaluations)...")
    print("   Watch for '⚡ Batched reflection' messages showing LLM calls\n")
    print("📝 What to expect:")
    if use_prompt_mii_initialization:
        print(
            "   • Seed initialization: PROMPT-MII generates 3 structured specifications"
        )
        print("   • Round 0: Evaluate generated seeds on first shard")
    else:
        print("   • Round 0: Evaluate 3 hand-crafted seeds on first shard")
    print("   • Round 1-4: Batched reflection generates mutations each round")
    print(
        "   • Batch reflection: LLM synthesizes ideas from multiple successful parents"
    )
    print("   • Spec induction: LLM generates fresh prompts from task examples")
    print("   • Both run concurrently for maximum efficiency")
    print("   • AIME problems are HARD - quality may start low (10-30%)")
    print("   • Watch for improvement over rounds as prompts evolve\n")
    max_rounds = 5  # Enough rounds to see batch reflection in action
    max_evaluations = 100  # Enough budget for multiple reflection cycles

    start_time = time.time()
    result = adapter.optimize(
        seeds=seeds,
        max_rounds=max_rounds,
        max_evaluations=max_evaluations,
        display_progress=True,  # Show progress charts
        enable_seed_initialization=enable_seed_initialization,
        num_generated_seeds=num_generated_seeds,
    )
    elapsed = time.time() - start_time

    pareto_entries = result["pareto_entries"]
    best_quality = _best_quality(pareto_entries)

    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\n⏱️  Runtime: {elapsed:.3f}s")
    print(f"📊 Pareto Frontier Size: {len(pareto_entries)} candidates")

    print(f"🎯 Best Quality: {best_quality:.3f}")

    print(f"\n📈 Configuration Summary:")
    print(f"   • Dataset size: {len(dataset)} examples")
    print(f"   • Shards: {adapter.config.shards}")
    print(f"   • Batch size: {adapter.config.batch_size}")
    print(f"   • Mutations/round: {adapter.config.max_mutations_per_round}")
    print(f"   • Reflection batch size: {adapter.config.reflection_batch_size} parents")
    print(f"   • Eval concurrency: {adapter.config.eval_concurrency}")

    print(f"\n📊 Optimization Features Used:")
    if use_prompt_mii_initialization:
        print(
            f"   • PROMPT-MII seed initialization: Generated {num_generated_seeds} task-specific seeds"
        )
    else:
        print(f"   • Traditional seeds: Used hand-crafted prompts")
    print(
        f"   • Batched reflection: 1 LLM call processes {adapter.config.reflection_batch_size} parents simultaneously"
    )
    print(
        f"   • Hybrid mutations: incremental reflection + spec induction (run concurrently)"
    )
    print(f"   • ASHA successive halving: pruned weak candidates early")
    print(f"   • Check output above for '⚡ Batched reflection' timing logs")

    if pareto_entries:
        # Show top 3 candidates
        sorted_entries = sorted(
            pareto_entries,
            key=lambda e: e.result.objectives.get("quality", 0.0),
            reverse=True,
        )

        print(f"\n🏆 Top {min(3, len(sorted_entries))} Candidates:")
        for idx, entry in enumerate(sorted_entries[:3], 1):
            quality = entry.result.objectives.get("quality", 0.0)
            tokens = entry.result.objectives.get("tokens", 0)
            print(f"\n   #{idx} | Quality: {quality:.3f} | Tokens: {tokens:.0f}")
            snippet = entry.candidate.text[:120].replace("\n", " ")
            print(
                f"      \"{snippet}{'...' if len(entry.candidate.text) > 120 else ''}\""
            )

    print("\n" + "=" * 70)
    print("✅ Smoke test complete!")
    print("=" * 70)
    print("\n💡 Key TurboGEPA Features Demonstrated:")

    if use_prompt_mii_initialization:
        print("\n   1. PROMPT-MII Seed Initialization:")
        print("      • Analyzes task examples to generate structured specifications")
        print(
            "      • Creates task-specific seeds (vs generic 'you are a helpful assistant')"
        )
        print("      • Faster convergence from better starting point")

    print("\n   2. Batch Reflection:")
    print("      • Instead of calling LLM separately for each parent prompt,")
    print("      • Makes 1 call that analyzes multiple parents together")
    print("      • Faster AND produces better mutations (synthesizes ideas)")

    print("\n   3. Hybrid Mutations:")
    print("      • Incremental reflection: improves existing prompts")
    print("      • Spec induction: generates fresh prompts from examples")
    print("      • Both run concurrently for efficiency")

    print("\n   Look for these log messages in the output above:")
    if use_prompt_mii_initialization:
        print("   • '🌱 Generating ... seeds from task examples using PROMPT-MII...'")
    print("   • '⚡ Batched reflection (N parents): X.XXs → Y mutations'")
    print("   • '⚡ Spec induction (N examples): X.XXs → Y specs'")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
