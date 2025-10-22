"""
Benchmark GEPA vs TurboGEPA - MAXIMUM SPEED comparison.

Both use the same simple optimize() API.

Usage:
    export OPENAI_API_KEY=your_key
    python examples/benchmark_max_speed.py
"""

import os
import time

import gepa
import turbo_gepa


def main():
    """Run max speed benchmark."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("export OPENROUTER_API_KEY=your_key")
        return

    print("\n" + "=" * 80)
    print("🏁 MAXIMUM SPEED BENCHMARK: GEPA vs TurboGEPA")
    print("=" * 80)
    print("\nBoth optimizers use the same simple API:")
    print("  - GEPA: reflection_lm=gpt-4o")
    print(
        "  - TurboGEPA: eval_concurrency=32, reflection_lm=gpt-4o, shards=(0.3, 1.0)"
    )
    print("\nExpected time: ~10-30 minutes total")
    print("=" * 80)

    # Load AIME dataset
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:50]
    valset = valset[:10]

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    print(f"\nDataset: {len(trainset)} train, {len(valset)} val")
    print(f"Budget: 150 evaluations\n")

    # ========================================================================
    # GEPA Benchmark
    # ========================================================================
    print("\n" + "=" * 80)
    print("GEPA - MAXIMUM SPEED")
    print("=" * 80)

    # gepa_start = time.time()

    # gepa_result = gepa.optimize(
    #     seed_candidate=seed_prompt,
    #     trainset=trainset,
    #     valset=valset,
    #     task_lm="openai/gpt-4.1-mini",
    #     max_metric_calls=150,
    #     reflection_lm="openai/gpt-5",  # <-- LLM-based reflection
    #     display_progress_bar=True,
    # )

    # gepa_duration = time.time() - gepa_start

    # print(f"\n⏱️  GEPA Time: {gepa_duration:.2f}s ({gepa_duration/60:.2f} min)")
    # print(f"📊 GEPA Score: {gepa_result.best_score:.2%}")
    # print(f"🏆 GEPA Prompt: {gepa_result.best_candidate['system_prompt'][:100]}...")

    # # ========================================================================
    # # TurboGEPA Benchmark
    # # ========================================================================
    # print("\n\n" + "=" * 80)
    # print("TurboGEPA - MAXIMUM SPEED")
    # print("=" * 80)

    turbo_start = time.time()

    turbo_result = turbo_gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        task_lm="openrouter/x-ai/grok-4-fast",
        max_metric_calls=150,
        reflection_lm="openrouter/x-ai/grok-4-fast",  # <-- LLM-based reflection
        eval_concurrency=100,  # <-- MAX concurrency
        shards=(0.3, 1.0),  # <-- Aggressive successive halving
        display_progress=True,  # <-- Show progress
    )

    turbo_duration = time.time() - turbo_start

    print(f"\n⏱️  Turbo Time: {turbo_duration:.2f}s ({turbo_duration/60:.2f} min)")
    print(f"📊 Turbo Score: {turbo_result.best_score:.2%}")
    print(f"🏆 Turbo Prompt: {turbo_result.best_candidate['system_prompt'][:100]}...")

    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # speedup = gepa_duration / turbo_duration
    # time_saved = gepa_duration - turbo_duration

    # print(f"\n⏱️  TIMING")
    # print(f"   GEPA:       {gepa_duration:.2f}s ({gepa_duration/60:.2f} min)")
    print(f"   TurboGEPA: {turbo_duration:.2f}s ({turbo_duration/60:.2f} min)")
    # print(f"   🚀 SPEEDUP: {speedup:.2f}× FASTER")
    # print(f"   ⏰ SAVED:   {time_saved:.2f}s ({time_saved/60:.2f} min)")

    print(f"\n📊 QUALITY")
    # print(f"   GEPA:       {gepa_result.best_score:.2%}")
    print(f"   TurboGEPA: {turbo_result.best_score:.2%}")
    # print(f"   Difference: {turbo_result.best_score - gepa_result.best_score:+.2%}")

    print("\n" + "=" * 80)
    # if speedup >= 5:
    #     print("✅ SIGNIFICANT speedup - TurboGEPA's async architecture wins!")
    # elif speedup >= 2:
    #     print("✅ Good speedup - TurboGEPA is notably faster")
    # else:
    #     print("⚠️  Similar performance")

    print("=" * 80)


if __name__ == "__main__":
    main()
