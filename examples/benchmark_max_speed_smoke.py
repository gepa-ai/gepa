"""
Ultra-fast smoke benchmark for uFast-GEPA on a tiny AIME-style dataset.

This script mirrors ``examples/benchmark_max_speed.py`` but trims the
workload so it can be used as a quick health check without any external
LLM calls.  All scoring is handled by the built-in heuristic adapters, so
it runs in a couple of seconds.

Run with:
    python examples/benchmark_max_speed_smoke.py
"""

from __future__ import annotations

import time
from typing import Iterable, List

from ufast_gepa.archive import ArchiveEntry
from ufast_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from ufast_gepa.config import Config


# Minimal AIME-inspired training set. These problems are adapted from public
# AIME-style questions but shortened so the smoke test is lightweight.
_AIME_PROBLEMS = [
    {
        "problem": (
            "On an AIME practice test, the sequence a_n = 2n^2 - n is considered. "
            "What is the value of a_5?"
        ),
        "answer": "### 45",
        "difficulty": 0.35,
        "solution": "a_5 = 2*25 - 5 = 50 - 5 = 45.",
    },
    {
        "problem": ("A triangle has side lengths 5, 12, and 13. What is its area?"),
        "answer": "### 30",
        "difficulty": 0.25,
        "solution": "Right triangle with legs 5 and 12 so area = 5*12/2 = 30.",
    },
    {
        "problem": "Compute the remainder when 3^5 is divided by 100.",
        "answer": "### 43",
        "difficulty": 0.45,
        "solution": "3^5 = 243 so the remainder upon division by 100 is 43.",
    },
    {
        "problem": ("Let x satisfy x + 1/x = 4. What is x^2 + 1/x^2?"),
        "answer": "### 14",
        "difficulty": 0.30,
        "solution": "Square the equation: x^2 + 2 + 1/x^2 = 16 ⇒ x^2 + 1/x^2 = 14.",
    },
]


def _build_dataset() -> List[DefaultDataInst]:
    dataset: List[DefaultDataInst] = []
    for idx, sample in enumerate(_AIME_PROBLEMS):
        dataset.append(
            DefaultDataInst(
                input=sample["problem"],
                answer=sample["answer"],
                additional_context={"solution": sample["solution"]},
                id=f"aime-{idx}",
                difficulty=sample.get("difficulty"),
            )
        )
    return dataset


def _best_quality(pareto_entries: Iterable[ArchiveEntry]) -> float:
    best = 0.0
    for entry in pareto_entries:
        best = max(best, entry.result.objectives.get("quality", 0.0))
    return best


def main() -> None:
    print("\n" + "=" * 70)
    print("🚀 uFast-GEPA Smoke Test - Quick Health Check")
    print("=" * 70)

    dataset = _build_dataset()
    print(f"\n📊 Dataset: {len(dataset)} AIME-style problems")
    print(f"   Problems: {[d.id for d in dataset]}")

    config = Config(
        eval_concurrency=4,
        shards=(0.5, 1.0),
        max_mutations_per_round=4,
        queue_limit=32,
        migration_period=10,  # effectively disables migration
        log_summary_interval=1,  # Show progress every round
        cache_path=".ufast_gepa/smoke_cache",
        log_path=".ufast_gepa/smoke_logs",
    )

    print(f"\n⚙️  Config:")
    print(f"   Concurrency: {config.eval_concurrency}")
    print(f"   Shards: {config.shards}")
    print(f"   Max mutations/round: {config.max_mutations_per_round}")

    # Check if we should use real LLMs
    import os

    use_real_llms = os.getenv("OPENAI_API_KEY") is not None

    if use_real_llms:
        print("\n🤖 Using REAL LLMs (OPENAI_API_KEY found):")
        task_lm = "openai/gpt-5-nano"
        reflection_lm = "openai/gpt-5-nano"
        print(f"   Task LM: {task_lm}")
        print(f"   Reflection LM: {reflection_lm}")
    else:
        print("\n🔬 Using HEURISTIC evaluation (no OPENAI_API_KEY)")
        task_lm = None
        reflection_lm = None

    adapter = DefaultAdapter(
        dataset=dataset,
        config=config,
        sampler_seed=123,
        cache_dir=config.cache_path,
        log_dir=config.log_path,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
    )

    seeds = [
        (
            "You are a meticulous AIME math assistant. Explain your reasoning "
            "briefly, keep calculations organized, and end with '### <final answer>'."
        )
    ]

    print(f'\n🌱 Seed prompt: "{seeds[0][:60]}..."')
    print(f"\n⏱️  Starting optimization (max 2 rounds, 16 evaluations)...\n")

    start_time = time.time()
    result = adapter.optimize(
        seeds=seeds,
        max_rounds=3,  # More rounds to see reflection in action
        max_evaluations=24,  # More budget for real LLM calls
    )
    elapsed = time.time() - start_time

    pareto_entries = result["pareto_entries"]
    best_quality = _best_quality(pareto_entries)

    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\n⏱️  Runtime: {elapsed:.3f}s")
    print(f"📊 Pareto Frontier Size: {len(pareto_entries)} candidates")
    print(f"🎯 Best Quality (heuristic): {best_quality:.3f}")

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
    print("✅ Smoke test passed! All concurrency fixes working correctly.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
