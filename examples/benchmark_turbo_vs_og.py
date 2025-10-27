"""
Simple benchmark comparing TurboGEPA vs Original GEPA.

Runs both on the same task and compares:
- Runtime (wall clock time)
- Final quality achieved
- Number of evaluations used

Usage:
    python examples/benchmark_turbo_vs_og.py
"""

import asyncio
import os
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    name: str
    runtime_seconds: float
    final_quality: float
    num_evaluations: int
    best_prompt: str


def create_simple_dataset(n_examples: int = 50):
    """Create a simple math dataset for benchmarking."""
    dataset = []
    for i in range(n_examples):
        a, b = i % 10 + 1, (i * 2) % 10 + 1
        dataset.append({
            "input": f"What is {a} + {b}?",
            "answer": str(a + b)
        })
    return dataset


async def benchmark_turbo_gepa(dataset, max_evals=100):
    """Benchmark TurboGEPA."""
    from turbo_gepa import DefaultAdapter

    print("\n🏎️  Running TurboGEPA benchmark...")
    start = time.time()

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=os.getenv("OPENAI_API_KEY") and "openai/gpt-4o-mini" or "openrouter/openai/gpt-4o-mini",
        reflection_lm=os.getenv("OPENAI_API_KEY") and "openai/gpt-4o-mini" or "openrouter/openai/gpt-4o-mini",
        auto_config=True,
    )

    result = await adapter.optimize_async(
        seeds=["You are a helpful math assistant. Answer concisely with just the number."],
        max_evaluations=max_evals,
        show_progress=False,
    )

    runtime = time.time() - start

    # Get best candidate from Pareto frontier
    best_candidate = max(
        result.archive.pareto_candidates(),
        key=lambda c: c.meta.get("quality", 0.0)
    )

    return BenchmarkResult(
        name="TurboGEPA",
        runtime_seconds=runtime,
        final_quality=best_candidate.meta.get("quality", 0.0),
        num_evaluations=result.evaluations,
        best_prompt=best_candidate.text[:100] + "..."
    )


def benchmark_og_gepa(dataset, max_evals=100):
    """Benchmark Original GEPA."""
    import gepa

    print("\n🐢 Running Original GEPA benchmark...")
    start = time.time()

    # Convert dataset format for OG GEPA
    class SimpleExample:
        def __init__(self, data):
            self.data = data

        def to_payload(self):
            return self.data

    gepa_dataset = [SimpleExample(d) for d in dataset]

    result = gepa.optimize(
        seed_candidate={"system_prompt": "You are a helpful math assistant. Answer concisely with just the number."},
        trainset=gepa_dataset,
        valset=gepa_dataset[:10],  # Use small valset
        task_lm=os.getenv("OPENAI_API_KEY") and "openai/gpt-4o-mini" or "openrouter/openai/gpt-4o-mini",
        reflection_lm=os.getenv("OPENAI_API_KEY") and "openai/gpt-4o-mini" or "openrouter/openai/gpt-4o-mini",
        max_metric_calls=max_evals,
    )

    runtime = time.time() - start

    return BenchmarkResult(
        name="Original GEPA",
        runtime_seconds=runtime,
        final_quality=result.best_score if hasattr(result, 'best_score') else 0.0,
        num_evaluations=max_evals,
        best_prompt=str(result.best_candidate.get("system_prompt", ""))[:100] + "..."
    )


def print_comparison(turbo_result: BenchmarkResult, og_result: BenchmarkResult):
    """Print a nice comparison table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS".center(80))
    print("="*80)

    # Runtime comparison
    speedup = og_result.runtime_seconds / turbo_result.runtime_seconds if turbo_result.runtime_seconds > 0 else 0
    print(f"\n⏱️  Runtime:")
    print(f"   TurboGEPA:     {turbo_result.runtime_seconds:.1f}s")
    print(f"   Original GEPA: {og_result.runtime_seconds:.1f}s")
    print(f"   Speedup:       {speedup:.2f}x faster")

    # Quality comparison
    print(f"\n📊 Final Quality:")
    print(f"   TurboGEPA:     {turbo_result.final_quality:.1%}")
    print(f"   Original GEPA: {og_result.final_quality:.1%}")
    quality_diff = turbo_result.final_quality - og_result.final_quality
    if abs(quality_diff) < 0.01:
        print(f"   Difference:    ~Same quality")
    else:
        print(f"   Difference:    {quality_diff:+.1%}")

    # Evaluations
    print(f"\n🔢 Evaluations:")
    print(f"   TurboGEPA:     {turbo_result.num_evaluations}")
    print(f"   Original GEPA: {og_result.num_evaluations}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY".center(80))
    print("="*80)
    if speedup > 2:
        print(f"✅ TurboGEPA is {speedup:.1f}x faster with similar quality!")
    elif speedup > 1.5:
        print(f"✅ TurboGEPA is {speedup:.1f}x faster!")
    else:
        print(f"⚠️  Results similar, may need larger dataset to see speedup")
    print("="*80 + "\n")


async def main():
    """Run the benchmark."""
    print("\n🔬 Starting TurboGEPA vs Original GEPA Benchmark")
    print("="*80)

    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable")
        return

    # Create dataset
    dataset = create_simple_dataset(n_examples=50)
    print(f"📚 Dataset: {len(dataset)} math examples")

    # Run benchmarks
    max_evals = 100
    print(f"🎯 Budget: {max_evals} evaluations per system")

    try:
        # TurboGEPA
        turbo_result = await benchmark_turbo_gepa(dataset, max_evals)

        # Original GEPA
        og_result = benchmark_og_gepa(dataset, max_evals)

        # Compare
        print_comparison(turbo_result, og_result)

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
