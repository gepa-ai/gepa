#!/usr/bin/env python3
"""
Benchmark script to compare OG GEPA vs TurboGEPA performance.

Measures:
- Time to reach target quality
- Evaluations per second
- Total evaluations needed
- Quality improvement over time
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from typing import List, Dict, Any


# Simple math task for benchmarking
TRAINSET = [
    {"question": "What is 5 + 3?", "answer": "8"},
    {"question": "What is 12 - 7?", "answer": "5"},
    {"question": "What is 4 * 6?", "answer": "24"},
    {"question": "What is 15 / 3?", "answer": "5"},
    {"question": "What is 9 + 11?", "answer": "20"},
    {"question": "What is 20 - 8?", "answer": "12"},
    {"question": "What is 7 * 3?", "answer": "21"},
    {"question": "What is 16 / 4?", "answer": "4"},
]


def score_output(output: str, expected: str) -> float:
    """Simple scoring: 1.0 if answer is in output, 0.0 otherwise."""
    return 1.0 if expected in output.strip() else 0.0


async def run_turbo_gepa(
    max_rounds: int = 5,
    max_evals: int = 100,
) -> Dict[str, Any]:
    """Run TurboGEPA and collect metrics."""
    from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
    from turbo_gepa.config import Config

    print("\n" + "="*80)
    print("  TURBOGEPA BENCHMARK")
    print("="*80 + "\n")

    # Track metrics over time
    metrics_history = []
    start_time = time.time()

    def metrics_callback(metrics):
        """Capture metrics at each round."""
        elapsed = time.time() - start_time
        metrics_history.append({
            "round": metrics.round,
            "evaluations": metrics.evaluations,
            "best_quality": metrics.best_quality,
            "avg_quality": metrics.avg_quality,
            "elapsed_time": elapsed,
            "eval_per_sec": metrics.evaluations / elapsed if elapsed > 0 else 0,
        })

    # Create adapter with mock LM
    def mock_task_lm(prompt: str, example: dict) -> dict:
        """Mock LM that sometimes gets the answer right."""
        # Simple heuristic: longer prompts with "calculate" or "math" do better
        base_quality = 0.3
        if "calculate" in prompt.lower():
            base_quality += 0.2
        if "math" in prompt.lower():
            base_quality += 0.2
        if len(prompt) > 50:
            base_quality += 0.1

        # Simulate getting answer right with probability = quality
        import random
        correct = random.random() < base_quality
        answer = example.answer if correct else "wrong"

        return {
            "quality": 1.0 if correct else 0.0,
            "neg_cost": -0.001,  # Small token cost
            "trace": {"prompt": prompt, "question": example.input, "output": answer}
        }

    def mock_reflect_lm(parent: str, traces: List[dict]) -> List[str]:
        """Mock reflection that generates variants."""
        return [
            parent + " Please calculate carefully.",
            parent + " Think step by step about the math.",
            "You are a helpful math tutor. " + parent,
        ]

    # Monkey-patch the LM functions
    import turbo_gepa.user_plugs_in as user_plugs
    user_plugs.task_lm_call = mock_task_lm
    user_plugs.reflect_lm_call = mock_reflect_lm

    # Convert to DefaultDataInst
    dataset = [
        DefaultDataInst(input=ex["question"], answer=ex["answer"], id=f"train_{i}")
        for i, ex in enumerate(TRAINSET)
    ]

    # Create config
    config = Config(
        eval_concurrency=16,
        batch_size=8,
        shards=[0.3, 1.0],
    )

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="mock",  # Not actually used
        auto_config=False,
    )
    adapter.config = config

    # Inject metrics callback
    original_build = adapter._build_orchestrator
    def wrapped_build(*args, **kwargs):
        kwargs['metrics_callback'] = metrics_callback
        return original_build(*args, **kwargs)
    adapter._build_orchestrator = wrapped_build

    start = time.time()
    result = await adapter.optimize_async(
        seeds=["You are a helpful assistant."],
        max_rounds=max_rounds,
        max_evaluations=max_evals,
        display_progress=False,
    )
    elapsed = time.time() - start

    best_quality = max(m["best_quality"] for m in metrics_history) if metrics_history else 0.0
    total_evals = metrics_history[-1]["evaluations"] if metrics_history else 0

    print(f"\n✓ TurboGEPA completed in {elapsed:.2f}s")
    print(f"  Best quality: {best_quality:.1%}")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Throughput: {total_evals/elapsed:.1f} eval/s")

    return {
        "name": "TurboGEPA",
        "elapsed": elapsed,
        "best_quality": best_quality,
        "total_evaluations": total_evals,
        "eval_per_sec": total_evals / elapsed,
        "metrics_history": metrics_history,
    }


def run_og_gepa(
    max_rounds: int = 5,
    max_evals: int = 100,
) -> Dict[str, Any]:
    """Run OG GEPA and collect metrics."""
    import gepa
    from gepa.core.adapter import EvaluationBatch

    print("\n" + "="*80)
    print("  OG GEPA BENCHMARK")
    print("="*80 + "\n")

    # Track metrics
    metrics_history = []
    start_time = time.time()
    eval_count = [0]  # Mutable counter

    # Create custom adapter to track metrics
    class BenchmarkAdapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            """Evaluate candidate on batch."""
            outputs = []
            scores = []
            traces = [] if capture_traces else None

            for example in batch:
                eval_count[0] += 1
                prompt = candidate.get("system_prompt", "")

                # Same heuristic as TurboGEPA
                base_quality = 0.3
                if "calculate" in prompt.lower():
                    base_quality += 0.2
                if "math" in prompt.lower():
                    base_quality += 0.2
                if len(prompt) > 50:
                    base_quality += 0.1

                import random
                correct = random.random() < base_quality
                answer = example["answer"] if correct else "wrong"
                score = 1.0 if correct else 0.0

                outputs.append(answer)
                scores.append(score)
                if capture_traces:
                    traces.append({"prompt": prompt, "question": example["question"], "output": answer})

            # Record metrics
            elapsed = time.time() - start_time
            avg_quality = sum(scores) / len(scores) if scores else 0.0
            metrics_history.append({
                "evaluations": eval_count[0],
                "avg_quality": avg_quality,
                "elapsed_time": elapsed,
                "eval_per_sec": eval_count[0] / elapsed if elapsed > 0 else 0,
            })

            return EvaluationBatch(outputs, scores, traces)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            """Extract traces for reflection."""
            return {
                comp: [{"trace": t, "score": s}
                       for t, s in zip(eval_batch.trajectories or [], eval_batch.scores)]
                for comp in components_to_update
            }

    adapter = BenchmarkAdapter()

    start = time.time()
    result = gepa.optimize(
        seed_candidate={"system_prompt": "You are a helpful assistant."},
        trainset=TRAINSET,
        valset=TRAINSET[:3],  # Small valset for speed
        adapter=adapter,
        max_metric_calls=max_evals,
        reflection_minibatch_size=3,
        candidate_selection_strategy="current_best",
    )
    elapsed = time.time() - start

    # Calculate best quality from history
    best_quality = max(m["avg_quality"] for m in metrics_history) if metrics_history else 0.0
    total_evals = eval_count[0]

    print(f"\n✓ OG GEPA completed in {elapsed:.2f}s")
    print(f"  Best quality: {best_quality:.1%}")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Throughput: {total_evals/elapsed:.1f} eval/s")

    return {
        "name": "OG GEPA",
        "elapsed": elapsed,
        "best_quality": best_quality,
        "total_evaluations": total_evals,
        "eval_per_sec": total_evals / elapsed,
        "metrics_history": metrics_history,
    }


def print_comparison(turbo_results: Dict, og_results: Dict):
    """Print comparison table."""
    print("\n" + "="*80)
    print("  BENCHMARK RESULTS")
    print("="*80 + "\n")

    print(f"{'Metric':<30} {'TurboGEPA':>15} {'OG GEPA':>15} {'Speedup':>15}")
    print("-" * 80)

    # Time comparison
    speedup = og_results["elapsed"] / turbo_results["elapsed"]
    print(f"{'Time to completion (s)':<30} {turbo_results['elapsed']:>15.2f} {og_results['elapsed']:>15.2f} {speedup:>14.2f}x")

    # Throughput comparison
    throughput_ratio = turbo_results["eval_per_sec"] / og_results["eval_per_sec"]
    print(f"{'Throughput (eval/s)':<30} {turbo_results['eval_per_sec']:>15.1f} {og_results['eval_per_sec']:>15.1f} {throughput_ratio:>14.2f}x")

    # Quality comparison
    quality_ratio = turbo_results["best_quality"] / og_results["best_quality"] if og_results["best_quality"] > 0 else 1.0
    print(f"{'Best quality':<30} {turbo_results['best_quality']:>14.1%} {og_results['best_quality']:>14.1%} {quality_ratio:>14.2f}x")

    # Evaluations comparison
    eval_ratio = turbo_results["total_evaluations"] / og_results["total_evaluations"]
    print(f"{'Total evaluations':<30} {turbo_results['total_evaluations']:>15} {og_results['total_evaluations']:>15} {eval_ratio:>14.2f}x")

    print("\n" + "="*80)
    print(f"  🚀 TurboGEPA is {speedup:.1f}x FASTER than OG GEPA!")
    print(f"  📊 TurboGEPA achieves {throughput_ratio:.1f}x HIGHER throughput!")
    print("="*80 + "\n")


def save_results(turbo_results: Dict, og_results: Dict, output_file: str = "benchmark_results.json"):
    """Save results to JSON file."""
    results = {
        "timestamp": time.time(),
        "turbo_gepa": turbo_results,
        "og_gepa": og_results,
        "speedup": og_results["elapsed"] / turbo_results["elapsed"],
        "throughput_ratio": turbo_results["eval_per_sec"] / og_results["eval_per_sec"],
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_file}")


async def main():
    """Run benchmark comparing TurboGEPA vs OG GEPA."""
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark TurboGEPA vs OG GEPA")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum rounds")
    parser.add_argument("--max-evals", type=int, default=100, help="Maximum evaluations")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    parser.add_argument("--skip-og", action="store_true", help="Skip OG GEPA (faster testing)")
    args = parser.parse_args()

    print("\n🏁 Starting GEPA Benchmark Race!")
    print(f"   Configuration: max_rounds={args.max_rounds}, max_evals={args.max_evals}\n")

    # Run TurboGEPA
    turbo_results = await run_turbo_gepa(
        max_rounds=args.max_rounds,
        max_evals=args.max_evals,
    )

    if not args.skip_og:
        # Run OG GEPA
        og_results = run_og_gepa(
            max_rounds=args.max_rounds,
            max_evals=args.max_evals,
        )

        # Compare results
        print_comparison(turbo_results, og_results)

        # Save results
        save_results(turbo_results, og_results, args.output)
    else:
        print("\n⏭️  Skipped OG GEPA comparison")
        print(f"\n✓ TurboGEPA completed:")
        print(f"  - Time: {turbo_results['elapsed']:.2f}s")
        print(f"  - Throughput: {turbo_results['eval_per_sec']:.1f} eval/s")
        print(f"  - Best quality: {turbo_results['best_quality']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
