#!/usr/bin/env python3
"""
Comparison script: Original GEPA vs TurboGEPA with Temperature Optimization

This script runs both implementations on the AIME dataset and compares:
1. Original GEPA - Standard prompt optimization
2. TurboGEPA - High-throughput with adaptive sharding
3. TurboGEPA + Temperature - Staged temperature optimization

Usage:
    python examples/compare_gepa_vs_turbo.py --mode [gepa|turbo|turbo-temp|all]

    --mode gepa: Run original GEPA only
    --mode turbo: Run TurboGEPA only
    --mode turbo-temp: Run TurboGEPA with temperature optimization
    --mode all: Run all three (default)
"""

import argparse
import json
import time
from pathlib import Path

# Import both versions
import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.interfaces import Candidate


def load_aime_dataset(limit=None):
    """Load AIME dataset in both formats."""
    print("📥 Loading AIME dataset...")
    trainset, valset, testset = gepa.examples.aime.init_dataset()

    if limit:
        trainset = trainset[:limit]
        valset = valset[:limit]

    # Convert to Turbo format
    turbo_trainset = [
        DefaultDataInst(
            input=x["input"],
            answer=x["answer"],
            additional_context=x.get("additional_context"),
            id=f"train-{i}",
            difficulty=0.5,  # AIME problems are generally hard
        )
        for i, x in enumerate(trainset)
    ]

    print(f"✓ Loaded {len(trainset)} training examples, {len(valset)} validation examples")
    return trainset, valset, testset, turbo_trainset


def run_original_gepa(trainset, valset, max_calls=150):
    """Run original GEPA optimization."""
    print("\n" + "="*80)
    print("🔬 ORIGINAL GEPA - Standard Prompt Optimization")
    print("="*80)

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    print(f"📊 Configuration:")
    print(f"   Task LM: openai/gpt-4.1-mini")
    print(f"   Reflection LM: openai/gpt-5")
    print(f"   Max metric calls: {max_calls}")
    print(f"   Seed prompt length: {len(seed_prompt['system_prompt'])} chars")

    start_time = time.time()

    try:
        result = gepa.optimize(
            seed_candidate=seed_prompt,
            trainset=trainset,
            valset=valset,
            task_lm="openai/gpt-4.1-mini",
            max_metric_calls=max_calls,
            reflection_lm="openai/gpt-5",
        )

        elapsed = time.time() - start_time

        print(f"\n✅ GEPA Optimization Complete!")
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"\n📝 Optimized Prompt:")
        print("-" * 80)
        print(result.best_candidate['system_prompt'])
        print("-" * 80)

        return {
            "method": "Original GEPA",
            "prompt": result.best_candidate['system_prompt'],
            "time": elapsed,
            "candidates_explored": len(result.candidates),
        }

    except Exception as e:
        print(f"❌ Error running original GEPA: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_turbo_gepa(trainset, max_rounds=10, max_evals=150, use_temperature=False):
    """Run TurboGEPA optimization."""
    temp_label = " + TEMPERATURE" if use_temperature else ""
    print("\n" + "="*80)
    print(f"⚡ UFAST-GEPA{temp_label} - High-Throughput Optimization")
    print("="*80)

    seed_prompt = "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"

    print(f"📊 Configuration:")
    print(f"   Max rounds: {max_rounds}")
    print(f"   Max evaluations: {max_evals}")
    print(f"   Temperature optimization: {'Yes (staged)' if use_temperature else 'No'}")
    print(f"   Seed prompt length: {len(seed_prompt)} chars")

    adapter = DefaultAdapter(
        dataset=trainset,
        sampler_seed=42,
    )

    # Note: DefaultAdapter uses heuristic evaluation (no real LLM calls)
    # For real LLM evaluation, implement task_lm_call in user_plugs_in.py
    print(f"\n⚠️  Note: Using heuristic evaluation (no real LLM calls)")
    print(f"   For production: implement task_lm_call() in turbo_gepa/user_plugs_in.py")

    start_time = time.time()

    try:
        result = adapter.optimize(
            seeds=[seed_prompt],
            max_rounds=max_rounds,
            max_evaluations=max_evals,
            optimize_temperature_after_convergence=use_temperature,
        )

        elapsed = time.time() - start_time

        pareto = result["pareto"]
        phase1_pareto = result.get("phase1_pareto", [])

        print(f"\n✅ TurboGEPA Optimization Complete!")
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📈 Pareto frontier: {len(pareto)} candidates")

        if use_temperature and phase1_pareto:
            print(f"📈 Phase 1 (prompts only): {len(phase1_pareto)} candidates")

        # Show best candidate by quality
        best = max(pareto, key=lambda c: c.meta.get("quality", 0.0))
        temp_info = f" (temp={best.meta.get('temperature', 'N/A')})" if use_temperature else ""

        print(f"\n📝 Best Optimized Prompt{temp_info}:")
        print("-" * 80)
        print(best.text)
        print("-" * 80)

        if use_temperature:
            # Show temperature distribution
            temps = [c.meta.get("temperature") for c in pareto if "temperature" in c.meta]
            if temps:
                print(f"\n🌡️  Temperature distribution: {sorted(set(temps))}")

        return {
            "method": f"TurboGEPA{temp_label}",
            "prompt": best.text,
            "temperature": best.meta.get("temperature"),
            "time": elapsed,
            "pareto_size": len(pareto),
            "phase1_size": len(phase1_pareto) if use_temperature else None,
        }

    except Exception as e:
        print(f"❌ Error running TurboGEPA: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results, output_file="comparison_results.json"):
    """Save comparison results to JSON."""
    output_path = Path(output_file)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare GEPA vs TurboGEPA")
    parser.add_argument(
        "--mode",
        choices=["gepa", "turbo", "turbo-temp", "all"],
        default="all",
        help="Which implementation(s) to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit dataset size (default: 20)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=150,
        help="Max evaluations/metric calls (default: 150)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Max rounds for TurboGEPA (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="comparison_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    print("🚀 GEPA vs TurboGEPA Comparison")
    print(f"Mode: {args.mode}")
    print(f"Dataset limit: {args.limit}")

    # Load dataset
    trainset, valset, testset, turbo_trainset = load_aime_dataset(limit=args.limit)

    results = []

    # Run requested comparisons
    if args.mode in ["gepa", "all"]:
        result = run_original_gepa(trainset, valset, max_calls=args.max_calls)
        if result:
            results.append(result)

    if args.mode in ["turbo", "all"]:
        result = run_turbo_gepa(
            turbo_trainset,
            max_rounds=args.max_rounds,
            max_evals=args.max_calls,
            use_temperature=False,
        )
        if result:
            results.append(result)

    if args.mode in ["turbo-temp", "all"]:
        result = run_turbo_gepa(
            turbo_trainset,
            max_rounds=args.max_rounds,
            max_evals=args.max_calls,
            use_temperature=True,
        )
        if result:
            results.append(result)

    # Print summary
    if results:
        print("\n" + "="*80)
        print("📊 COMPARISON SUMMARY")
        print("="*80)
        for r in results:
            print(f"\n{r['method']}:")
            print(f"  Time: {r['time']:.1f}s")
            if "candidates_explored" in r:
                print(f"  Candidates explored: {r['candidates_explored']}")
            if "pareto_size" in r:
                print(f"  Pareto size: {r['pareto_size']}")
            if r.get("temperature") is not None:
                print(f"  Best temperature: {r['temperature']}")
            print(f"  Prompt length: {len(r['prompt'])} chars")

        save_results(results, args.output)

    print("\n✨ Comparison complete!")


if __name__ == "__main__":
    main()
