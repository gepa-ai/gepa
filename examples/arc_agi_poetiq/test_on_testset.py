#!/usr/bin/env python3
"""Test seed vs best agent on the ARC-AGI test set."""

import argparse
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# Unbuffered print
print = partial(print, flush=True)

from examples.arc_agi_poetiq.evaluate import run_agent
from examples.arc_agi_poetiq.main import SEED_AGENT_CODE, load_arc_dataset


def evaluate_on_testset(
    agent_code: str,
    test_set: list,
    model_id: str,
    max_llm_calls: int,
    max_workers: int = 32,
    reasoning_effort: str | None = None,
) -> dict:
    """Evaluate agent on test set in parallel."""
    results = []

    def eval_one(ex):
        result = run_agent(
            agent_code=agent_code,
            train_in=ex.train_in,
            train_out=ex.train_out,
            test_in=ex.test_in,
            test_out=ex.test_out or None,
            model_id=model_id,
            max_llm_calls=max_llm_calls,
            reasoning_effort=reasoning_effort,
        )
        return {
            "problem_id": ex.problem_id,
            "training_score": result["training_score"],
            "test_score": result["test_score"],
            "cost": result["llm"].total_cost,
            "error": result["error"],
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_one, ex): ex.problem_id for ex in test_set}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                r = future.result()
                results.append(r)
                status = "✓" if r["test_score"] == 1.0 else "✗"
                print(f"[{r['problem_id']}] {status} train={r['training_score']:.0%} test={r['test_score']:.0%} cost=${r['cost']:.4f}")
            except Exception as e:
                print(f"[{pid}] ERROR: {e}")
                results.append({"problem_id": pid, "test_score": 0.0, "error": str(e)})

    # Aggregate
    solved = sum(1 for r in results if r.get("test_score", 0) == 1.0)
    total = len(results)
    total_cost = sum(r.get("cost", 0) for r in results)

    return {
        "solved": solved,
        "total": total,
        "accuracy": solved / total if total > 0 else 0.0,
        "total_cost": total_cost,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/artifacts/arc_agi_poetiq/gemini-3-flash_260201_230849")
    parser.add_argument("--model", default="openrouter/google/gemini-3-flash-preview")
    parser.add_argument("--max-llm-calls", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--num-problems", type=int, default=None, help="Limit number of test problems")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning-effort", type=str, default="high", choices=["low", "medium", "high"],
                        help="Reasoning effort for supported models (default: high)")
    args = parser.parse_args()

    # Load GEPA state to get best candidate
    state_path = Path(args.run_dir) / "gepa_state.bin"
    with open(state_path, "rb") as f:
        state = pickle.load(f)

    # Get best candidate by cumulative val score (scores are stored as dicts)
    val_agg = [sum(scores.values()) for scores in state["prog_candidate_val_subscores"]]
    best_idx = max(range(len(val_agg)), key=lambda i: val_agg[i])
    best_candidate = state["program_candidates"][best_idx]
    best_agent_code = best_candidate["agent_code"]

    print(f"Loaded GEPA state from {state_path}")
    print(f"Best program index: {best_idx}")
    print(f"Best val score: {val_agg[best_idx]:.1f}/200 ({val_agg[best_idx]/200:.1%})")
    print(f"Seed val score: {val_agg[0]:.1f}/200 ({val_agg[0]/200:.1%})")
    print()

    # Load test set
    _, _, test_set = load_arc_dataset(seed=args.seed)
    if args.num_problems:
        test_set = test_set[:args.num_problems]
    print(f"Test set: {len(test_set)} problems\n")

    # Evaluate best first
    print("=" * 60)
    print("BEST AGENT (from GEPA)")
    print("=" * 60)
    best_results = evaluate_on_testset(
        best_agent_code, test_set, args.model, args.max_llm_calls, args.max_workers, args.reasoning_effort
    )
    print(f"\nBest: {best_results['solved']}/{best_results['total']} solved ({best_results['accuracy']:.1%}), cost=${best_results['total_cost']:.2f}")

    # Evaluate seed
    print("\n" + "=" * 60)
    print("SEED AGENT")
    print("=" * 60)
    seed_results = evaluate_on_testset(
        SEED_AGENT_CODE, test_set, args.model, args.max_llm_calls, args.max_workers, args.reasoning_effort
    )
    print(f"\nSeed: {seed_results['solved']}/{seed_results['total']} solved ({seed_results['accuracy']:.1%}), cost=${seed_results['total_cost']:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Seed:  {seed_results['solved']}/{seed_results['total']} ({seed_results['accuracy']:.1%})")
    print(f"Best:  {best_results['solved']}/{best_results['total']} ({best_results['accuracy']:.1%})")
    improvement = best_results['accuracy'] - seed_results['accuracy']
    print(f"Δ:     {improvement:+.1%}")

    # Save results
    output_path = Path(args.run_dir) / "test_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "seed": seed_results,
            "best": best_results,
            "best_agent_code": best_agent_code,
        }, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
