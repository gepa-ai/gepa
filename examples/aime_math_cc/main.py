"""
Optimize AIME math-solving prompts using Claude Code as the reflection LM (seedless mode).

CC writes skills (generation prompts) that instruct GPT-4.1-mini how to solve AIME problems.
This exercises the full CC reflector + eval recorder + skill-framing prompt with real
multi-task Pareto dynamics (90 AIME problems across train/val).

Usage:
    OPENAI_API_KEY=... uv run python -m examples.aime_math_cc.main
"""

import os
from pathlib import Path

import dspy

from examples.aime_math_cc.utils import evaluate_on_dataset, load_math_dataset, math_metric, run_llm
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    """Evaluate a candidate prompt on a single AIME example."""
    print(f"  [eval] problem={example.input[:60]}... answer={example.answer}")
    prediction = run_llm(example, candidate)
    score, feedback = math_metric(example, prediction)
    print(f"  [eval] predicted={prediction.answer} correct={example.answer} score={score}")

    side_info = {
        "score": score,
        "input": example.input,
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
    }

    return score, side_info


def main():
    print("[main] Loading dataset...")
    api_key = os.environ.get("OPENAI_API_KEY")
    solver_lm = dspy.LM("gpt-4.1-mini", api_key=api_key, temperature=1.0, max_tokens=32000)
    dspy.configure(lm=solver_lm)

    trainset, valset, testset = load_math_dataset()
    # Use full train/val splits (45/45 from 90 total AIME problems)
    # trainset = trainset[:20]
    # valset = valset[:3]
    print(f"[main] Dataset loaded: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    run_dir = str(Path(__file__).parent / "run_output")
    print(f"[main] Run dir: {run_dir}")
    print("[main] Starting optimize_anything (seedless + claude_code)...")
    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir,
            max_metric_calls=500,
            track_best_outputs=True,
            parallel=True,
            max_workers=32,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm="claude_code",
        ),
    )

    print("[main] Config ready. Launching optimization loop...")
    result = optimize_anything(
        seed_candidate=None,
        evaluator=evaluate,
        objective="Write a detailed instruction prompt that teaches an LLM how to solve competition-level AIME math problems. The prompt should guide the solver through problem decomposition, algebraic manipulation, number theory reasoning, and careful numerical computation to arrive at a single integer answer (0-999).",
        dataset=trainset,
        valset=valset,
        config=gepa_config,
    )

    print(f"[main] Optimization complete. Best candidate index: {result.best_idx}")

    # Baseline Evaluation
    baseline_prompt = (
        "Solve the math problem carefully. Break down the steps and provide the final answer as a single number."
    )
    print("\nEvaluating Baseline (Initial Prompt)...")
    baseline_score = evaluate_on_dataset(baseline_prompt, testset)

    # Optimized Evaluation
    print("\nEvaluating Best Optimized Program...")
    best_prompt = result.best_candidate
    print(f"Best Prompt Found:\n{best_prompt}")

    optimized_score = evaluate_on_dataset(best_prompt, testset)

    print(f"Baseline Score: {baseline_score:.2%}")
    print(f"Optimized Score: {optimized_score:.2%}")
    print(f"Improvement: {optimized_score - baseline_score:.2%}")


if __name__ == "__main__":
    main()
