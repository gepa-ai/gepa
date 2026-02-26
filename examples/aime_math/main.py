import os

import dspy

from examples.aime_math.utils import evaluate_on_dataset, load_math_dataset, math_metric, run_llm
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    """Evaluate a candidate on a single example."""
    prediction = run_llm(example, candidate)
    score, feedback = math_metric(example, prediction)

    side_info = {
        "score": score,
        "input": example.input,
        "prompt": candidate,
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
    }

    return score, side_info


def main():
    INITIAL_PROMPT = (
        "Solve the math problem carefully. Break down the steps and provide the final answer as a single number."
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    solver_lm = dspy.LM("gpt-4.1-mini", api_key=api_key, temperature=1.0, max_tokens=32000)
    dspy.configure(lm=solver_lm)

    trainset, valset, testset = load_math_dataset()

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir="outputs/aime_math",
            max_metric_calls=600,
            track_best_outputs=True,
            parallel=True,
            max_workers=32,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm="openai/gpt-5.1",
        ),
    )

    result = optimize_anything(
        seed_candidate=INITIAL_PROMPT,
        evaluator=evaluate,
        dataset=trainset,
        valset=valset,
        config=gepa_config,
    )

    # Baseline Evaluation
    print("\nEvaluating Baseline (Initial Prompt)...")
    baseline_score = evaluate_on_dataset(INITIAL_PROMPT, testset)

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
