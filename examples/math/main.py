import os
from datetime import datetime

import dspy
from typing import Any

from examples.math.dataset import load_math_dataset
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    SideInfo,
    optimize_anything,
)


# ============================================================================
# EVALUATION LOGIC
# ============================================================================


def math_metric(example, prediction):
    """Compute score and detailed feedback for math problems."""
    correct_answer, written_solution = int(example.answer), getattr(example, "solution", "")
    solution_suffix = f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems" if written_solution else ""

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting. The correct answer is '{correct_answer}'.{solution_suffix}{' and ensure your final answer is a valid integer.' if written_solution else ''}"
        return dspy.Prediction(score=0.0, feedback=feedback_text)

    score = float(correct_answer == llm_answer)
    status = "correct" if score == 1.0 else "incorrect"
    feedback_text = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return dspy.Prediction(score=score, feedback=feedback_text)


def fitness_fn(candidate: dict[str, str], example) -> tuple[float, Any, SideInfo]:
    """Fitness function for GEPA optimization with single example evaluation."""
    prediction = run_llm(example, candidate["prompt"])
    metric_result = math_metric(example, prediction)
    score = metric_result.score
    feedback = metric_result.feedback

    output = {
        "prompt": candidate["prompt"],
        "answer": prediction.answer,
        "score": score,
    }

    side_info = {
        "Input": example.input,
        "Output": prediction.answer,
        "Reasoning": getattr(prediction, "reasoning", ""),
        "ExecutionFeedback": feedback,
    }

    return (score, output, side_info)


def evaluate_on_dataset(predictor, dataset, name="Evaluation"):
    """Evaluate a predictor on a dataset using dspy.Evaluate."""
    print(f"\n--- {name} ---")

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=math_metric,
        num_threads=16,
        display_progress=True,
    )

    eval_result = evaluator(predictor)
    return eval_result.score / 100.0


# ============================================================================
# DSPY SIGNATURES
# ============================================================================


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


INITIAL_PROMPT = """Solve the math problem carefully. Break down the steps and provide the final answer as a single number."""

predictor = dspy.ChainOfThought(MathSolverSignature)

def run_llm(example, prompt: str):
    """Run the LLM on a single example with the given prompt."""
    predictor.predict.signature.instructions = prompt
    return predictor(input=example.input)


# ============================================================================
# MAIN
# ============================================================================


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set.")

    solver_lm = dspy.LM(
        "gpt-4.1-mini", api_key=api_key, temperature=1.0, max_tokens=32000
    )
    proposal_lm = dspy.LM(
        "openai/gpt-5.1", api_key=api_key, temperature=1.0, max_tokens=32000
    )
    dspy.configure(lm=solver_lm)

    trainset, valset, testset = load_math_dataset()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = f"outputs/artifacts/aime_math/{timestamp}"

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=artifacts_dir,
            seed=42,
            max_metric_calls=600,
            track_best_outputs=True,
            parallel=True,
            max_workers=32,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            skip_perfect_score=False,
            reflection_lm=proposal_lm,
        ),
    )

    # Run GEPA optimization
    print("\nStarting GEPA Optimization for Math Problems...")

    result = optimize_anything(
        seed_candidate={"prompt": INITIAL_PROMPT},
        fitness_fn=fitness_fn,
        dataset=trainset,
        valset=valset,
        config=gepa_config,
    )

    # Baseline Evaluation
    print("\nEvaluating Baseline (Initial Prompt)...")
    predictor.predict.signature.instructions = INITIAL_PROMPT
    baseline_score = evaluate_on_dataset(predictor, testset, name="Baseline Test")

    # Optimized Evaluation
    print("\nEvaluating Best Optimized Program...")
    best_prompt = result.best_candidate["prompt"]
    print(f"Best Prompt Found:\n{best_prompt}")

    predictor.predict.signature.instructions = best_prompt
    optimized_score = evaluate_on_dataset(predictor, testset, name="Optimized Test")

    print(f"Baseline Score: {baseline_score:.2%}")
    print(f"Optimized Score: {optimized_score:.2%}")
    print(f"Improvement: {optimized_score - baseline_score:.2%}")

    print(f"\nOptimization complete. Artifacts saved to {artifacts_dir}")

if __name__ == "__main__":
    main()
