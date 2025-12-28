import os
from typing import Any, Sequence

import dspy
from dataset import load_math_dataset
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    TrackingConfig,
    optimize_anything,
)


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


def math_evaluation(example, prediction):
    """Compute score and detailed feedback for math problems."""
    correct_answer = int(example.answer)
    written_solution = getattr(example, "solution", "")

    reasoning = getattr(prediction, "reasoning", getattr(prediction, "rationale", "No reasoning provided."))

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."

        full_feedback = f"Model Reasoning:\n{reasoning}\n\n{feedback_text}"
        return 0.0, full_feedback

    score = float(correct_answer == llm_answer)

    if score == 1.0:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    full_feedback = f"Model Reasoning:\n{reasoning}\n\n{feedback_text}"
    return score, full_feedback


def math_metric(example, prediction, trace=None):
    """Metric wrapper for dspy.Evaluate."""
    score, _ = math_evaluation(example, prediction)
    return score


def evaluate_on_dataset(predictor, dataset, name="Evaluation"):
    """Evaluate a predictor on a dataset using dspy.Evaluate."""
    print(f"\n--- {name} ---")

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=math_metric,
        num_threads=16,
        display_progress=True,
    )

    score = evaluator(predictor)
    accuracy = score / 100.0
    print(f"{name} Accuracy: {accuracy:.2%}")
    return accuracy


def create_fitness_function(predictor: dspy.Module):
    """Create fitness function for GEPA optimization using dspy.Evaluate for parallel rollouts."""

    def fitness_fn(candidate: dict[str, str], batch: Sequence[Any], **kwargs) -> list[tuple[float, Any, SideInfo]]:
        predictor.predict.signature.instructions = candidate["prompt"]

        evaluator = dspy.Evaluate(
            devset=list(batch), metric=math_metric, num_threads=16, display_progress=True, display_table=False
        )

        eval_result = evaluator(predictor)

        results = []
        for example, prediction, score in eval_result.results:
            _, feedback = math_evaluation(example, prediction)

            side_info = {
                "Input": example.input,
                "Output": prediction.answer,
                "Rationale": getattr(prediction, "reasoning", getattr(prediction, "rationale", "")),
                "ExecutionFeedback": feedback,
                "score": score,
            }

            results.append((score, prediction.answer, side_info))

        return results

    return fitness_fn


INITIAL_PROMPT = (
    """Solve the math problem carefully. Break down the steps and provide the final answer as a single number."""
)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set.")

    lm = dspy.LM("gpt-4.1-mini", api_key=api_key, temperature=1.0, max_tokens=32000)
    reflection_lm = dspy.LM("openai/gpt-5", api_key=api_key, temperature=1.0, max_tokens=32000)
    dspy.configure(lm=lm)

    trainset, valset, testset = load_math_dataset()

    predictor = dspy.ChainOfThought(MathSolverSignature)

    task_name = "math"
    artifacts_dir = f"outputs/artifacts/{task_name}"
    logs_dir = f"outputs/logs/{task_name}"
    plots_dir = f"outputs/plots/{task_name}"

    for d in [artifacts_dir, logs_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=artifacts_dir,
            seed=42,
            max_metric_calls=600,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            skip_perfect_score=False,
            reflection_lm=reflection_lm,
        ),
        tracking=TrackingConfig(use_wandb=False),
    )

    print("\nStarting GEPA Optimization for Math Problems...")
    fitness_fn = create_fitness_function(predictor)

    result = optimize_anything(
        seed_candidate={"prompt": INITIAL_PROMPT},
        fitness_fn=fitness_fn,
        dataset=trainset,
        valset=valset,
        config=gepa_config,
    )

    print("\nEvaluating Baseline (Initial Prompt)...")
    predictor.predict.signature.instructions = INITIAL_PROMPT
    evaluate_on_dataset(predictor, testset, name="Baseline Test")

    print("\nEvaluating Best Optimized Program...")
    best_prompt = result.best_candidate["prompt"]
    print(f"Best Prompt Found:\n{best_prompt}")

    predictor.predict.signature.instructions = best_prompt
    evaluate_on_dataset(predictor, testset, name="Optimized Test")

    print(f"\nOptimization complete. Artifacts saved to {artifacts_dir}")
    print(f"Logs can be found in {logs_dir}")
    print(f"Plots can be found in {plots_dir}")


if __name__ == "__main__":
    main()
