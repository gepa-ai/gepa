import os
from typing import Any

import dspy

from examples.math.dataset import load_math_dataset
from gepa.core.adapter import DataInst
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    TrackingConfig,
    optimize_anything,
)

# ============================================================================
# DSPY SIGNATURES
# ============================================================================


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


predictor = dspy.ChainOfThought(MathSolverSignature)


# ============================================================================
# EVALUATION LOGIC
# ============================================================================


def run_llm(example: DataInst, prompt: str) -> dspy.Prediction:
    """Run the LLM with the given prompt and return the prediction."""
    predictor.predict.signature.instructions = prompt
    return predictor(example)


def math_metric(example: DataInst, prediction: dspy.Prediction) -> dspy.Prediction:
    """Compute score and detailed feedback for math problems."""
    correct_answer = int(example.answer)
    written_solution = getattr(example, "solution", "")

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."

        return dspy.Prediction(score=0.0, feedback=feedback_text)

    score = float(correct_answer == llm_answer)

    if score == 1.0:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback_text)


def create_fitness_function(predictor: dspy.Module):
    """Create fitness function for GEPA optimization using dspy.Evaluate for parallel rollouts."""

    def fitness_fn(candidate: dict[str, str], example: Any, **kwargs) -> list[tuple[float, Any, SideInfo]]:
        prediction = run_llm(predictor, example, candidate["prompt"])
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

    return fitness_fn


def evaluate_on_dataset(prompt: str, dataset: list[DataInst]) -> float:
    predictor.predict.signature.instructions = prompt
    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=math_metric,
        num_threads=32,
        display_progress=True,
    )
    eval_result = evaluator(predictor)
    return eval_result.score / 100.0


# ============================================================================
# Candidate
# ============================================================================


INITIAL_PROMPT = (
    """Solve the math problem carefully. Break down the steps and provide the final answer as a single number."""
)


# ============================================================================
# MAIN
# ============================================================================


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set.")

    lm = dspy.LM("gpt-4.1-mini", api_key=api_key, temperature=1.0, max_tokens=32000)
    dspy.configure(lm=lm)

    trainset, valset, testset = load_math_dataset()

    task_name = "math"
    artifacts_dir = f"outputs/artifacts/{task_name}"
    logs_dir = f"outputs/logs/{task_name}"
    plots_dir = f"outputs/plots/{task_name}"

    for d in [artifacts_dir, logs_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=artifacts_dir,
            max_metric_calls=600,
            track_best_outputs=True,
            parallel=True,
            max_workers=32,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            skip_perfect_score=False,
            reflection_lm="openai/gpt-5.1",
        ),
        tracking=TrackingConfig(use_wandb=False),
    )

    # Run GEPA optimization
    print("\nStarting GEPA Optimization for Math Problems...")
    fitness_fn = create_fitness_function(predictor)

    result = optimize_anything(
        seed_candidate={"prompt": INITIAL_PROMPT},
        fitness_fn=fitness_fn,
        dataset=trainset,
        valset=valset,
        config=gepa_config,
    )

    # Baseline Evaluation
    print("\nEvaluating Baseline (Initial Prompt)...")
    baseline_score = evaluate_on_dataset(INITIAL_PROMPT, testset)

    # Optimized Evaluation
    print("\nEvaluating Best Optimized Program...")
    best_prompt = result.best_candidate["prompt"]
    print(f"Best Prompt Found:\n{best_prompt}")

    optimized_score = evaluate_on_dataset(best_prompt, testset)

    print(f"Baseline Score: {baseline_score:.2%}")
    print(f"Optimized Score: {optimized_score:.2%}")
    print(f"Improvement: {optimized_score - baseline_score:.2%}")

    print(f"\nOptimization complete. Artifacts saved to {artifacts_dir}")
    print(f"Logs can be found in {logs_dir}")
    print(f"Plots can be found in {plots_dir}")


if __name__ == "__main__":
    main()
