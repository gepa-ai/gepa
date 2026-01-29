import json
import os
import random

import dspy
import numpy as np
from examples.arc_agi.config import (
    parse_arguments,
    get_experiment_config,
    get_log_directory,
)
from examples.arc_agi.data import load_data
from examples.arc_agi.prompt import BACKGROUND

from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)


def save_experiment_config(config_dict, log_dir):
    """Save experiment configuration to JSON file."""
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"💾 Saved config to: {config_path}")

program_src = """import dspy
from typing import List
import pydantic

MATRIX = List[List[int]]

class TrainingExample(pydantic.BaseModel):
    input: MATRIX
    output: MATRIX

class SolveTaskSignature(dspy.Signature):
    training_examples: List[TrainingExample] = dspy.InputField(description="Input and output examples demonstrating the task to be performed.")
    test_inputs: List[MATRIX] = dspy.InputField(description="Input matrices to be solved following the task described in the training examples.")
    test_outputs: List[MATRIX] = dspy.OutputField(description="Output matrices corresponding to the test inputs.")

program = dspy.ChainOfThought(SolveTaskSignature)"""


def is_valid_matrix(matrix, gold_matrix):
    if not isinstance(matrix, list):
        return (
            False,
            f"The matrix must be a List[List[int]]. The correct matrix is {gold_matrix}.",
        )
    n = len(matrix)
    if n == 0:
        return (
            False,
            f"The matrix must have at least one row. The correct matrix is {gold_matrix}.",
        )
    m = len(matrix[0])
    if m == 0:
        return (
            False,
            f"The matrix must have at least one column. The correct matrix is {gold_matrix}.",
        )
    for i in range(n):
        if not isinstance(matrix[i], list):
            return (
                False,
                f"The {i}-th row must be a List[int]. The correct matrix is {gold_matrix}.",
            )
        if len(matrix[i]) != m:
            return (
                False,
                f"The matrix is staggered. Row 0 has {m} columns, but row {i} has {len(matrix[i])} columns. The correct matrix is {gold_matrix}.",
            )
        for j in range(m):
            if not isinstance(matrix[i][j], int):
                return (
                    False,
                    f"The {i}-th row, {j}-th column must be an int, found {type(matrix[i][j])}. The correct matrix is {gold_matrix}.",
                )

    # Check consistency with gold matrix
    gold_n = len(gold_matrix)
    gold_m = len(gold_matrix[0])
    if (n, m) != (gold_n, gold_m):
        return (
            False,
            f"The matrix has dimensions {n}x{m}, but the gold matrix has dimensions {gold_n}x{gold_m}. The correct matrix is {gold_matrix}.",
        )

    same = True
    wrong_indices = []
    for i in range(n):
        for j in range(m):
            if matrix[i][j] != gold_matrix[i][j]:
                same = False
                wrong_indices.append((i, j))
    if same:
        return True, f"Your response is correct. The correct matrix is {gold_matrix}."
    else:
        if len(wrong_indices) < 10:
            return (
                False,
                f"The matrix is incorrect. The following indices are incorrect: {wrong_indices}. The correct matrix is {gold_matrix}.",
            )
        else:
            return (
                False,
                f"The matrix is incorrect. The correct matrix is {gold_matrix}.",
            )


def metric_fn(example, pred, trace=None):
    task_inputs = example.get("test_inputs", "empty")
    gold_task_outputs = example.get("test_outputs", "empty")
    pred_task_outputs = pred.get("test_outputs", "empty")

    if not isinstance(pred_task_outputs, list):
        return dspy.Prediction(
            score=0,
            feedback=f"The response must be a List[List[List[int]]]. The correct response is {gold_task_outputs}.",
        )

    valids = []
    feedbacks = []
    feedback = ""
    if len(task_inputs) != len(pred_task_outputs):
        feedback = f"The number of output matrices ({len(pred_task_outputs)}) must match the number of input matrices ({len(task_inputs)}). The correct response is {gold_task_outputs}."
        return dspy.Prediction(score=0, feedback=feedback)
    for i, (_input, gold_output, pred_output) in enumerate(
        zip(task_inputs, gold_task_outputs, pred_task_outputs, strict=False)
    ):
        is_valid, feedback = is_valid_matrix(pred_output, gold_output)
        valids.append(is_valid)
        feedbacks.append(f"Feedback on test input {i}: {feedback}")

    score = sum(valids) / len(valids)
    feedback_text = "\n".join(feedbacks)
    return dspy.Prediction(score=score, feedback=feedback_text)


def create_fitness_fn(adapter):
    """Create fitness function with adapter in closure."""

    def fitness_fn(candidate, example):
        program = candidate["program"]

        try:
            evaluation_results = adapter.evaluate(
                [example], candidate, capture_traces=True
            )
        except Exception as e:
            side_info = {"input": example, "error": str(e), "program": program}
            return (0.0, side_info, side_info)

        # Program error
        if (
            not isinstance(evaluation_results.trajectories, list)
            or len(evaluation_results.trajectories) == 0
        ):
            print("Error: ")
            print(evaluation_results.trajectories)
            side_info = {
                "input": example,
                "error": f"All examples failed. Program error: {evaluation_results.trajectories!s}",
                "program": program,
            }
            return (0.0, side_info, side_info)

        # Process evaluations with no program errors
        trajectory = evaluation_results.trajectories[0]
        metric_result = trajectory.get("score")
        score = metric_result.get("score")
        feedback = metric_result.get("feedback")
        prediction = trajectory.get("prediction")

        side_info = {
            "input": example,
            "reasoning": prediction.get("reasoning"),
            "feedback": feedback,
            "output": prediction.get("test_outputs"),
        }

        return (score, side_info, side_info)

    return fitness_fn


def main():
    """Main entry point for ARC-AGI GEPA optimization."""

    print("=" * 80)
    print("GEPA ARC-AGI Solver Optimization")
    print("=" * 80)

    args = parse_arguments()
    config = get_experiment_config(args)
    log_dir = get_log_directory(resume=args.resume)
    print(f"\n📁 Log directory: {log_dir}")

    save_experiment_config(config, log_dir)
    print(f"💾 Config saved to: {log_dir / 'config.json'}")

    # Create LMs based on configured model
    llm_model = config["llm_model"]
    if llm_model.startswith("openrouter/"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    task_lm = dspy.LM(
        model=llm_model,
        temperature=1.0,
        max_tokens=32000,
        cache=False,
        api_key=api_key,
    )

    # Create adapter
    adapter = DspyAdapter(
        task_lm=task_lm,
        metric_fn=metric_fn,
        num_threads=64,
        reflection_lm=llm_model,
        add_format_failure_as_feedback=True,
        rng=random.Random(config["seed"]),
    )
    fitness_fn = create_fitness_fn(adapter)

    # Configure GEPA
    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(log_dir),
            seed=config["seed"],
            max_metric_calls=4000,
            track_best_outputs=True,
            use_cloudpickle=True,
            parallel=True,
            max_workers=64,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=config["reflection_minibatch_size"],
            reflection_lm=llm_model,
            skip_perfect_score=False,
        ),
        tracking=TrackingConfig(
            use_wandb=False,
            wandb_api_key=None,
        ),
    )

    seed_candidate = {"program": program_src}

    train_set, val_set, test_set = load_data()

    print("\n🚀 Starting GEPA optimization...\n")

    # Run optimization
    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        dataset=train_set,
        valset=val_set,
        config=gepa_config,
        objective="Optimize the dspy agent program to solve ARC-AGI tasks effectively.",
        background=BACKGROUND
    )

    print("\n" + "=" * 80)
    print("Step 9: Results")
    print("=" * 80)

    print("\n✅ GEPA optimization complete!")

    print("Base evaluation:")
    base_results = adapter.evaluate(test_set, seed_candidate, capture_traces=False)
    base_score = np.mean(base_results.scores)
    print(f"   Base results: {base_score}")

    best_program = result.best_candidate["program"]

    print(f"✅ Best program: {best_program}")
    print(f"   Code length: {len(best_program)} characters")
    test_results = adapter.evaluate(
        test_set, result.best_candidate, capture_traces=False
    )
    test_score = np.mean(test_results.scores)
    print(f"   Test results: {test_score}")

    print(f"   Improvement: {test_score - base_score}")
    print(f"   Improvement percentage: {(test_score - base_score) / base_score * 100}%")
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"📁 Log directory: {log_dir}")
    print(f"__LOG_DIR__:{log_dir}")

    return result


if __name__ == "__main__":
    main()
