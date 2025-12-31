import os
import numpy as np
import dspy
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)
from experiments.arc_agi.config import (
    parse_arguments,
    get_experiment_config,
    get_log_directory,
)
from experiments.arc_agi.data import load_data
from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter
from src.experiment_io import save_experiment_config

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

REFLECTION_PROMPT = """
You are optimizing a DSPy program to solve ARC-AGI tasks.

The current program that solves ARC-AGI tasks is:
```
<curr_param>
```

Below is evaluation data showing how this parameter value performed across multiple test cases. The data contains performance metrics, diagnostic information, and other relevant details from the evaluation:
```
<side_info>
```

Your task is to propose a new, improved parameter value that can be used as a drop-in replacement for the current one.

Carefully analyze all the evaluation data provided above. Look for patterns that indicate what works and what doesn't. Pay special attention to:
- Performance metrics and how they correlate with parameter behavior
- Recurring issues, errors, or failure patterns across multiple test cases
- Successful patterns or behaviors that should be preserved or enhanced
- Any domain-specific requirements, constraints, or factual information revealed in the evaluation data
- Specific technical details that are crucial for understanding the parameter's role

Analyze the evaluation data and propose an improved DSPy program that addresses failures and solves the tasks better.

The code must be a valid, self-contained Python script with all necessary imports (e.g. import dspy) and definitions.

Provide complete, executable Python code within ``` blocks.
"""

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

    if task_inputs == "empty":
        print("task inputs are empty")
    if gold_task_outputs == "empty":
        print("gold_task_outputs are empty")
    if pred_task_outputs == "empty":
        print("pred task outputs are empty")

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
    for i, (input, gold_output, pred_output) in enumerate(
        zip(task_inputs, gold_task_outputs, pred_task_outputs, strict=False)
    ):
        is_valid, feedback = is_valid_matrix(pred_output, gold_output)
        valids.append(is_valid)
        feedbacks.append(f"Feedback on test input {i}: {feedback}")

    score = sum(valids) / len(valids)
    feedback_text = "\n".join(feedbacks)
    return dspy.Prediction(score=score, feedback=feedback_text)


def _create_error_results(program, batch, error_msg):
    """Create error results for all examples in batch."""
    feedback = error_msg

    results = []
    for _ in batch:
        side_info = {
            "score": 0.0,
            "error": error_msg,
            "program": program,
            "feedback": feedback,
        }

        output = {"error": error_msg, "program": program}
        results.append((0.0, output, side_info))
    return results


def _create_result_item(traj, program):
    """Create a single result item from trajectory."""
    metric_result = traj.get("score")
    score = metric_result.get("score")
    feedback = metric_result.get("feedback")

    prediction = traj.get("prediction")
    model_answer = prediction.get("test_outputs")

    rollout_output = {
        "program": program,
        "model_answer": model_answer,
        "score": score,
    }

    side_info = {
        "score": score,
        "input": traj.get("example"),
        "reasoning": prediction.get("reasoning"),
        "feedback": feedback,
        "output": model_answer,
    }

    return (
        score,
        rollout_output,
        side_info,
    )


def create_fitness_fn(adapter):
    """Create fitness function with adapter in closure."""

    def fitness_fn(candidate, batch):
        """Evaluate candidate program on batch and return results."""
        program = candidate["program"]
        try:
            eval_batch = adapter.evaluate(batch, candidate, capture_traces=True)
        except Exception as e:
            print(f"Error evaluating candidate: {e}")
            return _create_error_results(
                program,
                batch,
                str(e),
            )

        if isinstance(eval_batch.trajectories, str):
            # program error
            error_msg = (
                f"All examples failed. Program error message: {eval_batch.trajectories}"
            )
            print("⚠️  " + error_msg[:200])
            return _create_error_results(
                program,
                batch,
                error_msg,
            )

        if len(eval_batch.trajectories) == 0:
            # dspy error caused by the program
            error_msg = (
                "All examples failed - likely a serialization error. "
                "Do NOT pass lambdas, functions, classes, or type objects as arguments to dspy modules. "
                "Keep all logic inside class methods. Only pass serializable data (strings, numbers, lists, dicts)."
            )
            return _create_error_results(
                program,
                batch,
                error_msg,
            )

        # Process successful evaluations with no errors
        results = [
            _create_result_item(traj, program) for traj in eval_batch.trajectories
        ]

        return results

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
    print(f"💾 Config saved to: {log_dir / 'config.yaml'}")

    # Create LMs
    task_lm = dspy.LM(
        model=config["llm_model"],
        temperature=1.0,
        max_tokens=32000,
        api_key=OPENAI_API_KEY,
        seed=config["seed"],
    )
    reflection_lm = dspy.LM(
        model=config["llm_model"],
        temperature=1.0,
        api_key=OPENAI_API_KEY,
        max_tokens=32000,
        seed=config["seed"],
    )

    # Create adapter
    adapter = DspyAdapter(
        task_lm=task_lm,
        metric_fn=metric_fn,
        num_threads=64,
        reflection_lm=lambda x: reflection_lm(x)[0],
        add_format_failure_as_feedback=True,
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
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=config["reflection_minibatch_size"],
            reflection_prompt_template=REFLECTION_PROMPT,
            reflection_lm=reflection_lm,
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
    )

    print("\n" + "=" * 80)
    print("Step 9: Results")
    print("=" * 80)

    print("\n✅ GEPA optimization complete!")

    print("Base evaluation:")
    base_results = adapter.evaluate(test_set, seed_candidate, capture_traces=True)
    base_score = np.mean(base_results.scores)
    print(f"   Base results: {base_score}")

    best_program = result.best_candidate["program"]

    print(f"✅ Best program: {best_program}")
    print(f"   Code length: {len(best_program)} characters")
    test_results = adapter.evaluate(
        test_set, result.best_candidate, capture_traces=True
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
