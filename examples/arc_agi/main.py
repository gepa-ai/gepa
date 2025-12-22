import os
import warnings
import dspy
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)
import argparse

from config import parse_arguments, get_log_directory
from data import load_data
from llm import create_reflection_lm, REFLECTION_PROMPT
from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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
    """Validate matrix format and correctness."""
    if not isinstance(matrix, list) or len(matrix) == 0:
        return (
            False,
            f"Matrix must be a non-empty List[List[int]]. Expected: {gold_matrix}",
        )

    n, m = len(matrix), len(matrix[0])
    if m == 0:
        return False, f"Matrix must have at least one column. Expected: {gold_matrix}"

    # Validate structure
    for i in range(n):
        if not isinstance(matrix[i], list):
            return False, f"Row {i} must be a List[int]. Expected: {gold_matrix}"
        if len(matrix[i]) != m:
            return (
                False,
                f"Matrix is staggered (row 0 has {m} cols, row {i} has {len(matrix[i])}). Expected: {gold_matrix}",
            )
        for j in range(m):
            if not isinstance(matrix[i][j], int):
                return (
                    False,
                    f"Element [{i}][{j}] must be int, found {type(matrix[i][j])}. Expected: {gold_matrix}",
                )

    # Check dimensions match gold
    gold_n, gold_m = len(gold_matrix), len(gold_matrix[0])
    if (n, m) != (gold_n, gold_m):
        return (
            False,
            f"Dimensions {n}x{m} don't match expected {gold_n}x{gold_m}. Expected: {gold_matrix}",
        )

    # Check correctness
    wrong_indices = [
        (i, j) for i in range(n) for j in range(m) if matrix[i][j] != gold_matrix[i][j]
    ]
    if not wrong_indices:
        return True, f"Correct. Expected: {gold_matrix}"

    if len(wrong_indices) < 10:
        return False, f"Incorrect indices: {wrong_indices}. Expected: {gold_matrix}"
    return False, f"Matrix is incorrect. Expected: {gold_matrix}"


def metric_fn(example, pred, trace=None):
    """Evaluate prediction against gold outputs."""
    # Handle both dict and dspy.Prediction/Example formats
    if isinstance(pred, dict):
        pred_outputs = pred.get("test_outputs")
    else:
        pred_outputs = getattr(pred, "test_outputs", None)

    if isinstance(example, dict):
        gold_outputs = example["test_outputs"]
        test_inputs = example["test_inputs"]
    else:
        gold_outputs = example.test_outputs
        test_inputs = example.test_inputs

    if not isinstance(pred_outputs, list):
        return dspy.Prediction(
            score=0,
            feedback=f"Response must be a List[List[List[int]]]. Expected: {gold_outputs}",
        )

    if len(test_inputs) != len(pred_outputs):
        return dspy.Prediction(
            score=0,
            feedback=f"Output count ({len(pred_outputs)}) must match input count ({len(test_inputs)}). Expected: {gold_outputs}",
        )

    results = [
        is_valid_matrix(pred_out, gold_out)
        for pred_out, gold_out in zip(pred_outputs, gold_outputs, strict=False)
    ]

    score = sum(valid for valid, _ in results) / len(results)
    feedback = "\n".join(f"Test input {i}: {fb}" for i, (_, fb) in enumerate(results))

    return dspy.Prediction(score=score, feedback=feedback)


def _create_adapter(task_lm, reflection_lm):
    """Create DspyAdapter with task and reflection LMs."""
    return DspyAdapter(
        task_lm=task_lm,
        metric_fn=metric_fn,
        num_threads=32,
        reflection_lm=lambda x: reflection_lm(x)[0],
    )


# Global adapter - will be set in main()
_adapter = None


def _extract_feedback(traj, score):
    """Extract feedback from trajectory."""
    feedback = ""

    if isinstance(traj, dict):
        if "score" in traj and hasattr(traj["score"], "feedback"):
            feedback = traj["score"].feedback
        elif traj.get("error"):
            error_msg = traj["error"]
            feedback = f"Execution failed: {error_msg}"
            if "is not defined" in str(error_msg):
                feedback += ". NameError: ensure all functions are defined before use."
            elif "token" in str(error_msg).lower():
                feedback += ". Token limit exceeded: make reasoning more concise."

    if score == 0 and not feedback:
        feedback = (
            "Score is 0. Check for runtime errors, token limits, or malformed output."
        )

    return feedback


def _add_warnings_to_feedback(feedback, warning_messages):
    """Add warning messages to feedback string."""
    if warning_messages:
        warning_text = "\n".join(warning_messages)
        feedback += f"\n\nDSPy warnings:\n{warning_text}"
    return feedback


def _sanitize_trace(trace):
    """Remove signature objects from trace for pickling."""
    if trace is None:
        return None
    return [(inputs, outputs) for _, inputs, outputs in trace]


def _create_error_results(
    program, batch, error_msg, error_type, warning_messages, traceback_str=None
):
    """Create error results for all examples in batch."""
    error_feedbacks = {
        "evaluation_error": f"Program evaluation failed: {error_msg}. Check code structure.",
        "trajectories_none": "Program failed to execute. Check for undefined functions, syntax errors, or missing imports.",
    }
    feedback = error_feedbacks.get(error_type, f"Error: {error_msg}")
    feedback = _add_warnings_to_feedback(feedback, warning_messages)

    results = []
    for _ in batch:
        side_info = {
            "score": 0.0,
            "error": error_msg,
            "program": program,
            "feedback": feedback,
            "warnings": warning_messages,
        }
        if traceback_str:
            side_info["error_traceback"] = traceback_str

        output = {"error": error_type, "program": program}
        results.append((0.0, output, side_info))
    return results


def _create_result_item(score, traj, program, warning_messages):
    """Create a single result item from trajectory."""
    if not isinstance(traj, dict):
        feedback = _add_warnings_to_feedback(
            f"Invalid trajectory format: {type(traj).__name__}", warning_messages
        )
        return (
            score,
            {"error": "invalid_trajectory", "program": program, "score": score},
            {
                "score": score,
                "feedback": feedback,
                "program": program,
                "warnings": warning_messages,
            },
        )

    feedback = _add_warnings_to_feedback(
        _extract_feedback(traj, score), warning_messages
    )
    example = traj.get("example")
    prediction = traj.get("prediction")

    side_info = {
        "score": score,
        "reasoning": traj.get("reasoning"),
        "feedback": feedback,
        "trace": _sanitize_trace(traj.get("trace")),
        "prediction": prediction,
        "input": example,
        "training_examples": getattr(example, "training_examples", None)
        if example
        else None,
        "test_inputs": getattr(example, "test_inputs", None) if example else None,
        "test_outputs": getattr(example, "test_outputs", None) if example else None,
        "program": program,
        "warnings": warning_messages,
    }

    # Convert prediction to dict if it's a dspy.Prediction object
    if prediction and hasattr(prediction, "__dict__"):
        try:
            prediction = {**prediction}
        except (TypeError, ValueError):
            prediction = str(prediction)

    return (
        score,
        {"program": program, "score": score, "prediction": prediction},
        side_info,
    )


def fitness_fn(candidate, batch):
    """Evaluate candidate program on batch and return results."""
    global _adapter
    program = candidate["program"]
    print(f"🔍 Evaluating program: {program[:100]}...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            eval_batch = _adapter.evaluate(batch, candidate, capture_traces=True)
        except Exception as e:
            import traceback

            warning_messages = [str(warning.message) for warning in w]
            print(f"❌ Evaluation error: {str(e)}")
            return _create_error_results(
                program,
                batch,
                str(e),
                "evaluation_error",
                warning_messages,
                traceback.format_exc(),
            )

    warning_messages = [str(warning.message) for warning in w]

    # Handle evaluation failures
    if eval_batch.trajectories is None:
        print("⚠️  All examples failed - trajectories are None")
        return _create_error_results(
            program, batch, "All examples failed", "trajectories_none", warning_messages
        )

    if isinstance(eval_batch.trajectories, str):
        print(f"⚠️  Program build failed: {eval_batch.trajectories[:200]}...")
        return _create_error_results(
            program,
            batch,
            f"Program build failed: {eval_batch.trajectories}",
            "build_failed",
            warning_messages,
        )

    # Process successful evaluations
    results = [
        _create_result_item(score, traj, program, warning_messages)
        for score, traj in zip(eval_batch.scores, eval_batch.trajectories)
    ]

    # Ensure we always return exactly len(batch) results
    # If some evaluations failed silently, pad with error results
    if len(results) < len(batch):
        num_missing = len(batch) - len(results)
        print(f"⚠️  Missing {num_missing} results, padding with errors")
        error_results = _create_error_results(
            program,
            batch[:num_missing],
            "Evaluation failed silently (likely malformed LLM response)",
            "silent_failure",
            warning_messages,
        )
        results.extend(error_results)

    return results


def main():
    """Main entry point for ARC-AGI GEPA optimization."""

    print("=" * 80)
    print("GEPA ARC-AGI Solver Optimization")
    print("=" * 80)

    args = parse_arguments()
    log_dir = get_log_directory()
    print(f"\n📁 Log directory: {log_dir}")

    # Create LMs
    llm_model = "openai/gpt-5"  # Fixed model
    task_lm = dspy.LM(
        model=llm_model,
        max_tokens=32000,
        api_key=OPENAI_API_KEY,
        seed=args.seed,
    )
    reflection_lm = create_reflection_lm(
        llm_model,
        api_key=OPENAI_API_KEY,
        seed=args.seed,
    )

    # Create adapter and set global reference for fitness_fn
    global _adapter
    _adapter = _create_adapter(task_lm, reflection_lm)

    # Configure GEPA
    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(log_dir),
            seed=args.seed,
            max_metric_calls=args.max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
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

    # print("Base evaluation:")
    # base_results = _adapter.evaluate(test_set, seed_candidate, capture_traces=True)
    # print(f"   Base results: {base_results}")

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

    best_program = result.best_candidate["program"]
    print(f"✅ Best program: {best_program}")
    print(f"   Code length: {len(best_program)} characters")
    test_results = _adapter.evaluate(
        test_set, result.best_candidate, capture_traces=True
    )
    print(f"   Test results: {test_results}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"📁 Log directory: {log_dir}")
    print(f"__LOG_DIR__:{log_dir}")

    return result


if __name__ == "__main__":
    main()
