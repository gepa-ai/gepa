from unittest.mock import MagicMock

import pytest

from glean_gepa.al_adapter import Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.reflection_sampling import (
    deduplicate_reflective_examples,
    is_within_hamming_distance,
    strip_stdout_sections,
)
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.single_model_adapter import SingleModelAdapter


def test_hamming_distance_supports_small_variations_and_length_differences():
    assert is_within_hamming_distance("error-100", "error-101", 1)
    assert is_within_hamming_distance("abc", "ab", 1)
    assert not is_within_hamming_distance("error-100", "error-999", 1)


def test_strip_stdout_sections_preserves_stderr_and_surrounding_error_text():
    error = "command failed\nstdout:\nnoisy command output\nstderr:\npermission denied"

    assert strip_stdout_sections(error) == "command failed\nstderr:\npermission denied"


def test_deduplicate_reflective_examples_keeps_first_error_cluster():
    examples = [
        {"Inputs": {"entry_id": "first"}, "Execution Errors": ["error-100"]},
        {"Inputs": {"entry_id": "near"}, "Execution Errors": ["error-101"]},
        {"Inputs": {"entry_id": "different"}, "Execution Errors": ["error-999"]},
    ]

    result = deduplicate_reflective_examples(examples, k=1)

    assert [example["Inputs"]["entry_id"] for example in result] == ["first", "different"]


def test_deduplicate_reflective_examples_logs_every_dropped_sample():
    examples = [
        {"Inputs": {"entry_id": "first"}, "Execution Errors": ["error-100"]},
        {"Inputs": {"entry_id": "near-one"}, "Execution Errors": ["error-101"]},
        {"Inputs": {"entry_id": "near-two"}, "Execution Errors": ["error-110"]},
    ]
    logs: list[str] = []

    deduplicate_reflective_examples(examples, k=1, log=logs.append)

    assert len(logs) == 2
    assert "entry_id='near-one'" in logs[0]
    assert "Hamming distance 1 <= k=1" in logs[0]
    assert "entry_id='near-two'" in logs[1]


def test_deduplicate_reflective_examples_keeps_entries_without_errors():
    examples = [
        {"Inputs": {"entry_id": "first"}, "Execution Errors": []},
        {"Inputs": {"entry_id": "second"}, "Execution Errors": []},
    ]

    assert deduplicate_reflective_examples(examples, k=10) == examples


def test_hamming_distance_rejects_negative_k():
    with pytest.raises(ValueError, match="non-negative"):
        is_within_hamming_distance("a", "b", -1)


def test_single_model_adapter_all_mode_deduplicates_errors_before_prompting():
    adapter = SingleModelAdapter(
        runner=MagicMock(),
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        student_model="fast",
        bigquery_client=MagicMock(),
    )

    def trajectory(entry_id: str, error: str, score: float):
        objective_scores = {SHELL_SUCCESS_OBJECTIVE: score}
        output = {
            "entry_id": entry_id,
            "deployment_id": "scio-prod",
            "query": entry_id,
            "student_tool_errors": 1,
            "shell_error_messages": [error],
        }
        return {
            "data": {"eval_set_name": "small-eval-set"},
            "output": output,
            "score": score,
            "objective_scores": objective_scores,
        }

    trajectories = [
        trajectory("first", "error-100\nstdout:\nnoisy output", 0.1),
        trajectory("near", "error-101", 0.2),
        trajectory("different", "error-999", 0.3),
    ]
    batch = GleanEvaluationBatch(
        outputs=[item["output"] for item in trajectories],
        scores=[item["score"] for item in trajectories],
        trajectories=trajectories,
        objective_scores=[item["objective_scores"] for item in trajectories],
        summary=None,
    )

    examples = adapter.make_reflective_dataset(
        {"WRITING_CODE": "current instructions"},
        batch,
        ["WRITING_CODE"],
        k=None,
        error_hamming_distance_k=1,
    )["WRITING_CODE"]

    assert [example["Inputs"]["entry_id"] for example in examples] == ["first", "different"]
    assert all("stdout" not in str(example).lower() for example in examples)
    assert all("noisy output" not in str(example).lower() for example in examples)
