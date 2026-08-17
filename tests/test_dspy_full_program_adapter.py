"""Tests for DspyAdapter (full-program evolution adapter).

Covers:
1. evaluate() returned EvaluationBatch(outputs=None, ...) on build failure,
   crashing downstream zip() in cached_evaluate_full.
2. reflection_lm was typed as dspy.LM but must conform to the LanguageModel
   protocol (callable returning str, not list[str]).
3. make_reflective_dataset must collapse repeated calls to the same predictor
   (issue 97) without dropping distinct predictors in a multi-module program.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="dspy is not installed — skipping DspyAdapter tests")

from unittest.mock import MagicMock, patch

import dspy
from dspy.primitives import Example

from gepa.adapters.dspy_full_program_adapter.full_program_adapter import (
    DspyAdapter,
    _select_trace_instances_for_reflection,
)
from gepa.core.adapter import EvaluationBatch
from gepa.proposer.reflective_mutation.base import LanguageModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(reflection_lm=None):
    """Build a DspyAdapter with mocked dependencies."""
    task_lm = MagicMock(spec=dspy.LM)
    metric_fn = MagicMock(return_value=1.0)
    if reflection_lm is None:
        reflection_lm = MagicMock(spec=LanguageModel)
    return DspyAdapter(
        task_lm=task_lm,
        metric_fn=metric_fn,
        reflection_lm=reflection_lm,
        failure_score=0.0,
        num_threads=1,
    )


def _make_batch(n=3):
    """Create a minimal batch of DSPy Examples."""
    return [Example(question=f"q{i}").with_inputs("question") for i in range(n)]


# ---------------------------------------------------------------------------
# Bug 1: outputs must be a list, even on build failure
# ---------------------------------------------------------------------------


class TestEvaluateOutputsOnBuildFailure:
    """When the candidate program fails to build, evaluate() must still
    return an EvaluationBatch with a list of outputs (not None)."""

    def test_outputs_is_list_on_syntax_error(self):
        adapter = _make_adapter()
        candidate = {"program": "def foo(  # syntax error"}
        batch = _make_batch(4)

        result = adapter.evaluate(batch, candidate, capture_traces=False)

        assert isinstance(result, EvaluationBatch)
        assert isinstance(result.outputs, list), f"outputs should be a list, got {type(result.outputs)}"
        assert len(result.outputs) == len(batch)
        assert len(result.scores) == len(batch)
        assert all(s == 0.0 for s in result.scores)

    def test_outputs_is_list_on_missing_program_object(self):
        adapter = _make_adapter()
        # Valid Python but doesn't define `program`
        candidate = {"program": "x = 42"}
        batch = _make_batch(2)

        result = adapter.evaluate(batch, candidate, capture_traces=False)

        assert isinstance(result.outputs, list)
        assert len(result.outputs) == len(batch)

    def test_outputs_is_list_on_runtime_error(self):
        adapter = _make_adapter()
        candidate = {"program": "raise RuntimeError('boom')"}
        batch = _make_batch(5)

        result = adapter.evaluate(batch, candidate, capture_traces=False)

        assert isinstance(result.outputs, list)
        assert len(result.outputs) == len(batch)

    def test_outputs_zippable_with_example_ids(self):
        """Reproduce the exact crash from cached_evaluate_full:
        dict(zip(example_ids, outputs)) must not raise."""
        adapter = _make_adapter()
        candidate = {"program": "def foo(  # syntax error"}
        batch = _make_batch(3)

        result = adapter.evaluate(batch, candidate, capture_traces=False)
        example_ids = list(range(len(batch)))

        # This is the exact operation that crashed before the fix
        outputs_by_id = dict(zip(example_ids, result.outputs, strict=False))
        scores_by_id = dict(zip(example_ids, result.scores, strict=False))

        assert len(outputs_by_id) == len(batch)
        assert len(scores_by_id) == len(batch)


# ---------------------------------------------------------------------------
# Bug 2: reflection_lm must conform to LanguageModel protocol
# ---------------------------------------------------------------------------


class TestReflectionLmProtocol:
    """The reflection_lm parameter should accept any callable that returns str,
    not require a dspy.LM specifically."""

    def test_lambda_wrapper_accepted(self):
        """A lambda wrapping dspy.LM (as shown in GEPA's example notebook)
        should be accepted as reflection_lm."""
        mock_dspy_lm = MagicMock(spec=dspy.LM)
        mock_dspy_lm.return_value = ["response text"]
        wrapped = lambda x: mock_dspy_lm(x)[0]

        # Should not raise
        adapter = _make_adapter(reflection_lm=wrapped)
        assert adapter.reflection_lm is wrapped

    def test_plain_callable_accepted(self):
        """Any callable (str) -> str should work as reflection_lm."""

        def my_lm(prompt):
            return "generated response"

        adapter = _make_adapter(reflection_lm=my_lm)
        assert adapter.reflection_lm is my_lm

    def test_propose_new_texts_calls_lm_correctly(self):
        """propose_new_texts should pass the prompt to reflection_lm and use
        the str return value (not a list)."""
        mock_lm = MagicMock(return_value="<new_program>\nimport dspy\nprogram = dspy.Predict('q -> a')\n</new_program>")
        adapter = _make_adapter(reflection_lm=mock_lm)

        candidate = {"program": "import dspy\nprogram = dspy.Predict('q -> a')"}
        reflective_dataset = {"program": [{"input": "q1", "output": "a1", "score": 0.5}]}

        # The proposal signature will call lm(prompt) and expect a str back.
        # We mock the signature's run method to verify the LM is called.
        with patch(
            "gepa.adapters.dspy_full_program_adapter.dspy_program_proposal_signature.DSPyProgramProposalSignature.run",
            return_value={"new_program": "import dspy\nprogram = dspy.Predict('q -> a')"},
        ) as mock_run:
            result = adapter.propose_new_texts(candidate, reflective_dataset, ["program"])
            mock_run.assert_called_once_with(
                lm=mock_lm,
                input_dict={
                    "curr_program": candidate["program"],
                    "dataset_with_feedback": reflective_dataset["program"],
                },
            )
            assert "program" in result


# ---------------------------------------------------------------------------
# Issue #97: avoid redundant cumulative trace context
# ---------------------------------------------------------------------------


class TestReflectiveDatasetTraceSelection:
    def test_keeps_only_final_trace_when_no_failure(self):
        """Normal cumulative traces should contribute only their final entry."""
        adapter = _make_adapter()

        predictor = MagicMock()
        predictor.signature.equals.return_value = True

        proposed_program = MagicMock()
        proposed_program.named_predictors.return_value = [("react", predictor)]

        adapter.build_program = MagicMock(return_value=(proposed_program, None))

        trace = [
            (predictor, {"step": "1"}, {"thought": "first"}),
            (predictor, {"step": "2"}, {"thought": "second"}),
            (predictor, {"step": "3"}, {"answer": "final"}),
        ]

        example = Example(question="What is 2+2?").with_inputs("question")
        eval_batch = MagicMock()
        eval_batch.trajectories = [
            {
                "trace": trace,
                "example": example,
                "prediction": {"answer": "4"},
                "score": 1.0,
            }
        ]

        result = adapter.make_reflective_dataset(
            candidate={"program": "dummy"},
            eval_batch=eval_batch,
            components_to_update=["program"],
        )

        program_trace = result["program"][0]["Program Trace"]

        assert len(program_trace) == 1
        assert program_trace[0]["Generated Outputs"] == {"answer": "final"}


def _make_predictor():
    """A mock predictor whose signature.equals matches only this predictor."""
    predictor = MagicMock()
    predictor.signature.equals.side_effect = lambda other: other is predictor.signature
    return predictor


def _program_trace(adapter, trace, named_predictors):
    proposed_program = MagicMock()
    proposed_program.named_predictors.return_value = named_predictors
    adapter.build_program = MagicMock(return_value=(proposed_program, None))

    example = Example(question="What is 2+2?").with_inputs("question")
    eval_batch = MagicMock()
    eval_batch.trajectories = [
        {
            "trace": trace,
            "example": example,
            "prediction": {"answer": "4"},
            "score": 1.0,
        }
    ]
    result = adapter.make_reflective_dataset(
        candidate={"program": "dummy"},
        eval_batch=eval_batch,
        components_to_update=["program"],
    )
    return result["program"][0]["Program Trace"]


class TestSelectTraceInstancesForReflection:
    def test_repeated_calls_keep_last_per_predictor(self):
        react = object()
        extract = object()
        t0 = "thought_0: add\ntool_name_0: calculator\nobservation_0: 4"
        t1 = t0 + "\nthought_1: done\ntool_name_1: finish\nobservation_1: Completed."
        trace = [
            (
                react,
                {"question": "What is 2+2?", "trajectory": ""},
                {"next_thought": "add", "next_tool_name": "calculator"},
            ),
            (
                react,
                {"question": "What is 2+2?", "trajectory": t0},
                {"next_thought": "done", "next_tool_name": "finish"},
            ),
            (extract, {"question": "What is 2+2?", "trajectory": t1}, {"answer": "4"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert len(selected) == 2
        assert selected[0] is trace[1]
        assert selected[1] is trace[2]
        assert t0 in selected[0][1]["trajectory"]
        assert t0 in selected[1][1]["trajectory"]
        assert "thought_1: done" in selected[1][1]["trajectory"]

    def test_distinct_predictors_are_all_kept(self):
        reasoner = object()
        extractor = object()
        trace = [
            (reasoner, {"question": "q"}, {"reasoning": "subtract 3", "answer": "5"}),
            (extractor, {"question": "q", "reasoning": "subtract 3"}, {"answer": "4"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert selected[0][2]["answer"] == "5"
        assert selected[1][2]["answer"] == "4"

    def test_failed_prediction_keeps_the_failed_call_not_the_last_success(self):
        from dspy.teleprompt.bootstrap_trace import FailedPrediction

        predictor = object()
        failed = FailedPrediction(completion_text="RAW")
        trace = [
            (predictor, {"step": "1"}, {"thought": "first"}),
            (predictor, {"step": "2"}, failed),
            (predictor, {"step": "3"}, {"answer": "final"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == [trace[1]]
        assert selected[0][2] is failed


class TestReflectiveDatasetKeepsDistinctPredictors:
    def test_same_predictor_cumulative_trajectory_keeps_last_entry(self):
        """Issue 97 shape: later inputs contain earlier trajectory; keep one entry."""
        adapter = _make_adapter()
        react = _make_predictor()
        t0 = "thought_0: add\ntool_name_0: calculator\nobservation_0: 4"
        t1 = t0 + "\nthought_1: done\ntool_name_1: finish\nobservation_1: Completed."
        trace = [
            (react, {"question": "What is 2+2?", "trajectory": ""}, {"next_thought": "add"}),
            (react, {"question": "What is 2+2?", "trajectory": t0}, {"next_thought": "use result"}),
            (react, {"question": "What is 2+2?", "trajectory": t1}, {"next_thought": "done", "answer": "4"}),
        ]

        program_trace = _program_trace(adapter, trace, [("react", react)])

        assert len(program_trace) == 1
        assert program_trace[0]["Called Module"] == "react"
        assert program_trace[0]["Generated Outputs"]["answer"] == "4"
        assert t0 in program_trace[0]["Inputs"]["trajectory"]
        assert "thought_1: done" in program_trace[0]["Inputs"]["trajectory"]

    def test_react_keeps_last_react_call_and_extract(self):
        """Last-entry-only would drop extract's sibling react call, or keep all three.

        The correct trace is last react plus extract: two modules, with the full
        trajectory still present on the kept extract inputs.
        """
        adapter = _make_adapter()
        react = _make_predictor()
        extract = _make_predictor()
        t0 = "thought_0: add\ntool_name_0: calculator\nobservation_0: 4"
        t1 = t0 + "\nthought_1: done\ntool_name_1: finish\nobservation_1: Completed."
        trace = [
            (
                react,
                {"question": "What is 2+2?", "trajectory": ""},
                {"next_thought": "add", "next_tool_name": "calculator"},
            ),
            (
                react,
                {"question": "What is 2+2?", "trajectory": t0},
                {"next_thought": "done", "next_tool_name": "finish"},
            ),
            (
                extract,
                {"question": "What is 2+2?", "trajectory": t1},
                {"answer": "4"},
            ),
        ]

        program_trace = _program_trace(
            adapter,
            trace,
            [("react.react", react), ("react.extract.predict", extract)],
        )

        assert [entry["Called Module"] for entry in program_trace] == [
            "react.react",
            "react.extract.predict",
        ]
        assert program_trace[0]["Generated Outputs"]["next_tool_name"] == "finish"
        assert t0 in program_trace[0]["Inputs"]["trajectory"]
        assert program_trace[1]["Generated Outputs"]["answer"] == "4"
        assert "thought_1: done" in program_trace[1]["Inputs"]["trajectory"]

    def test_two_module_pipeline_keeps_both_predictors(self):
        """MATH-tutorial shape: reasoner then extractor, including a disagreed answer."""
        adapter = _make_adapter()
        reasoner = _make_predictor()
        extractor = _make_predictor()
        trace = [
            (reasoner, {"question": "If x+3=7, what is x?"}, {"reasoning": "subtract 3", "answer": "5"}),
            (
                extractor,
                {"question": "If x+3=7, what is x?", "reasoning": "subtract 3"},
                {"answer": "4"},
            ),
        ]

        program_trace = _program_trace(
            adapter,
            trace,
            [("reasoner.predict", reasoner), ("extractor", extractor)],
        )

        assert [entry["Called Module"] for entry in program_trace] == ["reasoner.predict", "extractor"]
        assert program_trace[0]["Generated Outputs"]["answer"] == "5"
        assert program_trace[1]["Generated Outputs"]["answer"] == "4"

    def test_failed_prediction_in_the_middle_is_kept(self):
        from dspy.teleprompt.bootstrap_trace import FailedPrediction

        adapter = _make_adapter()
        predictor = _make_predictor()
        failed = FailedPrediction(completion_text="RAW")
        trace = [
            (predictor, {"step": "1"}, {"thought": "first"}),
            (predictor, {"step": "2"}, failed),
            (predictor, {"step": "3"}, {"answer": "final"}),
        ]

        program_trace = _program_trace(adapter, trace, [("react", predictor)])

        assert len(program_trace) == 1
        assert program_trace[0]["Called Module"] == "react"
        assert "RAW" in program_trace[0]["Generated Outputs"]

    def test_independent_modules_keep_a_side_call_that_is_not_in_the_last_inputs(self):
        adapter = _make_adapter()
        writer = _make_predictor()
        critic = _make_predictor()
        trace = [
            (writer, {"question": "Write a haiku about rain."}, {"poem": "soft rain falling now"}),
            (critic, {"rubric": "score originality 1-5"}, {"score": "5"}),
        ]

        program_trace = _program_trace(
            adapter,
            trace,
            [("writer", writer), ("critic", critic)],
        )

        assert [entry["Called Module"] for entry in program_trace] == ["writer", "critic"]
        assert program_trace[0]["Generated Outputs"]["poem"] == "soft rain falling now"
        assert "poem" not in program_trace[1]["Inputs"]
