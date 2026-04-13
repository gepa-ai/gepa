"""Tests for DspyAdapter (full-program evolution adapter).

Covers two bugs:
1. evaluate() returned EvaluationBatch(outputs=None, ...) on build failure,
   crashing downstream zip() in cached_evaluate_full.
2. reflection_lm was typed as dspy.LM but must conform to the LanguageModel
   protocol (callable returning str, not list[str]).
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="dspy is not installed — skipping DspyAdapter tests")

from unittest.mock import MagicMock, patch

import dspy
from dspy.primitives import Example

from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter
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
