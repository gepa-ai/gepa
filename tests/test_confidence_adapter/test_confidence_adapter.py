# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Unit tests for ConfidenceAdapter: evaluate() and make_reflective_dataset().

All tests mock ``llm_structured_confidence.extract_logprobs`` to avoid
real LLM calls.  Evaluate tests use **callable models** so they don't
depend on litellm being installed.

Requires ``pip install "gepa[confidence]"`` -- the entire module is
skipped when ``llm_structured_confidence`` is not installed.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("llm_structured_confidence", reason="requires gepa[confidence] extra")

from gepa.adapters.confidence_adapter.confidence_adapter import (  # noqa: E402
    ConfidenceAdapter,
    ConfidenceDataInst,
    _build_feedback,
    _extract_answer_from_json,
)
from gepa.adapters.confidence_adapter.scoring import (  # noqa: E402
    LinearBlendScoring,
    ThresholdScoring,
)


# ---------------------------------------------------------------------------
# Helpers for building mock LLM responses
# ---------------------------------------------------------------------------


def _make_alt(token: str, probability: float, resolved_value: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(token=token, probability=probability, resolved_value=resolved_value)


def _make_field_logprob(
    joint_logprob: float,
    top_logprobs: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        joint_logprob=joint_logprob,
        top_logprobs=top_logprobs or [],
    )


def _make_entry(field_logprob: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(field_logprob=field_logprob)


def _make_litellm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_callable_model(*contents: str):
    """Return a callable model that yields mock responses in order."""
    responses = iter([_make_litellm_response(c) for c in contents])

    def model_fn(messages):
        return next(responses)

    return model_fn


def _sample_batch() -> list[ConfidenceDataInst]:
    return [
        {"input": "UBER EATS payment", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        {
            "input": "LIGHT electricity bill",
            "additional_context": {"merchant_type": "utility"},
            "answer": "Bills/Electricity",
        },
    ]


def _sample_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "category_name": {
                        "type": "string",
                        "enum": ["Food & Drinks/Restaurants", "Bills/Electricity", "Shopping/Electronics"],
                    }
                },
                "required": ["category_name"],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# _extract_answer_from_json
# ---------------------------------------------------------------------------


class TestExtractAnswerFromJson:
    def test_simple_field(self):
        text = json.dumps({"category_name": "Bills/Electricity"})
        assert _extract_answer_from_json(text, "category_name") == "Bills/Electricity"

    def test_nested_field(self):
        text = json.dumps({"classification": {"name": "Shopping"}})
        assert _extract_answer_from_json(text, "classification.name") == "Shopping"

    def test_invalid_json_returns_none(self):
        assert _extract_answer_from_json("not json", "category") is None

    def test_missing_field_returns_none(self):
        text = json.dumps({"other": "value"})
        assert _extract_answer_from_json(text, "category") is None


# ---------------------------------------------------------------------------
# _build_feedback
# ---------------------------------------------------------------------------


class TestBuildFeedback:
    def test_correct_high_confidence(self):
        fb = _build_feedback(
            is_correct=True,
            expected="Bills/Electricity",
            got="Bills/Electricity",
            logprob_score=-0.01,  # probability ~0.990
            top_alternatives=[],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.90,
        )
        assert fb == "Correct."

    def test_correct_medium_confidence(self):
        fb = _build_feedback(
            is_correct=True,
            expected="Bills/Electricity",
            got="Bills/Electricity",
            logprob_score=-0.05,  # probability ~0.951
            top_alternatives=[],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.90,
        )
        assert "Correct" in fb
        assert "probability" in fb

    def test_correct_low_confidence_lucky_guess(self):
        fb = _build_feedback(
            is_correct=True,
            expected="Bills/Electricity",
            got="Bills/Electricity",
            logprob_score=-2.3,  # probability ~0.100
            top_alternatives=[
                {"token": "gas", "probability": 0.09, "resolved_value": "Bills/Gas & Oil"},
            ],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.90,
        )
        assert "uncertain" in fb
        assert "Bills/Gas & Oil" in fb

    def test_incorrect_high_confidence(self):
        fb = _build_feedback(
            is_correct=False,
            expected="Shopping/Video Games",
            got="Shopping/Electronics",
            logprob_score=-0.005,  # probability ~0.995
            top_alternatives=[],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.50,
        )
        assert "WRONG" in fb
        assert "misleading" in fb

    def test_incorrect_low_confidence(self):
        fb = _build_feedback(
            is_correct=False,
            expected="Shopping/Video Games",
            got="Shopping/Electronics",
            logprob_score=-0.80,  # probability ~0.449
            top_alternatives=[
                {"token": "vid", "probability": 0.38, "resolved_value": "Shopping/Video Games"},
            ],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.50,
        )
        assert "Wrong" in fb
        assert "Shopping/Video Games" in fb

    def test_additional_context_included_on_incorrect(self):
        fb = _build_feedback(
            is_correct=False,
            expected="Bills/Electricity",
            got="Bills/Gas & Oil",
            logprob_score=-0.60,
            top_alternatives=[],
            additional_context={"merchant_type": "utility"},
            high_confidence_prob=0.99,
            low_confidence_prob=0.50,
        )
        assert "merchant_type" in fb
        assert "utility" in fb

    def test_none_logprob_correct_shows_correct(self):
        """When logprob is None and prediction is correct, feedback is just 'Correct.'."""
        fb = _build_feedback(
            is_correct=True,
            expected="Food",
            got="Food",
            logprob_score=None,
            top_alternatives=[],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.90,
        )
        assert fb == "Correct."

    def test_none_logprob_incorrect_shows_unknown_confidence(self):
        """When logprob is None and prediction is incorrect, feedback shows 'unknown confidence'."""
        fb = _build_feedback(
            is_correct=False,
            expected="Food",
            got="Drinks",
            logprob_score=None,
            top_alternatives=[],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.90,
        )
        assert "unknown" in fb

    def test_parse_error_shows_placeholder(self):
        fb = _build_feedback(
            is_correct=False,
            expected="Food",
            got=None,
            logprob_score=None,
            top_alternatives=[],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.90,
        )
        assert "<parse error>" in fb

    def test_feedback_contains_probability(self):
        """Feedback should show probability for non-trivial correct predictions."""
        fb = _build_feedback(
            is_correct=True,
            expected="Food",
            got="Food",
            logprob_score=-0.22,  # probability ~80%
            top_alternatives=[],
            additional_context={},
            high_confidence_prob=0.99,
            low_confidence_prob=0.50,
        )
        assert "probability" in fb


# ---------------------------------------------------------------------------
# ConfidenceAdapter.evaluate()
# ---------------------------------------------------------------------------


class TestConfidenceAdapterEvaluate:
    @patch("llm_structured_confidence.extract_logprobs")
    def test_correct_high_confidence_scores_one(self, mock_extract):
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(joint_logprob=-0.001, top_logprobs=[])
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS payment", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert len(result.scores) == 1
        assert result.scores[0] == 1.0
        assert result.objective_scores is not None
        assert result.objective_scores[0]["accuracy"] == 1.0
        assert result.objective_scores[0]["probability"] == pytest.approx(math.exp(-0.001))

    @patch("llm_structured_confidence.extract_logprobs")
    def test_correct_low_confidence_penalised(self, mock_extract):
        """Low logprob (e.g. -2.0 -> ~13% probability) should be penalised."""
        content = json.dumps({"category_name": "Bills/Electricity"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(joint_logprob=-2.0)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
            scoring_strategy=LinearBlendScoring(low_confidence_threshold=0.5, min_score_on_correct=0.3),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "LIGHT electricity bill", "additional_context": {}, "answer": "Bills/Electricity"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert len(result.scores) == 1
        assert result.scores[0] < 1.0
        assert result.scores[0] > 0.0

    @patch("llm_structured_confidence.extract_logprobs")
    def test_incorrect_scores_zero(self, mock_extract):
        content = json.dumps({"category_name": "Shopping/Electronics"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(joint_logprob=-0.22)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS payment", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 0.0
        assert result.objective_scores is not None
        assert result.objective_scores[0]["accuracy"] == 0.0

    @patch("llm_structured_confidence.extract_logprobs")
    def test_llm_error_returns_failure_score(self, mock_extract):
        def failing_model(messages):
            raise RuntimeError("API timeout")

        adapter = ConfidenceAdapter(
            model=failing_model,
            field_path="category_name",
            failure_score=0.0,
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 0.0
        assert result.outputs[0]["parsed_value"] is None

    @patch("llm_structured_confidence.extract_logprobs")
    def test_capture_traces_populates_trajectories(self, mock_extract):
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(
            joint_logprob=-0.16,
            top_logprobs=[_make_alt("food", 0.85, "Food & Drinks/Restaurants")],
        )
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=True)

        assert result.trajectories is not None
        assert len(result.trajectories) == 1
        traj = result.trajectories[0]
        assert traj["is_correct"] is True
        assert traj["logprob_score"] == pytest.approx(-0.16)
        assert traj["parsed_value"] == "Food & Drinks/Restaurants"
        assert len(traj["top_alternatives"]) == 1

    @patch("llm_structured_confidence.extract_logprobs")
    def test_no_traces_when_capture_traces_false(self, mock_extract):
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)
        mock_extract.return_value = [_make_entry(_make_field_logprob(-0.1))]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=False)

        assert result.trajectories is None

    @patch("llm_structured_confidence.extract_logprobs")
    def test_logprob_extraction_failure_degrades_gracefully(self, mock_extract):
        """When extract_logprobs fails, logprob_score should be None but scoring continues."""
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)
        mock_extract.side_effect = RuntimeError("logprob extraction error")

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 1.0
        assert result.outputs[0]["logprob_score"] is None

    @patch("llm_structured_confidence.extract_logprobs")
    def test_multiple_examples_in_batch(self, mock_extract):
        model = _make_callable_model(
            json.dumps({"category_name": "Food & Drinks/Restaurants"}),
            json.dumps({"category_name": "Bills/Electricity"}),
        )

        mock_extract.side_effect = [
            [_make_entry(_make_field_logprob(-0.001))],
            [_make_entry(_make_field_logprob(-0.001))],
        ]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )

        result = adapter.evaluate(_sample_batch(), {"system_prompt": "Classify."})

        assert len(result.scores) == 2
        assert len(result.outputs) == 2
        assert result.scores[0] == 1.0
        assert result.scores[1] == 1.0

    @patch("llm_structured_confidence.extract_logprobs")
    def test_threshold_strategy_gates_on_logprob(self, mock_extract):
        """Correct but below threshold probability -> score 0.0."""
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(joint_logprob=-0.51)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
            scoring_strategy=ThresholdScoring(threshold=0.7),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 0.0

    def test_callable_model_supported(self):
        """When model is a callable, it should be invoked directly."""
        called_with: list[Any] = []

        def fake_model(messages):
            called_with.append(messages)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
            return resp

        with patch("llm_structured_confidence.extract_logprobs") as mock_extract:
            mock_extract.return_value = [_make_entry(_make_field_logprob(-0.001))]

            adapter = ConfidenceAdapter(
                model=fake_model,
                field_path="category_name",
            )
            batch: list[ConfidenceDataInst] = [
                {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
            ]

            result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert len(called_with) == 1
        assert called_with[0][0]["role"] == "system"
        assert result.scores[0] == 1.0

    @patch("llm_structured_confidence.extract_logprobs")
    def test_case_insensitive_correctness(self, mock_extract):
        content = json.dumps({"category_name": "food & drinks/restaurants"})
        model = _make_callable_model(content)
        mock_extract.return_value = [_make_entry(_make_field_logprob(-0.001))]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 1.0

    @patch("llm_structured_confidence.extract_logprobs")
    def test_objective_scores_contain_accuracy_and_probability(self, mock_extract):
        """objective_scores should expose accuracy and probability for Pareto."""
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(joint_logprob=-0.35)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        obj = result.objective_scores[0]
        assert "accuracy" in obj
        assert "probability" in obj
        assert obj["probability"] == pytest.approx(math.exp(-0.35))


# ---------------------------------------------------------------------------
# ConfidenceAdapter.make_reflective_dataset()
# ---------------------------------------------------------------------------


class TestMakeReflectiveDataset:
    @patch("llm_structured_confidence.extract_logprobs")
    def test_reflective_dataset_structure(self, mock_extract):
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(
            joint_logprob=-1.05,
            top_logprobs=[_make_alt("elec", 0.30, "Shopping/Electronics")],
        )
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        eval_batch = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=True)
        dataset = adapter.make_reflective_dataset(
            candidate={"system_prompt": "Classify."},
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        assert "system_prompt" in dataset
        records = dataset["system_prompt"]
        assert len(records) == 1
        record = records[0]
        assert "Inputs" in record
        assert "Generated Outputs" in record
        assert "Feedback" in record

    @patch("llm_structured_confidence.extract_logprobs")
    def test_reflective_feedback_includes_confidence_info(self, mock_extract):
        content = json.dumps({"category_name": "Bills/Electricity"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(
            joint_logprob=-1.14,
            top_logprobs=[_make_alt("gas", 0.09, "Bills/Gas & Oil")],
        )
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "LIGHT electricity bill", "additional_context": {}, "answer": "Bills/Electricity"},
        ]

        eval_batch = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=True)
        dataset = adapter.make_reflective_dataset(
            candidate={"system_prompt": "Classify."},
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        feedback = dataset["system_prompt"][0]["Feedback"]
        assert "probability" in feedback
        assert "Bills/Gas & Oil" in feedback

    @patch("llm_structured_confidence.extract_logprobs")
    def test_generated_outputs_include_probability(self, mock_extract):
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)

        fl = _make_field_logprob(joint_logprob=-0.16)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        eval_batch = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=True)
        dataset = adapter.make_reflective_dataset(
            candidate={"system_prompt": "Classify."},
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        generated = dataset["system_prompt"][0]["Generated Outputs"]
        assert "probability" in generated

    @patch("llm_structured_confidence.extract_logprobs")
    def test_raises_when_no_trajectories(self, mock_extract):
        content = json.dumps({"category_name": "Food & Drinks/Restaurants"})
        model = _make_callable_model(content)
        mock_extract.return_value = [_make_entry(_make_field_logprob(-0.1))]

        adapter = ConfidenceAdapter(
            model=model,
            field_path="category_name",
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "additional_context": {}, "answer": "Food & Drinks/Restaurants"},
        ]

        eval_batch = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=False)

        with pytest.raises(AssertionError, match="Trajectories are required"):
            adapter.make_reflective_dataset(
                candidate={"system_prompt": "Classify."},
                eval_batch=eval_batch,
                components_to_update=["system_prompt"],
            )
