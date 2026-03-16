# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Unit tests for ConfidenceAdapter: evaluate() and make_reflective_dataset().

All tests mock ``llm_structured_confidence.extract_logprobs`` and
``ConfidenceAdapter._call_llm`` to avoid real LLM calls.  The mock
``field_logprob`` objects expose ``joint_logprob`` (the sum of per-token
logprobs) as the confidence metric.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gepa.adapters.confidence_adapter.confidence_adapter import (
    ConfidenceAdapter,
    ConfidenceDataInst,
    _build_feedback,
    _extract_answer_from_json,
)
from gepa.adapters.confidence_adapter.scoring import (
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


def _sample_batch() -> list[ConfidenceDataInst]:
    return [
        {"input": "UBER EATS payment", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        {"input": "LIGHT electricity bill", "answer": "Bills/Electricity", "additional_context": {"merchant_type": "utility"}},
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
            logprob_score=-0.05,
            top_alternatives=[],
            additional_context={},
        )
        assert "Correct with high confidence" in fb
        assert "logprob" in fb
        assert "probability" in fb

    def test_correct_low_confidence_lucky_guess(self):
        fb = _build_feedback(
            is_correct=True,
            expected="Bills/Electricity",
            got="Bills/Electricity",
            logprob_score=-2.3,
            top_alternatives=[
                {"token": "gas", "probability": 0.09, "resolved_value": "Bills/Gas & Oil"},
            ],
            additional_context={},
        )
        assert "unreliable" in fb
        assert "Bills/Gas & Oil" in fb

    def test_incorrect_high_confidence(self):
        fb = _build_feedback(
            is_correct=False,
            expected="Shopping/Video Games",
            got="Shopping/Electronics",
            logprob_score=-0.09,
            top_alternatives=[],
            additional_context={},
        )
        assert "Incorrect with high confidence" in fb
        assert "misleading" in fb

    def test_incorrect_low_confidence(self):
        fb = _build_feedback(
            is_correct=False,
            expected="Shopping/Video Games",
            got="Shopping/Electronics",
            logprob_score=-0.80,
            top_alternatives=[
                {"token": "vid", "probability": 0.38, "resolved_value": "Shopping/Video Games"},
            ],
            additional_context={},
        )
        assert "Incorrect" in fb
        assert "Shopping/Video Games" in fb

    def test_additional_context_included_on_incorrect(self):
        fb = _build_feedback(
            is_correct=False,
            expected="Bills/Electricity",
            got="Bills/Gas & Oil",
            logprob_score=-0.60,
            top_alternatives=[],
            additional_context={"merchant_type": "utility"},
        )
        assert "merchant_type" in fb
        assert "utility" in fb

    def test_none_logprob_shows_unknown(self):
        fb = _build_feedback(
            is_correct=True,
            expected="Food",
            got="Food",
            logprob_score=None,
            top_alternatives=[],
            additional_context={},
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
        )
        assert "<parse error>" in fb

    def test_feedback_contains_both_logprob_and_probability(self):
        """Feedback should show both logprob and derived probability for clarity."""
        fb = _build_feedback(
            is_correct=True,
            expected="Food",
            got="Food",
            logprob_score=-0.22,
            top_alternatives=[],
            additional_context={},
        )
        assert "logprob" in fb
        assert "probability" in fb


# ---------------------------------------------------------------------------
# ConfidenceAdapter.evaluate()
# ---------------------------------------------------------------------------


class TestConfidenceAdapterEvaluate:
    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_correct_high_confidence_scores_one(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(joint_logprob=-0.08, top_logprobs=[])
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS payment", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert len(result.scores) == 1
        assert result.scores[0] == 1.0
        assert result.objective_scores is not None
        assert result.objective_scores[0]["accuracy"] == 1.0
        assert result.objective_scores[0]["logprob"] == pytest.approx(-0.08)
        assert result.objective_scores[0]["probability"] == pytest.approx(math.exp(-0.08))

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_correct_low_confidence_penalised(self, mock_extract, mock_call):
        """Low logprob (e.g. -2.0 -> ~13% probability) should be penalised."""
        resp = _make_litellm_response(json.dumps({"category_name": "Bills/Electricity"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(joint_logprob=-2.0)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
            scoring_strategy=LinearBlendScoring(low_confidence_threshold=0.5, min_score_on_correct=0.3),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "LIGHT electricity bill", "answer": "Bills/Electricity", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert len(result.scores) == 1
        assert result.scores[0] < 1.0
        assert result.scores[0] > 0.0

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_incorrect_scores_zero(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Shopping/Electronics"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(joint_logprob=-0.22)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS payment", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 0.0
        assert result.objective_scores is not None
        assert result.objective_scores[0]["accuracy"] == 0.0

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_llm_error_returns_failure_score(self, mock_extract, mock_call):
        mock_call.side_effect = RuntimeError("API timeout")

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
            failure_score=0.0,
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 0.0
        assert result.outputs[0]["parsed_value"] is None

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_capture_traces_populates_trajectories(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(
            joint_logprob=-0.16,
            top_logprobs=[_make_alt("food", 0.85, "Food & Drinks/Restaurants")],
        )
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=True)

        assert result.trajectories is not None
        assert len(result.trajectories) == 1
        traj = result.trajectories[0]
        assert traj["is_correct"] is True
        assert traj["logprob_score"] == pytest.approx(-0.16)
        assert traj["parsed_value"] == "Food & Drinks/Restaurants"
        assert len(traj["top_alternatives"]) == 1

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_no_traces_when_capture_traces_false(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp
        mock_extract.return_value = [_make_entry(_make_field_logprob(-0.1))]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=False)

        assert result.trajectories is None

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_logprob_extraction_failure_degrades_gracefully(self, mock_extract, mock_call):
        """When extract_logprobs fails, logprob_score should be None but scoring continues."""
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp
        mock_extract.side_effect = RuntimeError("logprob extraction error")

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 1.0
        assert result.outputs[0]["logprob_score"] is None

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_multiple_examples_in_batch(self, mock_extract, mock_call):
        responses = [
            _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"})),
            _make_litellm_response(json.dumps({"category_name": "Bills/Electricity"})),
        ]
        mock_call.side_effect = responses

        mock_extract.side_effect = [
            [_make_entry(_make_field_logprob(-0.10))],
            [_make_entry(_make_field_logprob(-0.30))],
        ]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )

        result = adapter.evaluate(_sample_batch(), {"system_prompt": "Classify."})

        assert len(result.scores) == 2
        assert len(result.outputs) == 2
        assert result.scores[0] == 1.0
        assert result.scores[1] == 1.0

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_threshold_strategy_gates_on_logprob(self, mock_extract, mock_call):
        """Correct but below threshold probability -> score 0.0."""
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(joint_logprob=-0.51)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
            scoring_strategy=ThresholdScoring(threshold=0.7),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
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
            mock_extract.return_value = [_make_entry(_make_field_logprob(-0.05))]

            adapter = ConfidenceAdapter(
                model=fake_model,
                field_path="category_name",
            )
            batch: list[ConfidenceDataInst] = [
                {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
            ]

            result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert len(called_with) == 1
        assert called_with[0][0]["role"] == "system"
        assert result.scores[0] == 1.0

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_case_insensitive_correctness(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "food & drinks/restaurants"}))
        mock_call.return_value = resp
        mock_extract.return_value = [_make_entry(_make_field_logprob(-0.10))]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        assert result.scores[0] == 1.0

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_objective_scores_contain_logprob_and_probability(self, mock_extract, mock_call):
        """objective_scores should expose logprob and probability for Pareto."""
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(joint_logprob=-0.35)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        result = adapter.evaluate(batch, {"system_prompt": "Classify."})

        obj = result.objective_scores[0]
        assert "accuracy" in obj
        assert "logprob" in obj
        assert "probability" in obj
        assert obj["logprob"] == pytest.approx(-0.35)
        assert obj["probability"] == pytest.approx(math.exp(-0.35))


# ---------------------------------------------------------------------------
# ConfidenceAdapter.make_reflective_dataset()
# ---------------------------------------------------------------------------


class TestMakeReflectiveDataset:
    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_reflective_dataset_structure(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(
            joint_logprob=-1.05,
            top_logprobs=[_make_alt("elec", 0.30, "Shopping/Electronics")],
        )
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "UBER EATS", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
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

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_reflective_feedback_includes_logprob_info(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Bills/Electricity"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(
            joint_logprob=-1.14,
            top_logprobs=[_make_alt("gas", 0.09, "Bills/Gas & Oil")],
        )
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "LIGHT electricity bill", "answer": "Bills/Electricity", "additional_context": {}},
        ]

        eval_batch = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=True)
        dataset = adapter.make_reflective_dataset(
            candidate={"system_prompt": "Classify."},
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        feedback = dataset["system_prompt"][0]["Feedback"]
        assert "logprob" in feedback
        assert "probability" in feedback
        assert "Bills/Gas & Oil" in feedback

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_generated_outputs_include_logprob_and_probability(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp

        fl = _make_field_logprob(joint_logprob=-0.16)
        mock_extract.return_value = [_make_entry(fl)]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        eval_batch = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=True)
        dataset = adapter.make_reflective_dataset(
            candidate={"system_prompt": "Classify."},
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        generated = dataset["system_prompt"][0]["Generated Outputs"]
        assert "logprob" in generated
        assert "probability" in generated

    @patch("gepa.adapters.confidence_adapter.confidence_adapter.ConfidenceAdapter._call_llm")
    @patch("llm_structured_confidence.extract_logprobs")
    def test_raises_when_no_trajectories(self, mock_extract, mock_call):
        resp = _make_litellm_response(json.dumps({"category_name": "Food & Drinks/Restaurants"}))
        mock_call.return_value = resp
        mock_extract.return_value = [_make_entry(_make_field_logprob(-0.1))]

        adapter = ConfidenceAdapter(
            model="openai/gpt-4.1-mini",
            field_path="category_name",
            response_format=_sample_response_format(),
        )
        batch: list[ConfidenceDataInst] = [
            {"input": "test", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        ]

        eval_batch = adapter.evaluate(batch, {"system_prompt": "Classify."}, capture_traces=False)

        with pytest.raises(AssertionError, match="Trajectories are required"):
            adapter.make_reflective_dataset(
                candidate={"system_prompt": "Classify."},
                eval_batch=eval_batch,
                components_to_update=["system_prompt"],
            )
