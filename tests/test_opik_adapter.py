# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the Opik adapter.

These tests intentionally do **not** import ``opik``. The adapter is designed
to work with any object that matches the ``ScoreResult`` shape (``.value`` /
``.reason``) — so we exercise it with hand-rolled duck-typed scorers. CI does
not need the ``gepa[opik]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from gepa.adapters.opik_adapter import (
    OpikAdapter,
    opik_dataset_to_examples,
)
from gepa.core.adapter import EvaluationBatch


@dataclass
class FakeScoreResult:
    """Duck-typed stand-in for ``opik.evaluation.metrics.ScoreResult``."""

    value: float
    reason: str | None = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _exact_match_metric(item: dict, output: str) -> FakeScoreResult:
    if output.strip().lower() == str(item["label"]).strip().lower():
        return FakeScoreResult(value=1.0, reason="exact match")
    return FakeScoreResult(
        value=0.0,
        reason=f"expected {item['label']!r}, got {output!r}",
    )


def _make_chat_callable(responses_by_user_msg: dict[str, str]):
    def chat(messages):
        # messages = [{"role": "system", ...}, {"role": "user", "content": "..."}]
        user = messages[-1]["content"]
        return responses_by_user_msg[user]

    return chat


# ----------------------------------------------------------------------
# opik_dataset_to_examples
# ----------------------------------------------------------------------


class TestDatasetToExamples:
    def test_list_of_dicts_passthrough(self):
        items = [{"input": "q1", "label": "a1"}, {"input": "q2", "label": "a2"}]
        result = opik_dataset_to_examples(items)
        assert result == items
        # Defensive copy: mutating the result must not affect the source.
        result[0]["input"] = "mutated"
        assert items[0]["input"] == "q1"

    def test_dataset_with_get_items(self):
        class FakeDataset:
            def get_items(self):
                return [{"input": "x", "label": "y"}]

        assert opik_dataset_to_examples(FakeDataset()) == [{"input": "x", "label": "y"}]


# ----------------------------------------------------------------------
# OpikAdapter.evaluate
# ----------------------------------------------------------------------


class TestEvaluate:
    def test_happy_path_scores(self):
        batch = [
            {"input": "Capital of France?", "label": "Paris"},
            {"input": "Capital of Germany?", "label": "Berlin"},
            {"input": "Capital of Japan?", "label": "Mumbai"},  # we'll answer "Tokyo"
        ]
        chat = _make_chat_callable(
            {
                "Capital of France?": "Paris",
                "Capital of Germany?": "Berlin",
                "Capital of Japan?": "Tokyo",
            }
        )
        adapter = OpikAdapter(model=chat, metric=_exact_match_metric, input_field="input")
        result = adapter.evaluate(
            batch=batch,
            candidate={"system_prompt": "Answer concisely."},
            capture_traces=False,
        )

        assert result.scores == [1.0, 1.0, 0.0]
        assert result.trajectories is None
        assert result.outputs[0]["full_assistant_response"] == "Paris"

    def test_capture_traces(self):
        batch = [{"input": "q", "label": "a"}]
        chat = _make_chat_callable({"q": "a"})
        adapter = OpikAdapter(model=chat, metric=_exact_match_metric, input_field="input")
        result = adapter.evaluate(
            batch=batch,
            candidate={"system_prompt": "p"},
            capture_traces=True,
        )

        assert result.trajectories is not None and len(result.trajectories) == 1
        traj = result.trajectories[0]
        assert traj["data"] == batch[0]
        assert traj["full_assistant_response"] == "a"
        assert traj["feedback"] == "exact match"

    def test_system_text_resolution_prefers_system_prompt_key(self):
        captured: list = []

        def chat(messages):
            captured.append(messages[0]["content"])
            return "ok"

        adapter = OpikAdapter(model=chat, metric=_exact_match_metric, input_field="input")
        adapter.evaluate(
            batch=[{"input": "q", "label": "ok"}],
            candidate={"system_prompt": "EXPECTED-SYSTEM"},
            capture_traces=False,
        )
        assert captured == ["EXPECTED-SYSTEM"]

    def test_system_text_falls_back_through_keys(self):
        captured: list = []

        def chat(messages):
            captured.append(messages[0]["content"])
            return "ok"

        adapter = OpikAdapter(model=chat, metric=_exact_match_metric, input_field="input")
        adapter.evaluate(
            batch=[{"input": "q", "label": "ok"}],
            candidate={"system": "via-system-key"},  # not system_prompt
            capture_traces=False,
        )
        assert captured == ["via-system-key"]

    def test_missing_input_field_raises(self):
        adapter = OpikAdapter(
            model=lambda msgs: "x",
            metric=_exact_match_metric,
            input_field="question",  # different from what the item provides
        )
        with pytest.raises(KeyError, match="question"):
            adapter.evaluate(
                batch=[{"input": "q", "label": "a"}],
                candidate={"system_prompt": "p"},
                capture_traces=False,
            )


# ----------------------------------------------------------------------
# Metric return-value coercion
# ----------------------------------------------------------------------


class TestMetricCoercion:
    def _run(self, metric):
        adapter = OpikAdapter(
            model=_make_chat_callable({"q": "out"}),
            metric=metric,
            input_field="input",
        )
        return adapter.evaluate(
            batch=[{"input": "q", "label": "out"}],
            candidate={"system_prompt": "p"},
            capture_traces=True,
        )

    def test_score_result_object(self):
        result = self._run(lambda item, out: FakeScoreResult(value=0.7, reason="why"))
        assert result.scores == [0.7]
        assert result.trajectories[0]["feedback"] == "why"

    def test_tuple_return(self):
        result = self._run(lambda item, out: (0.5, "tuple-feedback"))
        assert result.scores == [0.5]
        assert result.trajectories[0]["feedback"] == "tuple-feedback"

    def test_bare_float(self):
        result = self._run(lambda item, out: 0.25)
        assert result.scores == [0.25]
        # feedback falls back to the synthesized default — must be a string.
        assert isinstance(result.trajectories[0]["feedback"], str)

    def test_score_attr_fallback(self):
        """Some scorer libraries expose ``.score`` instead of ``.value``."""

        class AltScore:
            score = 0.9
            reason = "alt"
            value = None  # explicit None forces the score-attr branch

        result = self._run(lambda item, out: AltScore())
        assert result.scores == [0.9]
        assert result.trajectories[0]["feedback"] == "alt"


# ----------------------------------------------------------------------
# OpikAdapter.make_reflective_dataset
# ----------------------------------------------------------------------


class TestMakeReflectiveDataset:
    def _eval_batch(self, adapter, batch):
        return adapter.evaluate(batch=batch, candidate={"system_prompt": "p"}, capture_traces=True)

    def test_record_shape_and_independent_lists(self):
        adapter = OpikAdapter(
            model=_make_chat_callable({"q1": "a1", "q2": "wrong"}),
            metric=_exact_match_metric,
            input_field="input",
        )
        eval_batch = self._eval_batch(
            adapter,
            [{"input": "q1", "label": "a1"}, {"input": "q2", "label": "a2"}],
        )
        ds = adapter.make_reflective_dataset(
            candidate={"system_prompt": "p"},
            eval_batch=eval_batch,
            components_to_update=["system_prompt", "another_component"],
        )
        assert set(ds.keys()) == {"system_prompt", "another_component"}
        # Records present and shaped right.
        sys_records = ds["system_prompt"]
        assert len(sys_records) == 2
        assert sys_records[0] == {
            "Inputs": "q1",
            "Generated Outputs": "a1",
            "Feedback": "exact match",
        }
        # Independent lists per component (regression guard for the shared-list
        # bug flagged in PR #141).
        sys_records.pop()
        assert len(ds["another_component"]) == 2

    def test_missing_trajectories_raises(self):
        adapter = OpikAdapter(
            model=_make_chat_callable({}),
            metric=_exact_match_metric,
            input_field="input",
        )
        empty = EvaluationBatch(outputs=[], scores=[], trajectories=None)
        with pytest.raises(ValueError, match="Trajectories are required"):
            adapter.make_reflective_dataset(
                candidate={"system_prompt": "p"},
                eval_batch=empty,
                components_to_update=["system_prompt"],
            )

    def test_empty_trajectories_raises(self):
        adapter = OpikAdapter(
            model=_make_chat_callable({}),
            metric=_exact_match_metric,
            input_field="input",
        )
        empty = EvaluationBatch(outputs=[], scores=[], trajectories=[])
        with pytest.raises(ValueError, match="No trajectories"):
            adapter.make_reflective_dataset(
                candidate={"system_prompt": "p"},
                eval_batch=empty,
                components_to_update=["system_prompt"],
            )


# ----------------------------------------------------------------------
# Litellm path (smoke test — uses callable to avoid network)
# ----------------------------------------------------------------------


class TestLitellmPath:
    def test_string_model_initializes_lm(self):
        """Construction with a litellm string must not fail (LM is lazily wrapped)."""
        adapter = OpikAdapter(model="openai/gpt-4o-mini", metric=_exact_match_metric)
        assert adapter._lm is not None
        assert adapter.model == "openai/gpt-4o-mini"
