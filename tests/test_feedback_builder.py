# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for FeedbackBuilder"""

from __future__ import annotations

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.adapter import EvaluationBatch
from gepa.utils.feedback_builder import FeedbackBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch(
    scores: list[float],
    trajectories: list[dict],
) -> EvaluationBatch:
    """Convenience factory for ``EvaluationBatch``."""
    return EvaluationBatch(
        outputs=[None] * len(scores),
        scores=scores,
        trajectories=trajectories,
    )


# ---------------------------------------------------------------------------
# base record construction
# ---------------------------------------------------------------------------


class TestBaseRecords:
    def test_build_base_records_from_side_info(self) -> None:
        """Scores rename, shared field passthrough, component-specific merge, other component-specific excluded."""
        batch = _batch(
            scores=[0.8],
            trajectories=[
                {
                    "scores": {"acc": 0.8},
                    "question": "What is 2+2?",
                    "comp_a_specific_info": {"hint_a": "yes"},
                    "comp_b_specific_info": {"hint_b": "no"},
                }
            ],
        )
        result = FeedbackBuilder().build(batch, ["comp_a"])
        records = result["comp_a"]
        assert len(records) == 1
        rec = records[0]

        # scores renamed
        assert rec["Scores (Higher is Better)"] == {"acc": 0.8}
        # shared field passed through
        assert rec["question"] == "What is 2+2?"
        # component-specific merged
        assert rec["hint_a"] == "yes"
        # other component-specific excluded
        assert "hint_b" not in rec
        assert "comp_b_specific_info" not in rec
        assert "comp_a_specific_info" not in rec

    def test_build_multiple_components(self) -> None:
        """Separate record lists per component with correct specific_info."""
        batch = _batch(
            scores=[0.5],
            trajectories=[
                {
                    "scores": {"f1": 0.5},
                    "shared": "ok",
                    "alpha_specific_info": {"x": 1},
                    "beta_specific_info": {"y": 2},
                }
            ],
        )
        result = FeedbackBuilder().build(batch, ["alpha", "beta"])

        assert "x" in result["alpha"][0]
        assert "y" not in result["alpha"][0]

        assert "y" in result["beta"][0]
        assert "x" not in result["beta"][0]

        # shared field present in both
        assert result["alpha"][0]["shared"] == "ok"
        assert result["beta"][0]["shared"] == "ok"


# ---------------------------------------------------------------------------
# Auto-inject score (#288)
# ---------------------------------------------------------------------------


class TestAutoInjectScore:
    def test_score_injected_when_not_in_side_info(self) -> None:
        """Score is auto-injected as record['Score'] when evaluator does not provide it."""
        batch = _batch(
            scores=[0.75],
            trajectories=[{"Input": "q1", "Output": "a1"}],
        )
        result = FeedbackBuilder().build(batch, ["comp"])
        rec = result["comp"][0]

        assert rec["Score"] == 0.75

    def test_score_not_injected_when_already_present(self) -> None:
        """Score is NOT injected if side_info already has a 'score' key (case-insensitive)."""
        batch = _batch(
            scores=[0.75],
            trajectories=[{"Input": "q1", "Score": 0.9}],
        )
        result = FeedbackBuilder().build(batch, ["comp"])
        rec = result["comp"][0]

        # Should keep the user-provided value, not overwrite with eval score
        assert rec["Score"] == 0.9

    def test_score_not_injected_case_insensitive(self) -> None:
        """Case-insensitive check: 'score' in side_info prevents injection."""
        batch = _batch(
            scores=[0.75],
            trajectories=[{"Input": "q1", "score": 0.9}],
        )
        result = FeedbackBuilder().build(batch, ["comp"])
        rec = result["comp"][0]

        assert rec["score"] == 0.9
        assert "Score" not in rec


# ---------------------------------------------------------------------------
# Rationale extraction
# ---------------------------------------------------------------------------


class TestRationale:
    def test_build_with_rationale_field(self) -> None:
        """Extracts named field as 'Rationale'; raw field not present as top-level key."""
        batch = _batch(
            scores=[0.9],
            trajectories=[{"reasoning": "because reasons", "other": "val"}],
        )
        result = FeedbackBuilder(rationale_field="reasoning").build(batch, ["comp"])
        rec = result["comp"][0]

        assert rec["Rationale"] == "because reasons"
        # raw key should have been skipped in base loop
        assert "reasoning" not in rec

    def test_build_rationale_field_missing_gracefully(self) -> None:
        """Missing field in some side_infos is silently skipped."""
        batch = _batch(
            scores=[0.9, 0.8],
            trajectories=[
                {"reasoning": "present"},
                {"other_key": "no reasoning here"},
            ],
        )
        result = FeedbackBuilder(rationale_field="reasoning").build(batch, ["comp"])
        recs = result["comp"]

        assert recs[0]["Rationale"] == "present"
        assert "Rationale" not in recs[1]


# ---------------------------------------------------------------------------
# Global constraints
# ---------------------------------------------------------------------------


class TestGlobalConstraints:
    def test_build_with_global_constraints(self) -> None:
        """Constraints field added to every record."""
        batch = _batch(
            scores=[0.5, 0.7],
            trajectories=[{"a": 1}, {"a": 2}],
        )
        builder = FeedbackBuilder(global_constraints=["Be concise", "No jargon"])
        result = builder.build(batch, ["comp"])

        for rec in result["comp"]:
            assert rec["Constraints"] == "- Be concise\n- No jargon"

    def test_build_empty_constraints_no_field(self) -> None:
        """Empty list means no Constraints field."""
        batch = _batch(
            scores=[0.5],
            trajectories=[{"a": 1}],
        )
        result = FeedbackBuilder(global_constraints=[]).build(batch, ["comp"])
        assert "Constraints" not in result["comp"][0]


# ---------------------------------------------------------------------------
# Combined feature tests
# ---------------------------------------------------------------------------


class TestCombined:
    def test_build_all_features_combined(self) -> None:
        """All enrichments work together without interference."""
        builder = FeedbackBuilder(
            rationale_field="explanation",
            global_constraints=["Stay general.", "Preserve schema."],
        )
        batch = _batch(
            scores=[0.3],
            trajectories=[
                {
                    "Input": "q1",
                    "Output": "a1",
                    "Feedback": "Wrong answer.",
                    "explanation": "The correct approach is X.",
                    "scores": {"accuracy": 0.0},
                    "prompt_specific_info": {"Detail": "prompt-level info"},
                },
            ],
        )
        result = builder.build(batch, ["prompt"])
        rec = result["prompt"][0]

        # Base record construction
        assert rec["Input"] == "q1"
        assert rec["Output"] == "a1"
        assert rec["Scores (Higher is Better)"] == {"accuracy": 0.0}
        assert rec["Detail"] == "prompt-level info"

        # Auto-injected score (#288)
        assert rec["Score"] == 0.3

        # Feedback unchanged (no score diff enrichment)
        assert rec["Feedback"] == "Wrong answer."

        # Rationale extracted
        assert rec["Rationale"] == "The correct approach is X."
        assert "explanation" not in rec

        # Constraints added
        assert "Stay general." in rec["Constraints"]
        assert "Preserve schema." in rec["Constraints"]

    def test_build_default_is_noop_enrichment(self) -> None:
        """Default FeedbackBuilder (no options) produces identical output to raw side_info extraction."""
        batch = _batch(
            scores=[1.0],
            trajectories=[
                {"Input": "q1", "Output": "a1", "Feedback": "Good.", "scores": {"acc": 1.0}},
            ],
        )
        result = FeedbackBuilder().build(batch, ["prompt"])
        rec = result["prompt"][0]

        assert "Rationale" not in rec
        assert "Constraints" not in rec
        assert rec["Feedback"] == "Good."


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_from_gepa_utils(self) -> None:
        """FeedbackBuilder is importable from gepa.utils (as shown in issue #264)."""
        from gepa.utils import FeedbackBuilder as FeedbackBuilderFromUtils

        assert FeedbackBuilderFromUtils is not None
        builder = FeedbackBuilderFromUtils(global_constraints=["test"])
        assert builder.global_constraints == ["test"]


# ---------------------------------------------------------------------------
# adapter integration
# ---------------------------------------------------------------------------


def _dummy_evaluator(candidate, example):
    return 1.0, "output", {"Input": example, "Output": "output"}


class TestAdapterIntegration:
    def test_adapter_delegates_to_feedback_builder(self) -> None:
        """OptimizeAnythingAdapter delegates make_reflective_dataset to FeedbackBuilder when set."""
        builder = FeedbackBuilder(global_constraints=["Do not overfit."])
        adapter = OptimizeAnythingAdapter(evaluator=_dummy_evaluator, feedback_builder=builder)
        eval_batch = EvaluationBatch(
            outputs=["out"],
            scores=[0.5],
            trajectories=[{"Input": "q1", "Output": "a1"}],
        )
        result = adapter.make_reflective_dataset({"prompt": "test"}, eval_batch, ["prompt"])
        assert "Constraints" in result["prompt"][0]
        assert "Do not overfit." in result["prompt"][0]["Constraints"]

    def test_adapter_without_feedback_builder_unchanged(self) -> None:
        """Without feedback_builder, make_reflective_dataset works as before."""
        adapter = OptimizeAnythingAdapter(evaluator=_dummy_evaluator)
        eval_batch = EvaluationBatch(
            outputs=["out"],
            scores=[0.5],
            trajectories=[{"Input": "q1", "Output": "a1", "scores": {"acc": 0.5}}],
        )
        result = adapter.make_reflective_dataset({"prompt": "test"}, eval_batch, ["prompt"])
        assert result["prompt"][0]["Input"] == "q1"
        assert result["prompt"][0]["Scores (Higher is Better)"] == {"acc": 0.5}
        assert "Constraints" not in result["prompt"][0]

    def test_adapter_fallback_auto_injects_score(self) -> None:
        """Without feedback_builder, adapter fallback also auto-injects score (#288)."""
        adapter = OptimizeAnythingAdapter(evaluator=_dummy_evaluator)
        eval_batch = EvaluationBatch(
            outputs=["out"],
            scores=[0.5],
            trajectories=[{"Input": "q1", "Output": "a1"}],
        )
        result = adapter.make_reflective_dataset({"prompt": "test"}, eval_batch, ["prompt"])
        assert result["prompt"][0]["Score"] == 0.5
