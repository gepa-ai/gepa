# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for FeedbackBuilder (Tasks 1-4)."""

from __future__ import annotations

import pytest

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
# Task 1 – base record construction
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
# Task 2 – score diff enrichment
# ---------------------------------------------------------------------------


class TestScoreDiff:
    def test_build_with_score_diff(self) -> None:
        """include_score_diff=True appends gap info and preserves existing Feedback."""
        batch = _batch(
            scores=[0.75],
            trajectories=[{"Feedback": "Good attempt"}],
        )
        result = FeedbackBuilder(include_score_diff=True).build(batch, ["comp"])
        rec = result["comp"][0]

        assert rec["Feedback"].startswith("Good attempt\n")
        assert "Gap from perfect: 0.25" in rec["Feedback"]
        assert "Score: 0.75" in rec["Feedback"]

    def test_build_score_diff_without_existing_feedback(self) -> None:
        """Creates Feedback field when none exists in side_info."""
        batch = _batch(
            scores=[0.6],
            trajectories=[{"question": "q1"}],
        )
        result = FeedbackBuilder(include_score_diff=True).build(batch, ["comp"])
        rec = result["comp"][0]

        assert "Feedback" in rec
        assert "Score: 0.6" in rec["Feedback"]
        assert "Gap from perfect: 0.40" in rec["Feedback"]


# ---------------------------------------------------------------------------
# Task 3 – rationale extraction
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
# Task 4 – global constraints
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
# Task 5 – combined feature tests
# ---------------------------------------------------------------------------


class TestCombined:
    def test_build_all_features_combined(self) -> None:
        """All enrichments work together without interference."""
        builder = FeedbackBuilder(
            include_score_diff=True,
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

        # Score diff appended to existing Feedback
        assert "Wrong answer." in rec["Feedback"]
        assert "Score: 0.3" in rec["Feedback"]
        assert "Gap from perfect: 0.70" in rec["Feedback"]

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
# Task 6 – import test
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_from_gepa_utils(self) -> None:
        """FeedbackBuilder is importable from gepa.utils (as shown in issue #264)."""
        from gepa.utils import FeedbackBuilder as FB

        assert FB is not None
        builder = FB(include_score_diff=True, global_constraints=["test"])
        assert builder.include_score_diff is True
