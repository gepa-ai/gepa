# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Unit tests for logprob-based confidence scoring strategies.

All strategies receive ``logprob_score`` which is the **joint logprob**
(sum of per-token logprobs) for the target field.  This is always <= 0;
closer to 0 means more confident.  Internally, strategies convert to
probability via ``exp(logprob_score)`` before applying their logic.
"""

from __future__ import annotations

import math

import pytest

from gepa.adapters.confidence_adapter.scoring import (
    LinearBlendScoring,
    ScoringStrategy,
    SigmoidScoring,
    ThresholdScoring,
)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestScoringStrategyProtocol:
    def test_linear_blend_is_scoring_strategy(self):
        assert isinstance(LinearBlendScoring(), ScoringStrategy)

    def test_threshold_is_scoring_strategy(self):
        assert isinstance(ThresholdScoring(), ScoringStrategy)

    def test_sigmoid_is_scoring_strategy(self):
        assert isinstance(SigmoidScoring(), ScoringStrategy)


# ---------------------------------------------------------------------------
# LinearBlendScoring
# ---------------------------------------------------------------------------


class TestLinearBlendScoring:
    def test_incorrect_always_zero(self):
        s = LinearBlendScoring()
        assert s.score(is_correct=False, logprob_score=-0.01) == 0.0
        assert s.score(is_correct=False, logprob_score=-5.0) == 0.0
        assert s.score(is_correct=False, logprob_score=None) == 0.0

    def test_correct_none_logprob_returns_one(self):
        s = LinearBlendScoring()
        assert s.score(is_correct=True, logprob_score=None) == 1.0

    def test_correct_high_confidence_returns_one(self):
        """logprob close to 0 -> probability close to 1 -> score = 1.0."""
        s = LinearBlendScoring(low_confidence_threshold=0.5)
        assert s.score(is_correct=True, logprob_score=-0.01) == 1.0
        assert s.score(is_correct=True, logprob_score=-0.1) == 1.0
        assert s.score(is_correct=True, logprob_score=0.0) == 1.0

    def test_correct_at_threshold_returns_one(self):
        """logprob at exactly the threshold probability boundary."""
        s = LinearBlendScoring(low_confidence_threshold=0.5)
        logprob_at_threshold = math.log(0.5)
        assert s.score(is_correct=True, logprob_score=logprob_at_threshold) == 1.0

    def test_correct_below_threshold_interpolates(self):
        s = LinearBlendScoring(low_confidence_threshold=0.5, min_score_on_correct=0.3)
        logprob = math.log(0.25)
        probability = 0.25
        expected = 0.3 + (1.0 - 0.3) * (probability / 0.5)
        assert s.score(is_correct=True, logprob_score=logprob) == pytest.approx(expected)

    def test_correct_very_low_logprob_returns_near_min_score(self):
        """Very negative logprob -> probability near 0 -> near min_score."""
        s = LinearBlendScoring(low_confidence_threshold=0.5, min_score_on_correct=0.3)
        result = s.score(is_correct=True, logprob_score=-10.0)
        assert result == pytest.approx(0.3, abs=0.01)

    def test_score_range_is_zero_to_one(self):
        s = LinearBlendScoring()
        for logprob in [-10.0, -5.0, -2.0, -1.0, -0.5, -0.1, 0.0]:
            for correct in [True, False]:
                result = s.score(is_correct=correct, logprob_score=logprob)
                assert 0.0 <= result <= 1.0

    def test_describe_includes_params(self):
        s = LinearBlendScoring(low_confidence_threshold=0.6, min_score_on_correct=0.2)
        desc = s.describe()
        assert "0.6" in desc
        assert "0.2" in desc

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            LinearBlendScoring(low_confidence_threshold=0.0)
        with pytest.raises(ValueError):
            LinearBlendScoring(low_confidence_threshold=1.5)

    def test_invalid_min_score_raises(self):
        with pytest.raises(ValueError):
            LinearBlendScoring(min_score_on_correct=-0.1)
        with pytest.raises(ValueError):
            LinearBlendScoring(min_score_on_correct=1.0)

    def test_monotonically_increasing_with_logprob(self):
        """More confident (higher logprob) should yield higher score when correct."""
        s = LinearBlendScoring(low_confidence_threshold=0.8, min_score_on_correct=0.1)
        prev = -1.0
        for logprob in [-10.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.2]:
            score = s.score(is_correct=True, logprob_score=logprob)
            assert score >= prev
            prev = score


# ---------------------------------------------------------------------------
# ThresholdScoring
# ---------------------------------------------------------------------------


class TestThresholdScoring:
    def test_incorrect_always_zero(self):
        s = ThresholdScoring(threshold=0.7)
        assert s.score(is_correct=False, logprob_score=-0.01) == 0.0
        assert s.score(is_correct=False, logprob_score=None) == 0.0

    def test_correct_none_logprob_returns_one(self):
        s = ThresholdScoring()
        assert s.score(is_correct=True, logprob_score=None) == 1.0

    def test_correct_above_threshold_returns_one(self):
        """High logprob (close to 0) -> high probability -> passes threshold."""
        s = ThresholdScoring(threshold=0.7)
        assert s.score(is_correct=True, logprob_score=math.log(0.7)) == 1.0
        assert s.score(is_correct=True, logprob_score=-0.01) == 1.0

    def test_correct_below_threshold_returns_zero(self):
        """Low logprob -> low probability -> fails threshold."""
        s = ThresholdScoring(threshold=0.7)
        assert s.score(is_correct=True, logprob_score=math.log(0.69)) == 0.0
        assert s.score(is_correct=True, logprob_score=-5.0) == 0.0

    def test_describe_includes_threshold(self):
        s = ThresholdScoring(threshold=0.8)
        assert "0.8" in s.describe()

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            ThresholdScoring(threshold=0.0)
        with pytest.raises(ValueError):
            ThresholdScoring(threshold=1.5)


# ---------------------------------------------------------------------------
# SigmoidScoring
# ---------------------------------------------------------------------------


class TestSigmoidScoring:
    def test_incorrect_always_zero(self):
        s = SigmoidScoring()
        assert s.score(is_correct=False, logprob_score=-0.01) == 0.0
        assert s.score(is_correct=False, logprob_score=None) == 0.0

    def test_correct_none_logprob_returns_one(self):
        s = SigmoidScoring()
        assert s.score(is_correct=True, logprob_score=None) == 1.0

    def test_at_midpoint_probability_returns_half(self):
        """When exp(logprob) == midpoint, sigmoid outputs 0.5."""
        s = SigmoidScoring(midpoint=0.5, steepness=10.0)
        logprob_at_midpoint = math.log(0.5)
        result = s.score(is_correct=True, logprob_score=logprob_at_midpoint)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_high_confidence_approaches_one(self):
        s = SigmoidScoring(midpoint=0.5, steepness=10.0)
        result = s.score(is_correct=True, logprob_score=-0.05)
        assert result > 0.98

    def test_low_confidence_approaches_zero(self):
        s = SigmoidScoring(midpoint=0.5, steepness=10.0)
        result = s.score(is_correct=True, logprob_score=-5.0)
        assert result < 0.02

    def test_monotonically_increasing_for_correct(self):
        s = SigmoidScoring()
        prev = -1.0
        for logprob in [-10.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.2, -0.05]:
            score = s.score(is_correct=True, logprob_score=logprob)
            assert score > prev
            prev = score

    def test_describe_includes_params(self):
        s = SigmoidScoring(midpoint=0.4, steepness=12.0)
        desc = s.describe()
        assert "0.4" in desc
        assert "12.0" in desc

    def test_invalid_midpoint_raises(self):
        with pytest.raises(ValueError):
            SigmoidScoring(midpoint=0.0)
        with pytest.raises(ValueError):
            SigmoidScoring(midpoint=1.0)

    def test_invalid_steepness_raises(self):
        with pytest.raises(ValueError):
            SigmoidScoring(steepness=0)
        with pytest.raises(ValueError):
            SigmoidScoring(steepness=-1)

    def test_manual_sigmoid_computation(self):
        """Verify the formula: sigmoid(steepness * (exp(logprob) - midpoint))."""
        midpoint, steepness = 0.6, 8.0
        s = SigmoidScoring(midpoint=midpoint, steepness=steepness)
        logprob = math.log(0.75)
        probability = math.exp(logprob)
        expected = 1.0 / (1.0 + math.exp(-steepness * (probability - midpoint)))
        assert s.score(is_correct=True, logprob_score=logprob) == pytest.approx(expected)
