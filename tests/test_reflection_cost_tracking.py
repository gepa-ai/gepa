# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for LM-level cost tracking."""

from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import CostTrackingLM, LM


class TestLMCostTracking:
    """LM.cost accumulates across calls."""

    def test_initial_cost_is_zero(self):
        lm = LM("openai/gpt-4.1-mini")
        assert lm.cost == 0.0

    @patch("litellm.completion")
    @patch("litellm.completion_cost", return_value=0.005)
    def test_cost_accumulates(self, mock_cost, mock_completion):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "response"
        mock_resp.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_resp

        lm = LM("openai/gpt-4.1-mini")

        lm("first call")
        assert lm.cost == pytest.approx(0.005)

        lm("second call")
        assert lm.cost == pytest.approx(0.010)

    @patch("litellm.completion")
    @patch("litellm.completion_cost", side_effect=Exception("unknown model"))
    def test_unknown_model_cost_is_zero(self, mock_cost, mock_completion):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "response"
        mock_resp.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_resp

        lm = LM("custom/my-model")
        lm("test")
        assert lm.cost == 0.0

    @patch("litellm.batch_completion")
    @patch("litellm.completion_cost", return_value=0.002)
    def test_batch_complete_accumulates_cost(self, mock_cost, mock_batch):
        resp1 = MagicMock()
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.content = " answer1 "
        resp1.choices[0].finish_reason = "stop"
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.content = " answer2 "
        resp2.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp1, resp2]

        lm = LM("openai/gpt-4.1-mini")
        lm.batch_complete([[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]])
        assert lm.cost == pytest.approx(0.004)


class TestCostTrackingLM:
    """CostTrackingLM wraps plain callables with a .cost property."""

    def test_wraps_callable(self):
        fn = lambda prompt: "hello world"
        lm = CostTrackingLM(fn)
        result = lm("test prompt")
        assert result == "hello world"

    def test_cost_is_always_zero_for_plain_callable(self):
        fn = lambda prompt: "response"
        lm = CostTrackingLM(fn)
        lm("test")
        lm("test")
        assert lm.cost == 0.0

    def test_delegates_cost_from_inner(self):
        """If the inner LM has .cost, CostTrackingLM delegates to it."""
        inner = MagicMock()
        inner.cost = 1.23
        inner.return_value = "response"
        lm = CostTrackingLM(inner)
        assert lm.cost == pytest.approx(1.23)

    def test_exposes_cost_attribute(self):
        """CostTrackingLM always has .cost, so hasattr checks pass."""
        lm = CostTrackingLM(lambda p: "ok")
        assert hasattr(lm, "cost")


class TestMaxReflectionCostStopper:
    """Stopper reads cost from the LM instance."""

    def test_stops_when_budget_exceeded(self):
        from gepa.utils import MaxReflectionCostStopper

        mock_lm = MagicMock()
        mock_lm.cost = 10.5

        stopper = MaxReflectionCostStopper(10.0, lms=[mock_lm])
        assert stopper(MagicMock()) is True

    def test_continues_when_under_budget(self):
        from gepa.utils import MaxReflectionCostStopper

        mock_lm = MagicMock()
        mock_lm.cost = 5.0

        stopper = MaxReflectionCostStopper(10.0, lms=[mock_lm])
        assert stopper(MagicMock()) is False

    def test_sums_multiple_lms(self):
        from gepa.utils import MaxReflectionCostStopper

        lm1 = MagicMock()
        lm1.cost = 6.0
        lm2 = MagicMock()
        lm2.cost = 5.0

        stopper = MaxReflectionCostStopper(10.0, lms=[lm1, lm2])
        assert stopper(MagicMock()) is True

    def test_plain_callable_never_trips(self):
        from gepa.utils import MaxReflectionCostStopper

        plain_lm = lambda prompt: "response"
        stopper = MaxReflectionCostStopper(0.001, lms=[plain_lm])
        assert stopper(MagicMock()) is False

    def test_cost_tracking_lm_never_trips(self):
        """CostTrackingLM wrapping a plain callable reports cost=0.0."""
        from gepa.utils import MaxReflectionCostStopper

        lm = CostTrackingLM(lambda p: "response")
        lm("test")
        stopper = MaxReflectionCostStopper(0.001, lms=[lm])
        assert stopper(MagicMock()) is False
