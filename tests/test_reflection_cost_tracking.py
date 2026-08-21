# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for LM-level cost and token tracking."""

from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import LM, TrackingLM


class TestLMCostTracking:
    """LM.total_cost and token counts accumulate across calls."""

    def test_initial_cost_is_zero(self):
        lm = LM("openai/gpt-4.1-mini")
        assert lm.total_cost == 0.0
        assert lm.total_tokens_in == 0
        assert lm.total_tokens_out == 0

    @patch("litellm.completion")
    @patch("litellm.completion_cost", return_value=0.005)
    def test_cost_and_tokens_accumulate(self, mock_cost, mock_completion):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "response"
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        mock_completion.return_value = mock_resp

        lm = LM("openai/gpt-4.1-mini")

        lm("first call")
        assert lm.total_cost == pytest.approx(0.005)
        assert lm.total_tokens_in == 100
        assert lm.total_tokens_out == 50

        lm("second call")
        assert lm.total_cost == pytest.approx(0.010)
        assert lm.total_tokens_in == 200
        assert lm.total_tokens_out == 100

    @patch("litellm.batch_completion")
    @patch("litellm.completion_cost", return_value=0.002)
    def test_batch_complete_accumulates_cost(self, mock_cost, mock_batch):
        resp1 = MagicMock()
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.content = " answer1 "
        resp1.choices[0].finish_reason = "stop"
        resp1.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.content = " answer2 "
        resp2.choices[0].finish_reason = "stop"
        resp2.usage = MagicMock(prompt_tokens=60, completion_tokens=30)
        mock_batch.return_value = [resp1, resp2]

        lm = LM("openai/gpt-4.1-mini")
        lm.batch_complete([[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]])
        assert lm.total_cost == pytest.approx(0.004)
        assert lm.total_tokens_in == 110
        assert lm.total_tokens_out == 50

    @patch("litellm.completion")
    @patch("litellm.completion_cost", side_effect=Exception("unknown model"))
    def test_unknown_model_cost_is_zero(self, mock_cost, mock_completion):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "response"
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = None
        mock_completion.return_value = mock_resp

        lm = LM("custom/my-model")
        lm("test")
        assert lm.total_cost == 0.0
        assert lm.total_tokens_in == 0
        assert lm.total_tokens_out == 0


class TestTrackingLM:
    """TrackingLM wraps plain callables to track estimated token usage."""

    def test_wraps_callable(self):
        fn = lambda prompt: "hello world"
        lm = TrackingLM(fn)
        result = lm("test prompt")
        assert result == "hello world"

    def test_tracks_estimated_tokens(self):
        fn = lambda prompt: "a" * 40  # ~10 tokens out
        lm = TrackingLM(fn)
        lm("b" * 80)  # ~20 tokens in
        assert lm.total_tokens_in == 20
        assert lm.total_tokens_out == 10

    def test_accumulates_across_calls(self):
        fn = lambda prompt: "response"
        lm = TrackingLM(fn)
        lm("prompt one")
        tokens_after_one = lm.total_tokens_out
        lm("prompt two")
        assert lm.total_tokens_out == tokens_after_one * 2

    def test_cost_is_always_zero(self):
        fn = lambda prompt: "response"
        lm = TrackingLM(fn)
        lm("test")
        assert lm.total_cost == 0.0

    def test_exposes_tracking_attributes(self):
        lm = TrackingLM(lambda p: "ok")
        assert hasattr(lm, "total_cost")
        assert hasattr(lm, "total_tokens_in")
        assert hasattr(lm, "total_tokens_out")

    def test_handles_list_prompt(self):
        fn = lambda prompt: "response"
        lm = TrackingLM(fn)
        lm([{"role": "user", "content": "hello"}])
        assert lm.total_tokens_in > 0


class TestCallableWrapping:
    """Plain callables are wrapped in TrackingLM by gepa.optimize / optimize_anything."""

    def test_plain_callable_gets_wrapped(self):
        fn = lambda prompt: "```\nnew text\n```"
        assert not hasattr(fn, "total_cost")
        wrapped = TrackingLM(fn) if not hasattr(fn, "total_cost") else fn
        assert isinstance(wrapped, TrackingLM)
        assert hasattr(wrapped, "total_cost")

    def test_lm_instance_not_double_wrapped(self):
        lm = LM("openai/gpt-4.1-mini")
        assert hasattr(lm, "total_cost")
        wrapped = TrackingLM(lm) if not hasattr(lm, "total_cost") else lm
        assert isinstance(wrapped, LM)


class TestMaxReflectionCostStopper:
    """Stopper reads cost from the LM instance."""

    def test_stops_when_budget_exceeded(self):
        from gepa.utils import MaxReflectionCostStopper

        mock_lm = MagicMock()
        mock_lm.total_cost = 10.5
        stopper = MaxReflectionCostStopper(10.0, reflection_lm=mock_lm)
        assert stopper(MagicMock()) is True

    def test_continues_when_under_budget(self):
        from gepa.utils import MaxReflectionCostStopper

        mock_lm = MagicMock()
        mock_lm.total_cost = 5.0
        stopper = MaxReflectionCostStopper(10.0, reflection_lm=mock_lm)
        assert stopper(MagicMock()) is False

    def test_tracking_lm_never_trips(self):
        from gepa.utils import MaxReflectionCostStopper

        lm = TrackingLM(lambda p: "response")
        lm("test")
        stopper = MaxReflectionCostStopper(0.001, reflection_lm=lm)
        assert stopper(MagicMock()) is False

    def test_none_lm_never_trips(self):
        from gepa.utils import MaxReflectionCostStopper

        stopper = MaxReflectionCostStopper(0.001, reflection_lm=None)
        assert stopper(MagicMock()) is False
