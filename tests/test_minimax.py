# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for MiniMax provider integration in gepa.lm.LM."""

import os
from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import (
    LM,
    MINIMAX_API_BASE,
    MINIMAX_MODELS,
    _clamp_minimax_temperature,
    _is_minimax_model,
    _resolve_minimax_model,
)


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestIsMiniMaxModel:
    """Test _is_minimax_model detection."""

    def test_minimax_prefix(self):
        assert _is_minimax_model("minimax/MiniMax-M2.7") is True

    def test_minimax_highspeed(self):
        assert _is_minimax_model("minimax/MiniMax-M2.7-highspeed") is True

    def test_minimax_m25(self):
        assert _is_minimax_model("minimax/MiniMax-M2.5") is True

    def test_openai_model_not_minimax(self):
        assert _is_minimax_model("openai/gpt-4.1") is False

    def test_anthropic_model_not_minimax(self):
        assert _is_minimax_model("anthropic/claude-sonnet-4-6") is False

    def test_bare_model_name_not_minimax(self):
        assert _is_minimax_model("MiniMax-M2.7") is False

    def test_empty_string(self):
        assert _is_minimax_model("") is False


class TestResolveMiniMaxModel:
    """Test _resolve_minimax_model model name extraction."""

    def test_m27(self):
        assert _resolve_minimax_model("minimax/MiniMax-M2.7") == "MiniMax-M2.7"

    def test_m27_highspeed(self):
        assert _resolve_minimax_model("minimax/MiniMax-M2.7-highspeed") == "MiniMax-M2.7-highspeed"

    def test_m25(self):
        assert _resolve_minimax_model("minimax/MiniMax-M2.5") == "MiniMax-M2.5"


class TestClampMiniMaxTemperature:
    """Test _clamp_minimax_temperature clamping logic."""

    def test_none_unchanged(self):
        assert _clamp_minimax_temperature(None) is None

    def test_valid_range_unchanged(self):
        assert _clamp_minimax_temperature(0.5) == 0.5

    def test_one_unchanged(self):
        assert _clamp_minimax_temperature(1.0) == 1.0

    def test_zero_clamped_up(self):
        assert _clamp_minimax_temperature(0.0) == 0.01

    def test_negative_clamped_up(self):
        assert _clamp_minimax_temperature(-0.5) == 0.01

    def test_above_one_clamped_down(self):
        assert _clamp_minimax_temperature(1.5) == 1.0

    def test_above_two_clamped_down(self):
        assert _clamp_minimax_temperature(2.0) == 1.0

    def test_small_positive_unchanged(self):
        assert _clamp_minimax_temperature(0.01) == 0.01

    def test_just_above_min(self):
        assert _clamp_minimax_temperature(0.02) == 0.02


class TestMiniMaxModelsRegistry:
    """Test that all expected MiniMax models are defined."""

    def test_m27_present(self):
        assert "MiniMax-M2.7" in MINIMAX_MODELS

    def test_m27_highspeed_present(self):
        assert "MiniMax-M2.7-highspeed" in MINIMAX_MODELS

    def test_m25_present(self):
        assert "MiniMax-M2.5" in MINIMAX_MODELS

    def test_m25_highspeed_present(self):
        assert "MiniMax-M2.5-highspeed" in MINIMAX_MODELS

    def test_all_have_max_context(self):
        for name, info in MINIMAX_MODELS.items():
            assert "max_context" in info, f"{name} missing max_context"
            assert info["max_context"] == 204_800, f"{name} has wrong max_context"


# ---------------------------------------------------------------------------
# Unit tests: LM class with MiniMax provider
# ---------------------------------------------------------------------------


class TestLMMiniMaxInit:
    """Test LM constructor with MiniMax model identifiers."""

    def test_minimax_model_rewritten_to_openai(self):
        lm = LM("minimax/MiniMax-M2.7")
        assert lm.model == "openai/MiniMax-M2.7"

    def test_minimax_api_base_auto_set(self):
        lm = LM("minimax/MiniMax-M2.7")
        assert lm.completion_kwargs.get("api_base") == MINIMAX_API_BASE

    def test_minimax_custom_api_base_preserved(self):
        custom_base = "https://custom.minimax.api/v1"
        lm = LM("minimax/MiniMax-M2.7", api_base=custom_base)
        assert lm.completion_kwargs["api_base"] == custom_base

    def test_minimax_temperature_clamped(self):
        lm = LM("minimax/MiniMax-M2.7", temperature=0.0)
        assert lm.completion_kwargs["temperature"] == 0.01

    def test_minimax_high_temperature_clamped(self):
        lm = LM("minimax/MiniMax-M2.7", temperature=1.5)
        assert lm.completion_kwargs["temperature"] == 1.0

    def test_minimax_valid_temperature_unchanged(self):
        lm = LM("minimax/MiniMax-M2.7", temperature=0.7)
        assert lm.completion_kwargs["temperature"] == 0.7

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-12345"})
    def test_minimax_api_key_from_env(self):
        lm = LM("minimax/MiniMax-M2.7")
        assert lm.completion_kwargs["api_key"] == "test-key-12345"

    def test_minimax_api_key_from_kwarg(self):
        lm = LM("minimax/MiniMax-M2.7", api_key="explicit-key")
        assert lm.completion_kwargs["api_key"] == "explicit-key"

    def test_minimax_max_tokens_forwarded(self):
        lm = LM("minimax/MiniMax-M2.7", max_tokens=4096)
        assert lm.completion_kwargs["max_tokens"] == 4096

    def test_openai_model_unchanged(self):
        lm = LM("openai/gpt-4.1")
        assert lm.model == "openai/gpt-4.1"
        assert "api_base" not in lm.completion_kwargs

    def test_minimax_highspeed_model(self):
        lm = LM("minimax/MiniMax-M2.7-highspeed")
        assert lm.model == "openai/MiniMax-M2.7-highspeed"
        assert lm.completion_kwargs.get("api_base") == MINIMAX_API_BASE


class TestLMMiniMaxCall:
    """Test LM __call__ with MiniMax models."""

    @patch("litellm.completion")
    def test_minimax_call_routes_correctly(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "MiniMax response"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("minimax/MiniMax-M2.7", temperature=0.7, api_key="test-key")
        result = lm("Solve this problem")

        assert result == "MiniMax response"
        mock_completion.assert_called_once_with(
            model="openai/MiniMax-M2.7",
            messages=[{"role": "user", "content": "Solve this problem"}],
            num_retries=3,
            drop_params=True,
            temperature=0.7,
            api_base=MINIMAX_API_BASE,
            api_key="test-key",
        )

    @patch("litellm.completion")
    def test_minimax_call_with_messages(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "chat response"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("minimax/MiniMax-M2.7", api_key="test-key")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = lm(messages)

        assert result == "chat response"
        call_kwargs = mock_completion.call_args
        assert call_kwargs[1]["model"] == "openai/MiniMax-M2.7"
        assert call_kwargs[1]["api_base"] == MINIMAX_API_BASE

    @patch("litellm.batch_completion")
    def test_minimax_batch_complete(self, mock_batch):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = " answer "
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("minimax/MiniMax-M2.7", temperature=0.5, api_key="test-key")
        msgs = [[{"role": "user", "content": "q1"}]]
        results = lm.batch_complete(msgs)

        assert results == ["answer"]
        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["model"] == "openai/MiniMax-M2.7"
        assert call_kwargs["api_base"] == MINIMAX_API_BASE
        assert call_kwargs["temperature"] == 0.5


class TestLMMiniMaxRepr:
    """Test LM __repr__ with MiniMax models."""

    def test_repr_shows_openai_prefix(self):
        lm = LM("minimax/MiniMax-M2.7", temperature=0.7)
        r = repr(lm)
        assert "openai/MiniMax-M2.7" in r
        assert "temperature=0.7" in r


# ---------------------------------------------------------------------------
# Integration tests (require MINIMAX_API_KEY environment variable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set — skipping live API integration tests",
)
class TestLMMiniMaxIntegration:
    """Integration tests that hit the real MiniMax API."""

    def test_simple_completion(self):
        lm = LM("minimax/MiniMax-M2.7", temperature=0.7, max_tokens=50)
        response = lm("Reply with exactly one word: hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_messages(self):
        lm = LM("minimax/MiniMax-M2.7", temperature=0.7, max_tokens=50)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with exactly one word: yes"},
        ]
        response = lm(messages)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_batch_complete(self):
        lm = LM("minimax/MiniMax-M2.7", temperature=0.7, max_tokens=256)
        msgs = [
            [{"role": "user", "content": "What is 1+1? Reply with just the number."}],
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        ]
        results = lm.batch_complete(msgs, max_workers=2)
        assert len(results) == 2
        assert all(isinstance(r, str) and len(r) > 0 for r in results)
