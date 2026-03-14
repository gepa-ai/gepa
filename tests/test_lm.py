# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import LM, _is_reasoning_model


class TestIsReasoningModel:
    """Test reasoning model detection regex."""

    @pytest.mark.parametrize(
        "model",
        [
            "o1",
            "o1-mini",
            "o1-pro",
            "o1-mini-2025-01-01",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "openai/o1",
            "openai/o3-mini",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "openai/gpt-5-mini-2025-06-01",
        ],
    )
    def test_reasoning_models_detected(self, model):
        assert _is_reasoning_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-5-chat",
            "openai/gpt-4.1",
            "openai/gpt-5-chat",
            "anthropic/claude-sonnet-4-6",
            "claude-sonnet-4-6",
            "deepseek/deepseek-r1",
        ],
    )
    def test_non_reasoning_models_not_detected(self, model):
        assert _is_reasoning_model(model) is False


class TestLMInit:
    """Test LM constructor parameter handling."""

    def test_reasoning_model_defaults(self):
        lm = LM("openai/gpt-5-mini")
        assert lm.is_reasoning is True
        assert lm.completion_kwargs["temperature"] == 1.0
        assert lm.completion_kwargs["max_completion_tokens"] == 16000
        assert "max_tokens" not in lm.completion_kwargs

    def test_reasoning_model_custom_max_tokens(self):
        lm = LM("openai/gpt-5", max_tokens=32000)
        assert lm.completion_kwargs["max_completion_tokens"] == 32000

    def test_reasoning_model_temperature_1_ok(self):
        lm = LM("openai/gpt-5", temperature=1.0)
        assert lm.completion_kwargs["temperature"] == 1.0

    def test_reasoning_model_temperature_none_ok(self):
        lm = LM("o3-mini", temperature=None)
        assert lm.completion_kwargs["temperature"] == 1.0

    def test_reasoning_model_bad_temperature_raises(self):
        with pytest.raises(ValueError, match="requires temperature=1.0"):
            LM("openai/gpt-5", temperature=0.7)

    def test_standard_model_defaults(self):
        lm = LM("openai/gpt-4.1")
        assert lm.is_reasoning is False
        assert "temperature" not in lm.completion_kwargs
        assert "max_tokens" not in lm.completion_kwargs

    def test_standard_model_custom_params(self):
        lm = LM("openai/gpt-4.1", temperature=0.5, max_tokens=4096)
        assert lm.completion_kwargs["temperature"] == 0.5
        assert lm.completion_kwargs["max_tokens"] == 4096

    def test_extra_kwargs_forwarded(self):
        lm = LM("openai/gpt-4.1", top_p=0.9, stop=["\n"])
        assert lm.completion_kwargs["top_p"] == 0.9
        assert lm.completion_kwargs["stop"] == ["\n"]


class TestLMCall:
    """Test LM __call__ method."""

    @patch("litellm.completion")
    def test_string_prompt(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response text"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1", temperature=0.5)
        result = lm("hello")

        assert result == "response text"
        mock_completion.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": "hello"}],
            num_retries=3,
            drop_params=True,
            temperature=0.5,
        )

    @patch("litellm.completion")
    def test_messages_prompt(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "chat response"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = lm(messages)

        assert result == "chat response"
        mock_completion.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=messages,
            num_retries=3,
            drop_params=True,
        )

    @patch("litellm.completion")
    def test_reasoning_model_params(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "answer"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-5-mini")
        lm("solve this")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["max_completion_tokens"] == 16000
        assert "max_tokens" not in call_kwargs

    @patch("litellm.completion")
    def test_truncation_warning(self, mock_completion, caplog):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "truncated"
        mock_response.choices[0].finish_reason = "length"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1", max_tokens=100)
        result = lm("hello")

        assert result == "truncated"
        assert "truncated" in caplog.text.lower()


class TestLMBatchComplete:
    """Test LM batch_complete method."""

    @patch("litellm.batch_completion")
    def test_batch_complete(self, mock_batch):
        resp1 = MagicMock()
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.content = " answer1 "
        resp1.choices[0].finish_reason = "stop"
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.content = " answer2 "
        resp2.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp1, resp2]

        lm = LM("openai/gpt-4.1")
        msgs = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
        ]
        results = lm.batch_complete(msgs, max_workers=5)

        assert results == ["answer1", "answer2"]
        mock_batch.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=msgs,
            max_workers=5,
            num_retries=3,
            drop_params=True,
        )

    @patch("litellm.batch_completion")
    def test_batch_reasoning_model_params(self, mock_batch):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ans"
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("openai/gpt-5")
        lm.batch_complete([[{"role": "user", "content": "q"}]])

        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["max_completion_tokens"] == 16000


class TestLMRepr:
    def test_repr_standard(self):
        lm = LM("openai/gpt-4.1", temperature=0.5)
        assert "gpt-4.1" in repr(lm)
        assert "temperature=0.5" in repr(lm)

    def test_repr_reasoning(self):
        lm = LM("openai/gpt-5-mini")
        assert "reasoning=True" in repr(lm)


class TestLMConformsToProtocol:
    """Verify LM satisfies the LanguageModel protocol."""

    def test_callable(self):
        lm = LM("openai/gpt-4.1")
        assert callable(lm)
        assert hasattr(lm, "__call__")
