# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import LM


class TestLMInit:
    """Test LM constructor parameter handling."""

    def test_defaults(self):
        lm = LM("openai/gpt-4.1")
        assert "temperature" not in lm.completion_kwargs
        assert "max_tokens" not in lm.completion_kwargs

    def test_custom_params(self):
        lm = LM("openai/gpt-4.1", temperature=0.5, max_tokens=4096)
        assert lm.completion_kwargs["temperature"] == 0.5
        assert lm.completion_kwargs["max_tokens"] == 4096

    def test_extra_kwargs_forwarded(self):
        lm = LM("openai/gpt-4.1", top_p=0.9, stop=["\n"])
        assert lm.completion_kwargs["top_p"] == 0.9
        assert lm.completion_kwargs["stop"] == ["\n"]

    def test_reasoning_model_no_special_treatment(self):
        """Reasoning models should NOT get special parameter handling."""
        lm = LM("openai/gpt-5-mini", temperature=0.7, max_tokens=4096)
        assert lm.completion_kwargs["temperature"] == 0.7
        assert lm.completion_kwargs["max_tokens"] == 4096
        assert "max_completion_tokens" not in lm.completion_kwargs


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
    def test_batch_complete_forwards_extra_kwargs(self, mock_batch):
        """Extra kwargs passed to batch_complete should be forwarded to litellm."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ans"
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("openai/gpt-4.1", temperature=0.5)
        lm.batch_complete(
            [[{"role": "user", "content": "q"}]],
            max_workers=3,
            timeout=30,
            api_base="https://custom.api",
        )

        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30
        assert call_kwargs["api_base"] == "https://custom.api"

    @patch("litellm.batch_completion")
    def test_batch_complete_kwargs_override_init(self, mock_batch):
        """Kwargs passed at call time should override init kwargs."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ans"
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("openai/gpt-4.1", temperature=0.5)
        lm.batch_complete(
            [[{"role": "user", "content": "q"}]],
            temperature=0.9,  # override init value
        )

        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["temperature"] == 0.9


class TestLMRepr:
    def test_repr(self):
        lm = LM("openai/gpt-4.1", temperature=0.5)
        assert "gpt-4.1" in repr(lm)
        assert "temperature=0.5" in repr(lm)


class TestLMConformsToProtocol:
    """Verify LM satisfies the LanguageModel protocol."""

    def test_callable(self):
        lm = LM("openai/gpt-4.1")
        assert callable(lm)
        assert hasattr(lm, "__call__")
