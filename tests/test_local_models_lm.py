# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the local / self-hosted model path documented in guides/local-models.md.

Every engine is reached the same way, ``LM("openai/<model>", api_base=..., api_key=...)``;
these check that the endpoint settings reach litellm for each engine's default URL.
No server is needed (litellm is mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from gepa.lm import LM

# (engine, model name as the server reports it, OpenAI-compatible base URL), as in the guide.
ENGINES = [
    ("vllm", "Qwen/Qwen3-8B", "http://localhost:8000/v1"),
    ("sglang", "Qwen/Qwen3-8B", "http://localhost:30000/v1"),
    ("ollama", "qwen3:8b", "http://localhost:11434/v1"),
    ("llama.cpp", "qwen3-8b", "http://localhost:8080/v1"),
]
ENGINE_IDS = [engine for engine, _, _ in ENGINES]


def _response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].finish_reason = "stop"
    return resp


@pytest.mark.parametrize(("engine", "model_name", "api_base"), ENGINES, ids=ENGINE_IDS)
@patch("litellm.completion")
def test_call_reaches_local_endpoint(mock_completion, engine, model_name, api_base):
    mock_completion.return_value = _response("hi")

    lm = LM(f"openai/{model_name}", api_base=api_base, api_key="EMPTY")

    assert lm("hello") == "hi"
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == f"openai/{model_name}"
    assert kwargs["api_base"] == api_base
    assert kwargs["api_key"] == "EMPTY"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(("engine", "model_name", "api_base"), ENGINES, ids=ENGINE_IDS)
@patch("litellm.batch_completion")
def test_default_adapter_batches_requests_to_local_endpoint(mock_batch, engine, model_name, api_base):
    """An LM passed as ``task_lm`` evaluates the whole batch in one parallel litellm call."""
    mock_batch.return_value = [_response("4"), _response("Paris")]
    lm = LM(f"openai/{model_name}", api_base=api_base, api_key="EMPTY")
    adapter = DefaultAdapter(model=lm, max_litellm_workers=4)

    batch = [
        {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
        {"input": "Capital of France?", "additional_context": {}, "answer": "Paris"},
    ]
    result = adapter.evaluate(batch, {"system_prompt": "Answer concisely."})

    assert result.scores == [1.0, 1.0]
    mock_batch.assert_called_once()
    kwargs = mock_batch.call_args.kwargs
    assert kwargs["model"] == f"openai/{model_name}"
    assert kwargs["api_base"] == api_base
    assert kwargs["api_key"] == "EMPTY"
    assert kwargs["max_workers"] == 4
    assert len(kwargs["messages"]) == 2
