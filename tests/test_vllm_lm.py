# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for make_vllm_lm — the local vLLM / Hugging Face convenience helper.

These verify GEPA-side configuration and argument forwarding only; they never
require a running vLLM server (litellm is mocked).
"""

from unittest.mock import MagicMock, patch

from gepa.lm import LM, make_vllm_lm


class TestMakeVllmLMConfig:
    """Construction: prefixing, defaults, and argument forwarding."""

    def test_returns_lm_instance(self):
        lm = make_vllm_lm("qwen2.5-7b-instruct")
        assert isinstance(lm, LM)

    def test_adds_openai_prefix_when_missing(self):
        lm = make_vllm_lm("qwen2.5-7b-instruct")
        assert lm.model == "openai/qwen2.5-7b-instruct"

    def test_preserves_existing_openai_prefix(self):
        lm = make_vllm_lm("openai/qwen2.5-7b-instruct")
        assert lm.model == "openai/qwen2.5-7b-instruct"

    def test_preserves_hosted_vllm_prefix(self):
        lm = make_vllm_lm("hosted_vllm/qwen2.5-7b-instruct")
        assert lm.model == "hosted_vllm/qwen2.5-7b-instruct"

    def test_served_name_with_slash_is_prefixed(self):
        # A served name that itself contains "/" (e.g. the HF repo id) still gets
        # the provider prefix; litellm splits provider on the first "/".
        lm = make_vllm_lm("Qwen/Qwen2.5-7B-Instruct")
        assert lm.model == "openai/Qwen/Qwen2.5-7B-Instruct"

    def test_default_api_base_and_key(self):
        lm = make_vllm_lm("qwen2.5-7b-instruct")
        assert lm.completion_kwargs["api_base"] == "http://localhost:8000/v1"
        assert lm.completion_kwargs["api_key"] == "EMPTY"

    def test_custom_api_base_and_key_forwarded(self):
        lm = make_vllm_lm("m", api_base="http://gpu-box:9000/v1", api_key="sk-abc")
        assert lm.completion_kwargs["api_base"] == "http://gpu-box:9000/v1"
        assert lm.completion_kwargs["api_key"] == "sk-abc"

    def test_sampling_params_forwarded(self):
        lm = make_vllm_lm("m", temperature=0.3, max_tokens=256, top_p=0.9)
        assert lm.completion_kwargs["temperature"] == 0.3
        assert lm.completion_kwargs["max_tokens"] == 256
        assert lm.completion_kwargs["top_p"] == 0.9

    def test_num_retries_forwarded(self):
        lm = make_vllm_lm("m", num_retries=7)
        assert lm.num_retries == 7
        # num_retries is an LM constructor arg, not a completion kwarg.
        assert "num_retries" not in lm.completion_kwargs


class TestMakeVllmLMCall:
    """Runtime: api_base / api_key actually reach litellm."""

    @patch("litellm.completion")
    def test_call_forwards_endpoint(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hi"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = make_vllm_lm("qwen2.5-7b-instruct", api_base="http://localhost:8000/v1")
        result = lm("hello")

        assert result == "hi"
        _, kwargs = mock_completion.call_args
        assert kwargs["model"] == "openai/qwen2.5-7b-instruct"
        assert kwargs["api_base"] == "http://localhost:8000/v1"
        assert kwargs["api_key"] == "EMPTY"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]

    @patch("litellm.batch_completion")
    def test_batch_complete_forwards_endpoint(self, mock_batch):
        def _resp(text):
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = text
            r.choices[0].finish_reason = "stop"
            return r

        mock_batch.return_value = [_resp("a"), _resp("b")]

        lm = make_vllm_lm("qwen2.5-7b-instruct", api_base="http://gpu:8000/v1")
        results = lm.batch_complete([[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]])

        assert results == ["a", "b"]
        _, kwargs = mock_batch.call_args
        assert kwargs["model"] == "openai/qwen2.5-7b-instruct"
        assert kwargs["api_base"] == "http://gpu:8000/v1"
        assert kwargs["api_key"] == "EMPTY"


def test_reexported_from_optimize_anything():
    import gepa.optimize_anything as oa

    assert hasattr(oa, "make_vllm_lm")
    assert oa.make_vllm_lm is make_vllm_lm
