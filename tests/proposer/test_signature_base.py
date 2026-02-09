# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.proposer.reflective_mutation.base import LanguageModel, Signature


class MockSignature(Signature):
    """Mock Signature implementation for testing."""

    @classmethod
    def prompt_renderer(cls, input_dict):
        return "test_prompt"

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        return {"output": lm_out}


class TestSignatureRun:
    """Test Signature.run() method with different LM response types."""

    def test_lm_returns_string(self):
        """Test that Signature.run() handles string responses correctly."""

        class MockLM:
            def __call__(self, prompt: str) -> str:
                return "  response text  "

        lm = MockLM()
        result = MockSignature.run(lm, {})

        assert result == {"output": "response text"}

    def test_lm_returns_list_with_one_element(self):
        """Test that Signature.run() handles list responses with one element."""

        class MockLM:
            def __call__(self, prompt: str) -> list[str]:
                return ["  response from list  "]

        lm = MockLM()
        result = MockSignature.run(lm, {})

        assert result == {"output": "response from list"}

    def test_lm_returns_tuple_with_one_element(self):
        """Test that Signature.run() handles tuple responses with one element."""

        class MockLM:
            def __call__(self, prompt: str) -> tuple[str, ...]:
                return ("  response from tuple  ",)

        lm = MockLM()
        result = MockSignature.run(lm, {})

        assert result == {"output": "response from tuple"}

    def test_lm_returns_list_with_multiple_elements(self):
        """Test that Signature.run() uses the first element of multi-element lists."""

        class MockLM:
            def __call__(self, prompt: str) -> list[str]:
                return ["  first response  ", "second response", "third response"]

        lm = MockLM()
        result = MockSignature.run(lm, {})

        assert result == {"output": "first response"}

    def test_lm_returns_empty_list(self):
        """Test that Signature.run() handles empty list edge case."""

        class MockLM:
            def __call__(self, prompt: str) -> list[str]:
                return []

        lm = MockLM()

        with pytest.raises(IndexError):
            MockSignature.run(lm, {})

    def test_lm_returns_empty_tuple(self):
        """Test that Signature.run() handles empty tuple edge case."""

        class MockLM:
            def __call__(self, prompt: str) -> tuple[str, ...]:
                return ()

        lm = MockLM()

        with pytest.raises(IndexError):
            MockSignature.run(lm, {})

    def test_prompt_renderer_called_with_input_dict(self):
        """Test that the prompt_renderer is called with the input_dict."""

        class TrackingSignature(MockSignature):
            called_with = None

            @classmethod
            def prompt_renderer(cls, input_dict):
                cls.called_with = input_dict
                return "test_prompt"

        class MockLM:
            def __call__(self, prompt: str) -> str:
                return "response"

        lm = MockLM()
        input_dict = {"key": "value"}
        TrackingSignature.run(lm, input_dict)

        assert TrackingSignature.called_with == input_dict

    def test_output_extractor_receives_stripped_output(self):
        """Test that output_extractor receives stripped output from LM."""

        class TrackingSignature(MockSignature):
            received_output = None

            @classmethod
            def output_extractor(cls, lm_out: str) -> dict[str, str]:
                cls.received_output = lm_out
                return {"output": lm_out}

        class MockLM:
            def __call__(self, prompt: str) -> str:
                return "  response with spaces  "

        lm = MockLM()
        TrackingSignature.run(lm, {})

        assert TrackingSignature.received_output == "response with spaces"
