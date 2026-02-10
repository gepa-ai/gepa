# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa.proposer.reflective_mutation.base import Signature


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
