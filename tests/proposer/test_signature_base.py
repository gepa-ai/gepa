# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa.proposer.reflective_mutation.base import Signature, SignatureParseResult


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

    def test_adapter_can_supply_format_and_parse_without_changing_run_interface(self):
        class Adapter:
            def format(self, signature, input_dict):
                return f"adapter prompt: {input_dict['value']}"

            def parse(self, signature, lm_out):
                return SignatureParseResult.success({"adapted": lm_out.upper()})

        class AdapterSignature(Signature):
            adapter = Adapter()

        seen = []

        def lm(prompt):
            seen.append(prompt)
            return " response "

        assert AdapterSignature.run(lm, {"value": "x"}) == {"adapted": "RESPONSE"}
        assert seen == ["adapter prompt: x"]

    def test_legacy_override_takes_precedence_over_inherited_adapter(self):
        class FailingAdapter:
            def format(self, signature, input_dict):
                raise AssertionError("inherited adapter format should not replace an override")

            def parse(self, signature, lm_out):
                raise AssertionError("inherited adapter parser should not replace an override")

        class AdaptedBase(Signature):
            adapter = FailingAdapter()

        class LegacySubclass(AdaptedBase):
            @classmethod
            def prompt_renderer(cls, input_dict):
                return "legacy prompt"

            @classmethod
            def output_extractor(cls, lm_out):
                return {"legacy": lm_out}

        assert LegacySubclass.run(lambda prompt: " result ", {}) == {"legacy": "result"}
