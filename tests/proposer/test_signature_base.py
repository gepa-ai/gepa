# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import ClassVar

import pytest

from gepa.proposer.reflective_mutation.base import Signature, SignatureParseResult
from gepa.strategies.instruction_proposal import InstructionProposalSignature


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

    @pytest.mark.parametrize("with_metadata", [False, True])
    def test_inherited_prompt_renderer_takes_precedence_over_adapter(self, with_metadata):
        class Adapter:
            def format(self, signature, input_dict):
                raise AssertionError("adapter format should not replace an inherited override")

            def parse(self, signature, lm_out):
                return SignatureParseResult.success({"output": lm_out})

        class LegacySignature(Signature):
            adapter = Adapter()
            prompt_template = "parent"

            @classmethod
            def prompt_renderer(cls, input_dict):
                return f"{cls.prompt_template}: {input_dict['value']}"

        class ChildSignature(LegacySignature):
            prompt_template = "child"

        seen = []

        def lm(prompt):
            seen.append(prompt)
            return " result "

        if with_metadata:
            assert ChildSignature.run_with_metadata(lm, {"value": "x"}) == (
                {"output": "result"},
                "child: x",
                "result",
            )
        else:
            assert ChildSignature.run(lm, {"value": "x"}) == {"output": "result"}
        assert seen == ["child: x"]

    @pytest.mark.parametrize("with_metadata", [False, True])
    def test_inherited_output_extractor_takes_precedence_over_adapter(self, with_metadata):
        class Adapter:
            def format(self, signature, input_dict):
                return "adapter prompt"

            def parse(self, signature, lm_out):
                raise AssertionError("adapter parser should not replace an inherited override")

        class LegacySignature(Signature):
            adapter = Adapter()
            output_keys: ClassVar[list[str]] = ["parent"]

            @classmethod
            def output_extractor(cls, lm_out):
                return {cls.output_keys[0]: lm_out}

        class ChildSignature(LegacySignature):
            output_keys: ClassVar[list[str]] = ["child"]

        if with_metadata:
            assert ChildSignature.run_with_metadata(lambda prompt: " result ", {}) == (
                {"child": "result"},
                "adapter prompt",
                "result",
            )
        else:
            assert ChildSignature.run(lambda prompt: " result ", {}) == {"child": "result"}

    @pytest.mark.parametrize("with_metadata", [False, True])
    def test_instruction_proposal_subclass_inherits_custom_output_extractor(self, with_metadata):
        class CustomProposal(InstructionProposalSignature):
            @classmethod
            def output_extractor(cls, lm_out):
                return {"new_instruction": f"custom: {lm_out}"}

        class ChildProposal(CustomProposal):
            pass

        inputs = {"current_instruction_doc": "seed", "dataset_with_feedback": []}
        if with_metadata:
            output, prompt, raw_output = ChildProposal.run_with_metadata(lambda prompt: " result ", inputs)
            assert output == {"new_instruction": "custom: result"}
            assert prompt == ChildProposal.prompt_renderer(inputs)
            assert raw_output == "result"
        else:
            assert ChildProposal.run(lambda prompt: " result ", inputs) == {"new_instruction": "custom: result"}
