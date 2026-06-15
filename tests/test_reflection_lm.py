# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the ReflectionLM protocol and StatelessReflectionLM (#329 Phase 1)."""

from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionLM,
    ReflectionProposal,
    StatelessReflectionLM,
)


class RecordingLM:
    """A fake reflection LM: records prompts, returns a fenced instruction."""

    def __init__(self, reply: str = "improved instruction"):
        self.reply = reply
        self.calls: list = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return f"Here is the update:\n```\n{self.reply}\n```"


class CollectingLogger:
    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def _reflective_dataset(components):
    return {name: [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "bad"}] for name in components}


def test_stateless_reflection_lm_satisfies_protocol():
    lm = StatelessReflectionLM(RecordingLM())
    assert isinstance(lm, ReflectionLM)


def test_reflect_returns_self_as_next_lm():
    """Stateless: the next LM is the same object (no carried context)."""
    lm = StatelessReflectionLM(RecordingLM())
    candidate = {"system_prompt": "old"}
    proposal, next_lm = lm.reflect(candidate, _reflective_dataset(["system_prompt"]), ["system_prompt"])
    assert isinstance(proposal, ReflectionProposal)
    assert next_lm is lm


def test_reflect_proposes_new_text_per_component():
    fake = RecordingLM(reply="brand new prompt")
    lm = StatelessReflectionLM(fake)
    candidate = {"a": "old_a", "b": "old_b"}
    proposal, _ = lm.reflect(candidate, _reflective_dataset(["a", "b"]), ["a", "b"])
    assert proposal.new_texts == {"a": "brand new prompt", "b": "brand new prompt"}
    # Diagnostics populated per component.
    assert set(proposal.prompts.keys()) == {"a", "b"}
    assert set(proposal.raw_lm_outputs.keys()) == {"a", "b"}
    assert len(fake.calls) == 2


def test_reflect_skips_components_without_feedback():
    fake = RecordingLM()
    logger = CollectingLogger()
    lm = StatelessReflectionLM(fake, logger=logger)
    candidate = {"a": "old_a", "b": "old_b"}
    # Only 'a' has data; 'b' is requested but absent from the reflective dataset.
    reflective_dataset = _reflective_dataset(["a"])
    proposal, _ = lm.reflect(candidate, reflective_dataset, ["a", "b"])
    assert set(proposal.new_texts.keys()) == {"a"}
    assert len(fake.calls) == 1
    assert any("'b' is not in reflective dataset" in m for m in logger.messages)


def test_per_component_template_dict_and_missing_warning():
    fake = RecordingLM()
    logger = CollectingLogger()
    # Template provided for 'a' only; 'b' should warn once and fall back to default.
    template_a = "Improve <curr_param> using this feedback: <side_info>"
    lm = StatelessReflectionLM(fake, reflection_prompt_template={"a": template_a}, logger=logger)
    candidate = {"a": "old_a", "b": "old_b"}
    reflective_dataset = _reflective_dataset(["a", "b"])
    lm.reflect(candidate, reflective_dataset, ["a", "b"])
    # Warn again on a second call — but only once per component overall.
    lm.reflect(candidate, reflective_dataset, ["a", "b"])
    missing_warnings = [m for m in logger.messages if "No reflection_prompt_template found for parameter 'b'" in m]
    assert len(missing_warnings) == 1
