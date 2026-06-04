# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the ReflectionLM abstraction (#329 Phase 1).

These pin the behavior-preserving contract: StatelessReflectionLM reproduces
the historical per-component reflection loop, and the proposer delegates to it
without changing observable behavior.
"""

from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionLM,
    ReflectionProposal,
    StatelessReflectionLM,
)


class RecordingLM:
    """Fake reflection LM: records prompts and returns a fenced instruction."""

    def __init__(self):
        self.calls: list[str | list[dict]] = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        # output_extractor pulls the text between the first/last ``` fences.
        return "```\nNEW INSTRUCTION\n```"


class ListLogger:
    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str):
        self.messages.append(message)


CANDIDATE = {"a": "old A", "b": "old B"}
REFLECTIVE = {
    "a": [{"input": "x", "feedback": "do better"}],
    "b": [{"input": "y", "feedback": "improve"}],
}


def test_is_a_reflection_lm():
    """StatelessReflectionLM satisfies the runtime-checkable protocol."""
    assert isinstance(StatelessReflectionLM(RecordingLM()), ReflectionLM)


def test_reflect_returns_new_texts_and_self():
    lm = RecordingLM()
    impl = StatelessReflectionLM(lm)

    proposal, next_lm = impl.reflect(CANDIDATE, REFLECTIVE, ["a", "b"])

    assert isinstance(proposal, ReflectionProposal)
    assert proposal.new_texts == {"a": "NEW INSTRUCTION", "b": "NEW INSTRUCTION"}
    # Metadata is captured per component.
    assert set(proposal.prompts) == {"a", "b"}
    assert set(proposal.raw_lm_outputs) == {"a", "b"}
    assert proposal.raw_lm_outputs["a"] == "```\nNEW INSTRUCTION\n```"
    # Stateless: the "extended" LM is the same object.
    assert next_lm is impl
    # One LM call per requested component.
    assert len(lm.calls) == 2


def test_skips_components_without_feedback():
    lm = RecordingLM()
    logger = ListLogger()
    impl = StatelessReflectionLM(lm, logger=logger)

    # "b" has no data in the reflective dataset → skipped, no LM call for it.
    proposal, _ = impl.reflect(CANDIDATE, {"a": REFLECTIVE["a"], "b": []}, ["a", "b"])

    assert proposal.new_texts == {"a": "NEW INSTRUCTION"}
    assert len(lm.calls) == 1
    assert any("Component 'b' is not in reflective dataset" in m for m in logger.messages)


def test_single_template_is_used_for_all_components():
    lm = RecordingLM()
    template = "INSTRUCTION: <curr_param>\nDATA: <side_info>"
    impl = StatelessReflectionLM(lm, reflection_prompt_template=template)

    impl.reflect(CANDIDATE, REFLECTIVE, ["a"])

    assert "INSTRUCTION: old A" in lm.calls[0]


def test_dict_template_warns_once_for_missing_component():
    lm = RecordingLM()
    logger = ListLogger()
    # Provide a template for "a" only; "b" should fall back to default + warn.
    templates = {"a": "A-ONLY: <curr_param> <side_info>"}
    impl = StatelessReflectionLM(lm, reflection_prompt_template=templates, logger=logger)

    impl.reflect(CANDIDATE, REFLECTIVE, ["a", "b"])
    impl.reflect(CANDIDATE, REFLECTIVE, ["a", "b"])  # second pass must not re-warn

    warnings = [m for m in logger.messages if "No reflection_prompt_template found for parameter 'b'" in m]
    assert len(warnings) == 1
    # "a" used its custom template.
    assert "A-ONLY: old A" in lm.calls[0]


def test_empty_components_yields_empty_proposal():
    impl = StatelessReflectionLM(RecordingLM())
    proposal, next_lm = impl.reflect(CANDIDATE, REFLECTIVE, [])
    assert proposal.new_texts == {}
    assert proposal.prompts == {}
    assert next_lm is impl
