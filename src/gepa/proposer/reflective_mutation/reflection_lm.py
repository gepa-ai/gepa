# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Reflection LM abstraction (Phase 1 of the unified reflection LM pool, #329).

GEPA currently drives reflection through a single stateless ``reflection_lm``
callable.  Several in-flight efforts (stateful sessions, coding agents, ComBEE
aggregation, parallel proposals) each need a richer reflection primitive, and
have been inventing their own incompatible abstractions.

This module introduces the unifying concept from issue #329: a
:class:`ReflectionLM` that, given a candidate and its reflective dataset,
proposes new component texts **and** returns the (possibly extended) reflection
LM to use next.  The stateless default (:class:`StatelessReflectionLM`) returns
itself unchanged, so behavior is identical to today.

This is intentionally a behavior-preserving refactor: it defines the protocol,
ships the stateless implementation, and routes the existing proposer through it.
Sessions, agents, and ComBEE become additional ``ReflectionLM`` implementations
in later phases, instead of bespoke branches in the engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.instruction_proposal import InstructionProposalSignature


@dataclass
class ReflectionProposal:
    """Output of a single :meth:`ReflectionLM.reflect` call.

    ``new_texts`` is the primary result — a mapping of component name to its
    proposed new text.  ``prompts`` and ``raw_lm_outputs`` are optional
    diagnostics (keyed by component) surfaced to callbacks and experiment
    trackers; implementations that don't issue a direct LM call (e.g. agents)
    may leave them empty.
    """

    new_texts: dict[str, str]
    prompts: dict[str, str | list[dict[str, Any]]] = field(default_factory=dict)
    raw_lm_outputs: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ReflectionLM(Protocol):
    """A reflection primitive that proposes new component texts.

    ``reflect`` returns ``(proposal, next_lm)`` where ``next_lm`` is the
    reflection LM to use for the following call.  Stateless implementations
    return ``self``; stateful ones (sessions, agents) return an extended copy
    that has incorporated this call's context, leaving the original untouched
    so it can be safely reused/forked.  See issue #329.
    """

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ReflectionLM]: ...


class StatelessReflectionLM:
    """The default reflection LM: one stateless LM call per component.

    Wraps a plain ``reflection_lm`` callable and reproduces GEPA's historical
    behavior — for each component with feedback, render the instruction-proposal
    prompt (honoring a global or per-component template) and parse the new
    instruction from the response.  ``reflect`` returns ``self`` as the next LM
    because there is no carried state.
    """

    def __init__(
        self,
        lm: LanguageModel,
        reflection_prompt_template: str | dict[str, str] | None = None,
        logger: Any | None = None,
    ):
        self.lm = lm
        self.reflection_prompt_template = reflection_prompt_template
        self.logger = logger
        # Track components for which we've already warned about a missing
        # per-component template, so the warning fires at most once each.
        self._missing_template_warnings: set[str] = set()

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    def _resolve_template(self, name: str) -> str | None:
        if isinstance(self.reflection_prompt_template, dict):
            template = self.reflection_prompt_template.get(name)
            if template is None and name not in self._missing_template_warnings:
                self._log(f"No reflection_prompt_template found for parameter '{name}'. Using default template.")
                self._missing_template_warnings.add(name)
            return template
        return self.reflection_prompt_template

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, StatelessReflectionLM]:
        new_texts: dict[str, str] = {}
        prompts: dict[str, str | list[dict[str, Any]]] = {}
        raw_lm_outputs: dict[str, str] = {}

        for name in components_to_update:
            # Gracefully handle a selected component with no data in the reflective dataset.
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue

            result, prompt, raw_output = InstructionProposalSignature.run_with_metadata(
                lm=self.lm,
                input_dict={
                    "current_instruction_doc": candidate[name],
                    "dataset_with_feedback": reflective_dataset[name],
                    "prompt_template": self._resolve_template(name),
                },
            )
            new_texts[name] = result["new_instruction"]
            prompts[name] = prompt
            raw_lm_outputs[name] = raw_output

        return ReflectionProposal(new_texts=new_texts, prompts=prompts, raw_lm_outputs=raw_lm_outputs), self
