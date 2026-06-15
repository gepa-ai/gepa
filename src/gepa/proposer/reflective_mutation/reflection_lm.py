# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Reflection LM abstraction (#329 Phase 1).

A ``ReflectionLM`` proposes new component texts and returns the (possibly
extended) reflection LM to use next.  The stateless default
(:class:`StatelessReflectionLM`) returns itself, so behavior is identical to
GEPA's historical single-callable reflection.  Sessions, agents, and ComBEE
become additional implementations in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.instruction_proposal import InstructionProposalSignature


@dataclass
class ReflectionProposal:
    """Output of one :meth:`ReflectionLM.reflect` call.

    ``new_texts`` maps component name -> proposed text.  ``prompts`` and
    ``raw_lm_outputs`` are optional per-component diagnostics for callbacks and
    experiment trackers.
    """

    new_texts: dict[str, str]
    prompts: dict[str, str | list[dict[str, Any]]] = field(default_factory=dict)
    raw_lm_outputs: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ReflectionLM(Protocol):
    """Proposes new component texts.

    ``reflect`` returns ``(proposal, next_lm)``: the reflection LM to use next.
    Stateless implementations return ``self``; stateful ones return an extended
    copy, leaving the original reusable/forkable.  See #329.
    """

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ReflectionLM]: ...


class StatelessReflectionLM:
    """Default reflection LM: one stateless LM call per component.

    For each component with feedback, render the instruction-proposal prompt
    (honoring a global or per-component template) and parse the new instruction.
    ``reflect`` returns ``self`` — there is no carried state.
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
        # Components already warned about a missing per-component template (warn once).
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

        # Stateless: the next LM is this same object (no carried context).
        return ReflectionProposal(new_texts=new_texts, prompts=prompts, raw_lm_outputs=raw_lm_outputs), self
