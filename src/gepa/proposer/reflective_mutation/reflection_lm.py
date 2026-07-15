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

# One reflection job = (candidate, reflective_dataset, components_to_update).
ReflectionJob = tuple[dict[str, str], "Mapping[str, Sequence[Mapping[str, Any]]]", list[str]]


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
    # Free-form diagnostics for multi-call reflection strategies. ``prompts``/
    # ``raw_lm_outputs`` assume ONE LM call per component; strategies that make
    # several (e.g. ComBEE's k map calls + 1 reduce call per component) should
    # record per-call intermediates here (namespaced keys, e.g.
    # "combee:level1_prompts"). Merged into CandidateProposal.metadata, so it
    # reaches on_proposal_end consumers, experiment trackers, and the run
    # manifest without a future protocol break.
    metadata: dict[str, Any] = field(default_factory=dict)


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


@runtime_checkable
class BatchReflectionLM(ReflectionLM, Protocol):
    """A :class:`ReflectionLM` that can reflect on many jobs in one batched call.

    ``reflect_many`` is the vectorized form of ``reflect``: given N independent
    jobs (one per candidate proposed this iteration), it returns N
    ``(proposal, next_lm)`` results in order.  Implementations issue the
    underlying LLM calls as a single batched/concurrent request (e.g.
    ``litellm.batch_completion``), so per-iteration proposal throughput comes
    from one batched call at the reflection edge rather than engine threads.

    ``reflect`` is just the N=1 case and must stay consistent with it.
    """

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ReflectionLM]]: ...


class StatelessReflectionLM:
    """Default reflection LM: one stateless LM call per component (or one batched call covering all tasks/components when the underlying LM provides ``batch_complete``).

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

    def _render(self, current_instruction_doc: str, dataset_with_feedback: Any, prompt_template: str | None):
        """Render a reflection prompt and its chat-messages form."""
        prompt = InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": current_instruction_doc,
                "dataset_with_feedback": dataset_with_feedback,
                "prompt_template": prompt_template,
            }
        )
        # Normalize to a chat-messages list (matches gepa.lm.LM.__call__).
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return prompt, messages

    def _batch_complete(self, prompts: list[Any], messages_list: list[list[dict[str, Any]]]) -> list[str]:
        """Issue the reflection completions, batched when possible.

        A single prompt uses the plain completion path, so N=1 is byte-identical
        to the historical single reflection.  When the LM exposes
        ``batch_complete`` (``litellm.batch_completion``), all prompts go out as
        one concurrent request; a custom callable without it runs sequentially.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.lm(prompts[0])]
        batch_complete = getattr(self.lm, "batch_complete", None)
        if batch_complete is not None:
            return list(batch_complete(messages_list))
        return [self.lm(prompt) for prompt in prompts]

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, StatelessReflectionLM]:
        # N=1 case of reflect_many — same code path, so behavior is identical.
        return self.reflect_many([(candidate, reflective_dataset, components_to_update)])[0]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, StatelessReflectionLM]]:
        # Flatten every (job, component) with feedback into one list of rendered
        # prompts, issue them as a single batched completion, then scatter the
        # parsed results back into one ReflectionProposal per job.
        rendered: list[tuple[int, str, Any, list[dict[str, Any]]]] = []
        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            for name in components_to_update:
                # Gracefully handle a selected component with no data in the reflective dataset.
                if name not in reflective_dataset or not reflective_dataset.get(name):
                    self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                    continue
                prompt, messages = self._render(candidate[name], reflective_dataset[name], self._resolve_template(name))
                rendered.append((job_idx, name, prompt, messages))

        raw_outputs = self._batch_complete([r[2] for r in rendered], [r[3] for r in rendered])

        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}) for _ in jobs]
        for (job_idx, name, prompt, _messages), raw_output in zip(rendered, raw_outputs, strict=True):
            new_instruction = InstructionProposalSignature.output_extractor(raw_output.strip())["new_instruction"]
            proposals[job_idx].new_texts[name] = new_instruction
            proposals[job_idx].prompts[name] = prompt
            proposals[job_idx].raw_lm_outputs[name] = raw_output

        # Stateless: the next LM is this same object (no carried context).
        return [(proposal, self) for proposal in proposals]
