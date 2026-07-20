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

import json
import re
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


@dataclass(frozen=True)
class _TextualGradientDraft:
    text: str
    prompt: str | list[dict[str, Any]]
    raw_output: str


def format_reflective_examples_for_textual_gradient_prompt(
    dataset_with_feedback: Sequence[Mapping[str, Any]],
) -> str:
    """Render reflective feedback records for a text-only ranking prompt."""
    parts = []
    for idx, example in enumerate(dataset_with_feedback, start=1):
        parts.append(f"Example {idx}:\n{json.dumps(example, indent=2, default=str)}")
    return "\n".join(parts) or "(no feedback records)"


def build_textual_gradient_selection_prompt(
    *,
    component_name: str,
    component_type: str,
    previous_text: str,
    dataset_with_feedback: Sequence[Mapping[str, Any]],
    proposals: Sequence[str],
) -> str:
    dataset_text = format_reflective_examples_for_textual_gradient_prompt(dataset_with_feedback)
    candidate_block = "\n\n".join(f"Candidate {idx + 1}:\n{proposal.strip()}" for idx, proposal in enumerate(proposals))
    return (
        f"You are ranking candidate {component_type} updates for the GEPA component '{component_name}'.\n"
        "Consider the current version, the execution feedback, and each candidate update. "
        "Select the option that best addresses the feedback while preserving useful behavior.\n\n"
        f"Current {component_type}:\n{previous_text}\n\n"
        f"Feedback from recent executions:\n{dataset_text}\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        "Reply on the first two lines using the template:\n"
        "BEST_INDEX: <number>\n"
        "REASON: <short explanation>\n"
        "Do not include any other structured text before these lines."
    )


def extract_best_textual_gradient_index(response_text: str, max_index: int) -> int:
    """Parse a 1-indexed BEST_INDEX response."""
    match = re.search(r"best_index\s*[:=]\s*(\d+)", response_text, flags=re.IGNORECASE)
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            return idx

    fallback_numbers = re.findall(r"\d+", response_text)
    for number in fallback_numbers:
        idx = int(number)
        if 1 <= idx <= max_index:
            return idx

    raise ValueError(f"Could not parse BEST_INDEX from response: {response_text!r}")


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
    """Default reflection LM.

    The default path makes one stateless LM call per component, or one batched
    call covering all tasks/components when the underlying LM provides
    ``batch_complete``. ``best_of_n`` samples several textual evolutions and
    uses the reflection LM to rank them. ``num_iterations`` applies that
    evolution loop repeatedly before returning the final text.

    For each component with feedback, render the instruction-proposal prompt
    (honoring a global or per-component template) and parse the new instruction.
    ``reflect`` returns ``self`` — there is no carried state.
    """

    def __init__(
        self,
        lm: LanguageModel,
        reflection_prompt_template: str | dict[str, str] | None = None,
        logger: Any | None = None,
        best_of_n: int = 1,
        num_iterations: int = 1,
    ):
        if best_of_n < 1:
            raise ValueError("best_of_n must be >= 1.")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1.")
        self.lm = lm
        self.reflection_prompt_template = reflection_prompt_template
        self.logger = logger
        self.best_of_n = best_of_n
        self.num_iterations = num_iterations
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
        # Each textual-gradient iteration depends on the text selected by the
        # previous iteration, so iterations are sequential. Within one iteration,
        # all jobs, components, and Best-of-N samples are still batched together.
        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}) for _ in jobs]
        current_candidates = [dict(candidate) for candidate, _reflective_dataset, _components in jobs]

        for iteration_idx in range(self.num_iterations):
            rendered: list[tuple[int, str, str | list[dict[str, Any]], list[dict[str, Any]]]] = []
            for job_idx, (_candidate, reflective_dataset, components_to_update) in enumerate(jobs):
                for name in components_to_update:
                    # Gracefully handle a selected component with no data in the reflective dataset.
                    if name not in reflective_dataset or not reflective_dataset.get(name):
                        if iteration_idx == 0:
                            self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                        continue
                    prompt, messages = self._render(
                        current_candidates[job_idx][name],
                        reflective_dataset[name],
                        self._resolve_template(name),
                    )
                    for _sample_idx in range(self.best_of_n):
                        rendered.append((job_idx, name, prompt, messages))

            raw_outputs = self._batch_complete([r[2] for r in rendered], [r[3] for r in rendered])

            drafts_by_component: dict[tuple[int, str], list[_TextualGradientDraft]] = {}
            for (job_idx, name, prompt, _messages), raw_output in zip(rendered, raw_outputs, strict=True):
                new_instruction = InstructionProposalSignature.output_extractor(raw_output.strip())["new_instruction"]
                drafts_by_component.setdefault((job_idx, name), []).append(
                    _TextualGradientDraft(text=new_instruction, prompt=prompt, raw_output=raw_output)
                )

            selected: dict[
                tuple[int, str],
                tuple[int, list[_TextualGradientDraft], _TextualGradientDraft, str | None, str | None],
            ] = {}
            selection_requests: list[tuple[int, str, str, list[dict[str, str]], list[_TextualGradientDraft]]] = []
            for (job_idx, name), drafts in drafts_by_component.items():
                if len(drafts) == 1 or self.best_of_n == 1:
                    selected[(job_idx, name)] = (1, drafts, drafts[0], None, None)
                    continue

                _candidate, reflective_dataset, _components_to_update = jobs[job_idx]
                selection_prompt = build_textual_gradient_selection_prompt(
                    component_name=name,
                    component_type="instruction",
                    previous_text=current_candidates[job_idx][name],
                    dataset_with_feedback=reflective_dataset[name],
                    proposals=[draft.text for draft in drafts],
                )
                selection_requests.append(
                    (job_idx, name, selection_prompt, [{"role": "user", "content": selection_prompt}], drafts)
                )

            selection_outputs = self._batch_complete(
                [request[2] for request in selection_requests],
                [request[3] for request in selection_requests],
            )
            for (job_idx, name, selection_prompt, _messages, drafts), selection_raw_output in zip(
                selection_requests, selection_outputs, strict=True
            ):
                try:
                    selected_index = extract_best_textual_gradient_index(selection_raw_output, len(drafts))
                except Exception as exc:
                    self._log(
                        f"Failed to pick best candidate for {name} due to {exc}. Defaulting to the first proposal."
                    )
                    selected_index = 1
                selected[(job_idx, name)] = (
                    selected_index,
                    drafts,
                    drafts[selected_index - 1],
                    selection_prompt,
                    selection_raw_output,
                )

            for (job_idx, name), (
                selected_index,
                drafts,
                selected_draft,
                selection_prompt,
                selection_raw_output,
            ) in selected.items():
                current_candidates[job_idx][name] = selected_draft.text
                proposals[job_idx].new_texts[name] = selected_draft.text
                proposals[job_idx].prompts[name] = selected_draft.prompt
                proposals[job_idx].raw_lm_outputs[name] = selected_draft.raw_output

                if self.best_of_n > 1 or self.num_iterations > 1:
                    iteration_record: dict[str, Any] = {
                        "iteration": iteration_idx + 1,
                        "proposal_texts": [draft.text for draft in drafts],
                        "proposal_raw_lm_outputs": [draft.raw_output for draft in drafts],
                        "selected_index": selected_index,
                    }
                    if selection_prompt is not None:
                        iteration_record["selection_prompt"] = selection_prompt
                    if selection_raw_output is not None:
                        iteration_record["selection_raw_lm_output"] = selection_raw_output
                    proposals[job_idx].metadata.setdefault(f"textual_gradient:{name}:iterations", []).append(
                        iteration_record
                    )

        # Stateless: the next LM is this same object (no carried context).
        return [(proposal, self) for proposal in proposals]
