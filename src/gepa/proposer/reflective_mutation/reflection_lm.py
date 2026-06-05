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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.instruction_proposal import InstructionProposalSignature

# One reflection job = (candidate, reflective_dataset, components_to_update).
ReflectionJob = tuple[dict[str, str], "Mapping[str, Sequence[Mapping[str, Any]]]", list[str]]


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


@runtime_checkable
class BatchReflectionLM(ReflectionLM, Protocol):
    """A :class:`ReflectionLM` that can reflect on a batch of jobs at once.

    ``reflect_many`` is the vectorized form of ``reflect``: given N independent
    jobs (one per candidate proposed this iteration), it returns N
    ``(proposal, next_lm)`` results in order.  Implementations are free to issue
    all the underlying LLM calls as a single batched/concurrent request — this
    is where GEPA's per-iteration proposal throughput comes from, replacing
    engine-level threading with one batched call at the reflection edge.

    ``reflect`` is just the N=1 case and must remain consistent with it.
    """

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ReflectionLM]]: ...


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
        max_workers: int = 10,
    ):
        self.lm = lm
        self.reflection_prompt_template = reflection_prompt_template
        self.logger = logger
        # Upper bound on concurrent reflection LM calls issued by reflect_many.
        self.max_workers = max_workers
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
        # N=1 case of reflect_many — share the exact same code path.
        return self.reflect_many([(candidate, reflective_dataset, components_to_update)])[0]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, StatelessReflectionLM]]:
        # Flatten every (job, component) pair that has feedback into a single
        # list of independent LM calls, run them in one batched/concurrent pass,
        # then scatter the results back into one ReflectionProposal per job.
        calls: list[tuple[int, str, dict[str, Any]]] = []
        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            for name in components_to_update:
                # Gracefully handle a selected component with no data in the reflective dataset.
                if name not in reflective_dataset or not reflective_dataset.get(name):
                    self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                    continue
                calls.append(
                    (
                        job_idx,
                        name,
                        {
                            "current_instruction_doc": candidate[name],
                            "dataset_with_feedback": reflective_dataset[name],
                            "prompt_template": self._resolve_template(name),
                        },
                    )
                )

        def run_one(call: tuple[int, str, dict[str, Any]]) -> tuple[int, str, str, Any, str]:
            job_idx, name, input_dict = call
            result, prompt, raw_output = InstructionProposalSignature.run_with_metadata(
                lm=self.lm, input_dict=input_dict
            )
            return job_idx, name, result["new_instruction"], prompt, raw_output

        # Short-circuit the single-call case so behavior is byte-identical to a
        # plain sequential reflect() (no thread, no executor overhead).
        if len(calls) <= 1:
            results = [run_one(c) for c in calls]
        else:
            with ThreadPoolExecutor(max_workers=min(len(calls), self.max_workers)) as executor:
                results = list(executor.map(run_one, calls))

        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}) for _ in jobs]
        for job_idx, name, new_text, prompt, raw_output in results:
            proposals[job_idx].new_texts[name] = new_text
            proposals[job_idx].prompts[name] = prompt
            proposals[job_idx].raw_lm_outputs[name] = raw_output

        return [(proposal, self) for proposal in proposals]
