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

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
        # Concurrency for the underlying litellm.batch_completion call (passed
        # through to the LM's batch_complete); not an application-level pool.
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

    def _render(
        self,
        current_instruction_doc: str,
        dataset_with_feedback: Any,
        prompt_template: str | None,
    ) -> tuple[Any, list[dict[str, Any]]]:
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

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, StatelessReflectionLM]:
        # N=1 case of reflect_many — share the exact same code path.
        return self.reflect_many([(candidate, reflective_dataset, components_to_update)])[0]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, StatelessReflectionLM]]:
        # Flatten every (job, component) pair that has feedback into one list of
        # rendered prompts, issue them as a single vectorized completion
        # (litellm.batch_completion), then scatter the parsed results back into
        # one ReflectionProposal per job.
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

        return [(proposal, self) for proposal in proposals]

    def _batch_complete(self, prompts: list[Any], messages_list: list[list[dict[str, Any]]]) -> list[str]:
        """Issue the reflection completions, vectorized when possible.

        Prefers a single ``litellm.batch_completion`` via the LM's
        ``batch_complete``. A single call uses the plain completion path (so N=1
        is byte-identical to the historical single reflection); a custom callable
        with no batch API runs sequentially. No application-level threads.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.lm(prompts[0])]
        if hasattr(self.lm, "batch_complete"):
            return list(self.lm.batch_complete(messages_list, max_workers=self.max_workers))
        # Custom callable without a vectorized API → sequential completions.
        return [self.lm(prompt) for prompt in prompts]


class ComBEEReflectionLM(StatelessReflectionLM):
    """ComBEE map-shuffle-reduce reflection, vectorized (#307, arXiv:2604.04247).

    For each ``(job, component)`` with ``n`` reflection records: duplicate each
    record ``duplication_factor`` times and shuffle (augmented shuffle), split
    into ``k = ⌊√n⌋`` groups, reflect once per group (**Level-1 / Map**), then
    aggregate the ``k`` intermediate instructions into one (**Level-2 /
    Reduce**). When ``k ≤ 1`` (small ``n``) it degrades to a single stateless
    reflection — i.e. ``StatelessReflectionLM``.

    Both rounds are vectorized across the whole proposal batch: **all** Level-1
    group calls (≈ N·k) go out as one batched completion, then **all** Level-2
    reduces (≈ N) as a second. So a batched-ComBEE iteration is two
    ``batch_complete`` round-trips regardless of N.

    This is orthogonal to the memory axis: ComBEE is stateless in memory
    (``next_lm = self``); the aggregation wraps whatever ``lm`` it is given.
    """

    def __init__(
        self,
        lm: LanguageModel,
        reflection_prompt_template: str | dict[str, str] | None = None,
        aggregation_prompt_template: str | None = None,
        duplication_factor: int = 2,
        rng: random.Random | None = None,
        logger: Any | None = None,
        max_workers: int = 10,
    ):
        super().__init__(lm, reflection_prompt_template, logger, max_workers)
        self.duplication_factor = duplication_factor
        self.aggregation_prompt_template = (
            aggregation_prompt_template or InstructionProposalSignature.default_aggregation_prompt_template
        )
        self.rng = rng if rng is not None else random.Random(0)

    @staticmethod
    def _split(items: list[Any], k: int) -> list[list[Any]]:
        """Split ``items`` into ``k`` contiguous, balanced groups (ComBEE §3.1)."""
        base, remainder, offset, groups = len(items) // k, len(items) % k, 0, []
        for i in range(k):
            size = base + (1 if i < remainder else 0)
            groups.append(items[offset : offset + size])
            offset += size
        return groups

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, StatelessReflectionLM]]:
        # ---- Round 1 (Map): build every group prompt across all (job, component) ----
        l1: list[tuple[Any, list[dict[str, Any]]]] = []
        l1_key: list[tuple[int, str, str]] = []  # (job_idx, component, kind) kind in {"naive","group"}
        curr_doc: dict[tuple[int, str], str] = {}  # remember the parent text for the reduce step
        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            for name in components_to_update:
                if name not in reflective_dataset or not reflective_dataset.get(name):
                    self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                    continue
                records = list(reflective_dataset[name])
                k = max(1, int(len(records) ** 0.5))
                template = self._resolve_template(name)
                curr_doc[(job_idx, name)] = candidate[name]
                if k <= 1:
                    # Degenerate case: a single stateless reflection over the original records.
                    l1.append(self._render(candidate[name], records, template))
                    l1_key.append((job_idx, name, "naive"))
                    continue
                augmented = records * self.duplication_factor
                self.rng.shuffle(augmented)
                for group in self._split(augmented, k):
                    if not group:
                        continue
                    l1.append(self._render(candidate[name], group, template))
                    l1_key.append((job_idx, name, "group"))

        l1_raw = self._batch_complete([x[0] for x in l1], [x[1] for x in l1])

        # ---- Gather Level-1 results ----
        final: dict[tuple[int, str], tuple[str, Any, str]] = {}  # ready new_texts (naive / single-group)
        group_props: dict[tuple[int, str], list[str]] = defaultdict(list)
        for (job_idx, name, kind), (prompt, _m), raw in zip(l1_key, l1, l1_raw, strict=True):
            text = InstructionProposalSignature.output_extractor(raw.strip())["new_instruction"]
            if kind == "naive":
                final[(job_idx, name)] = (text, prompt, raw)
            else:
                group_props[(job_idx, name)].append(text)

        # ---- Round 2 (Reduce): one aggregation per (job, component) with ≥2 groups ----
        l2: list[tuple[Any, list[dict[str, Any]]]] = []
        l2_key: list[tuple[int, str]] = []
        for key, props in group_props.items():
            if len(props) == 1:
                final[key] = (props[0], "", "")  # only one surviving group → no reduce (#307)
                continue
            agg_records = [{"Proposed Instruction Update": p} for p in props]
            l2.append(self._render(curr_doc[key], agg_records, self.aggregation_prompt_template))
            l2_key.append(key)

        l2_raw = self._batch_complete([x[0] for x in l2], [x[1] for x in l2])
        for key, (prompt, _m), raw in zip(l2_key, l2, l2_raw, strict=True):
            text = InstructionProposalSignature.output_extractor(raw.strip())["new_instruction"]
            final[key] = (text, prompt, raw)

        # ---- Scatter into one proposal per job ----
        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}) for _ in jobs]
        for (job_idx, name), (text, prompt, raw) in final.items():
            proposals[job_idx].new_texts[name] = text
            proposals[job_idx].prompts[name] = prompt
            proposals[job_idx].raw_lm_outputs[name] = raw
        return [(proposal, self) for proposal in proposals]
