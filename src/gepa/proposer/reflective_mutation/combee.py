# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""ComBEE parallel scan aggregation as a :class:`ReflectionLM` (#329 Phase 3).

ComBEE (arXiv:2604.04247, §3.1-3.2) scales reflection to large minibatches by
replacing the single monolithic reflection call with a map/reduce scheme, per
component:

1. **Augmented shuffle** (§3.2): duplicate each reflection record ``p`` times
   (``duplication_factor``) and shuffle the augmented list with a seeded RNG.
2. **Split** into ``k = ⌊√n⌋`` groups (``n`` = number of original records).
3. **Level-1 (Map)**: one reflection-LM call per group, each proposing an
   updated instruction from its subset of feedback.
4. **Level-2 (Reduce)**: one final LM call synthesizing the ``k`` intermediate
   proposals into a single instruction (``aggregation_prompt_template``).

With ``n < 4`` (so ``k = 1``) ComBEE degenerates to the standard single-call
reflection, making it a drop-in replacement that only activates when the
reflective dataset is large enough to benefit — pair it with a raised
``reflection_minibatch_size`` (e.g. 9-25).

Usage::

    from gepa.proposer.reflective_mutation.combee import ComBEEReflectionLM

    result = gepa.optimize(
        ...,
        reflection_strategy=ComBEEReflectionLM("openai/gpt-5"),
        reflection_minibatch_size=9,
    )

Cost: ``k + 1`` reflection-LM calls per component per proposal (vs 1 for the
default reflector). Per-call intermediates and call counts are recorded in
``ReflectionProposal.metadata`` under ``combee:``-namespaced keys, and GEPA
forwards them to ``on_proposal_end`` and experiment trackers. ``total_cost``/
token totals are exposed by delegation to the wrapped LM, so
``max_reflection_cost`` works.

Originally contributed by @nuglifeleoji in
https://github.com/gepa-ai/gepa/pull/307; re-hosted onto the ReflectionLM
protocol introduced in #369.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionJob, ReflectionProposal
from gepa.strategies.instruction_proposal import InstructionProposalSignature

DEFAULT_AGGREGATION_PROMPT_TEMPLATE = """I provided an assistant with the following current instruction:
```
<curr_param>
```

Multiple parallel processes each independently proposed an updated instruction based on a different subset of evaluation examples. Here are the proposed updates:
```
<side_info>
```

Your task is to synthesize these proposed instruction updates into a single, comprehensive instruction. Incorporate the key improvements and specific insights from all proposals. When proposals offer complementary guidance, include all relevant details. When they conflict, prefer more specific and task-relevant guidance.

Provide the final synthesized instruction within ``` blocks."""


# One logical reflection-LM completion.  The prompt fingerprint guards the
# retry cache against a caller reusing an occurrence id with different input.
_CompletionId = tuple[int, int, str, int]


@dataclass
class _CachedCompletion:
    prompt_fingerprint: str
    raw_output: str


@dataclass
class _ReflectionAttempt:
    """State kept only while retrying one ``reflect_many`` invocation.

    ``reflect_many`` plans calls in map/reduce wave order, while the engine's
    failure fallback calls ``reflect`` in job order.  Completion ids therefore
    name a logical call occurrence instead of using prompt text as identity.
    """

    num_jobs: int
    completions: dict[_CompletionId, _CachedCompletion] = field(default_factory=dict)
    next_fallback_job_idx: int = 0


class ComBEEReflectionLM:
    """Map/reduce reflection over large minibatches (ComBEE, arXiv:2604.04247).

    Implements the :class:`~gepa.proposer.reflective_mutation.reflection_lm.ReflectionLM`
    protocol; pass as ``reflection_strategy=`` to :func:`gepa.optimize` or via
    ``ReflectionConfig.reflection_strategy`` in ``optimize_anything``.

    Args:
        lm: The reflection language model — a model name string (resolved via
            :class:`gepa.lm.LM`) or any ``LanguageModel`` callable. Callables
            without cost tracking are wrapped in :class:`gepa.lm.TrackingLM`
            for token estimates, but cannot support ``max_reflection_cost``.
        lm_kwargs: Completion options used when ``lm`` is a model name. When
            omitted, GEPA's public ``reflection_lm_kwargs`` are applied at
            wiring time; explicit constructor kwargs take precedence.
        reflection_prompt_template: Level-1 (map) prompt template — a string
            applied to all components or a dict mapping component names to
            templates (components without an entry warn once and use the
            default), exactly like the default reflector.
        aggregation_prompt_template: Level-2 (reduce) template. Must contain
            ``<curr_param>`` and ``<side_info>`` placeholders. Defaults to
            :data:`DEFAULT_AGGREGATION_PROMPT_TEMPLATE`.
        duplication_factor: ``p`` — how many times each record is duplicated
            before shuffling (§3.2). Default 2.
        rng: RNG for the augmented shuffle. When omitted, GEPA binds the
            engine RNG so the single-proposal path reproduces #307 exactly.
            Pass an explicit RNG to keep ComBEE independent of engine sampling.
        logger: Optional logger with a ``log(str)`` method. When the strategy
            is passed to a GEPA front door, its configured logger is used when
            this argument is omitted.
        batch_reflection: When True (default), multi-proposal iterations batch
            all Level-1 calls into one wave and all Level-2 calls into a
            second when the underlying LM provides ``batch_complete``. Without
            that capability, it automatically preserves #307's strict
            sequential per-proposal order. Set False to enforce that order
            even for batch-capable, order-dependent callables. The strategy
            still implements ``reflect_many`` in sequential mode so retries
            remain transactional.
    """

    def __init__(
        self,
        lm: LanguageModel | str,
        reflection_prompt_template: str | dict[str, str] | None = None,
        aggregation_prompt_template: str | None = None,
        duplication_factor: int = 2,
        rng: random.Random | None = None,
        logger: Any | None = None,
        batch_reflection: bool = True,
        *,
        lm_kwargs: dict[str, Any] | None = None,
    ):
        self._model_name = lm if isinstance(lm, str) else None
        self._lm_kwargs_explicit = lm_kwargs is not None
        if isinstance(lm, str):
            from gepa.lm import LM

            self._cost_tracking_supported = True
            lm = LM(lm, **(lm_kwargs or {}))
        else:
            from gepa.lm import TrackingLM

            # TrackingLM estimates tokens and deliberately reports zero cost;
            # accepting it for a dollar cap would make the stopper inert.
            self._cost_tracking_supported = hasattr(lm, "total_cost") and not isinstance(lm, TrackingLM)
            if not hasattr(lm, "total_cost"):
                lm = TrackingLM(lm)
        self.lm = lm
        self._reflection_prompt_template_explicit = reflection_prompt_template is not None
        self.reflection_prompt_template = reflection_prompt_template
        # Only None means "use the default": an explicit (even empty) template
        # must go through placeholder validation and fail loudly if invalid.
        self.aggregation_prompt_template = (
            DEFAULT_AGGREGATION_PROMPT_TEMPLATE if aggregation_prompt_template is None else aggregation_prompt_template
        )
        if not isinstance(duplication_factor, int) or isinstance(duplication_factor, bool) or duplication_factor < 1:
            raise ValueError(f"duplication_factor must be an integer >= 1, got {duplication_factor!r}")
        self.duplication_factor = duplication_factor
        self._rng_explicit = rng is not None
        self._rng = rng if rng is not None else random.Random(0)
        self._batch_reflection = batch_reflection
        # This is intentionally absent outside a reflect_many attempt. Normal
        # reflect() calls must never turn into a cross-iteration response cache.
        self._attempt: _ReflectionAttempt | None = None
        self._active_job_idx: int | None = None
        self.logger = logger
        self._missing_template_warnings: set[str] = set()

        if isinstance(reflection_prompt_template, dict):
            for template in reflection_prompt_template.values():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)
        InstructionProposalSignature.validate_prompt_template(self.aggregation_prompt_template)

    # Cost/token delegation so MaxReflectionCostStopper works with this strategy.
    @property
    def total_cost(self) -> float:
        return getattr(self.lm, "total_cost", 0.0)

    @property
    def total_tokens_in(self) -> int:
        return getattr(self.lm, "total_tokens_in", 0)

    @property
    def total_tokens_out(self) -> int:
        return getattr(self.lm, "total_tokens_out", 0)

    def supports_cost_tracking(self) -> bool:
        """Whether ``total_cost`` represents real provider spend."""
        return self._cost_tracking_supported

    def bind_rng(self, rng: random.Random) -> None:
        """Bind GEPA's run RNG unless the user supplied one explicitly.

        Sharing the engine stream preserves the #307 call and shuffle sequence
        in the legacy single-proposal path. An explicit RNG remains the opt-in
        isolation mechanism for callers that prefer independent streams.
        """
        if not self._rng_explicit:
            self._rng = rng

    def bind_logger(self, logger: Any) -> None:
        """Use GEPA's configured logger unless the caller supplied one."""
        if self.logger is None:
            self.logger = logger

    def bind_lm_kwargs(self, lm_kwargs: dict[str, Any] | None) -> None:
        """Apply public completion options to a model-name strategy.

        The strategy is created before GEPA's front doors have finished
        resolving configuration. Recreate the lightweight LM wrapper here,
        before any reflection call, unless constructor kwargs were explicit.
        """
        if self._model_name is None or self._lm_kwargs_explicit or lm_kwargs is None:
            return
        from gepa.lm import LM

        self.lm = LM(self._model_name, **lm_kwargs)

    def bind_reflection_prompt_template(self, template: str | dict[str, str] | None) -> None:
        """Use GEPA's public template unless the strategy supplied one."""
        if self._reflection_prompt_template_explicit:
            return
        if isinstance(template, dict):
            for value in template.values():
                InstructionProposalSignature.validate_prompt_template(value)
        else:
            InstructionProposalSignature.validate_prompt_template(template)
        self.reflection_prompt_template = template
        self._missing_template_warnings.clear()

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

    @staticmethod
    def _prompt_fingerprint(prompt: Any) -> str:
        return prompt if isinstance(prompt, str) else repr(prompt)

    @staticmethod
    def _completion_id(job_idx: int | None, component_idx: int, phase: str, call_idx: int) -> _CompletionId | None:
        if job_idx is None:
            return None
        return job_idx, component_idx, phase, call_idx

    def _cached_completion(self, completion_id: _CompletionId | None, prompt: Any) -> str | None:
        if self._attempt is None or completion_id is None:
            return None
        cached = self._attempt.completions.get(completion_id)
        if cached is not None and cached.prompt_fingerprint == self._prompt_fingerprint(prompt):
            return cached.raw_output
        return None

    def _remember_completion(self, completion_id: _CompletionId | None, prompt: Any, raw_output: str) -> None:
        if self._attempt is not None and completion_id is not None:
            self._attempt.completions[completion_id] = _CachedCompletion(
                prompt_fingerprint=self._prompt_fingerprint(prompt), raw_output=raw_output
            )

    def _complete_one(self, prompt: Any, completion_id: _CompletionId | None = None) -> str:
        """Issue one completion, reusing only its exact failed-attempt slot."""
        cached = self._cached_completion(completion_id, prompt)
        if cached is not None:
            return cached
        raw = self.lm(prompt).strip()
        self._remember_completion(completion_id, prompt, raw)
        return raw

    def _call_signature(
        self,
        current_instruction: str,
        dataset_with_feedback: Any,
        prompt_template: str | None,
        completion_id: _CompletionId | None = None,
    ) -> tuple[str, str | list[dict[str, Any]], str]:
        # Equivalent to InstructionProposalSignature.run_with_metadata (render
        # -> complete -> strip -> extract), routed through the completion memo.
        prompt, _messages = self._render_prompt(current_instruction, dataset_with_feedback, prompt_template)
        raw_output = self._complete_one(prompt, completion_id)
        result = InstructionProposalSignature.output_extractor(raw_output)
        return result["new_instruction"], prompt, raw_output

    def _reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        job_idx: int | None,
    ) -> tuple[ReflectionProposal, ComBEEReflectionLM]:
        proposal = ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={})
        total_lm_calls = 0

        for component_idx, comp in enumerate(components_to_update):
            if comp not in reflective_dataset or not reflective_dataset.get(comp):
                self._log(f"Component '{comp}' is not in reflective dataset. Skipping.")
                continue
            if comp not in candidate:
                self._log(f"Component '{comp}' is missing from candidate. Skipping.")
                continue

            records = list(reflective_dataset[comp])
            n = len(records)
            # k = ⌊√n⌋ — ComBEE paper §3.1 default.
            k = max(1, int(n**0.5))
            level1_template = self._resolve_template(comp)

            if k <= 1:
                # Degenerate case (n < 4): standard single-call reflection.
                self._log(
                    f"ComBEE: component '{comp}' has n={n} reflection records (k=1); "
                    "falling back to standard single-call reflection. Raise "
                    "reflection_minibatch_size to >= 4 to activate ComBEE aggregation."
                )
                new_text, lm_prompt, raw_output = self._call_signature(
                    candidate[comp],
                    records,
                    level1_template,
                    self._completion_id(job_idx, component_idx, "fallback", 0),
                )
                proposal.new_texts[comp] = new_text
                proposal.prompts[comp] = lm_prompt
                proposal.raw_lm_outputs[comp] = raw_output
                proposal.metadata[f"combee:{comp}:num_lm_calls"] = 1
                proposal.metadata[f"combee:{comp}:mode"] = "fallback_single_call"
                total_lm_calls += 1
                continue

            # --- Augmented shuffle (§3.2): duplicate each record p times, shuffle.
            augmented: list[Mapping[str, Any]] = records * self.duplication_factor
            self._rng.shuffle(augmented)
            total_aug = len(augmented)

            # --- Level-1 (Map): one LM call per group.
            base_group_size = total_aug // k
            remainder = total_aug % k
            group_proposals: list[str] = []
            level1_prompts: list[str | list[dict[str, Any]]] = []
            level1_outputs: list[str] = []
            offset = 0
            for i in range(k):
                size = base_group_size + (1 if i < remainder else 0)
                group_records = augmented[offset : offset + size]
                offset += size
                # Defensive (mirrored from #307): unreachable by construction —
                # total_aug = p*n >= n >= k, so no group is ever empty; kept as
                # a guard for future changes that let map calls fail soft.
                if not group_records:
                    continue
                new_text, lm_prompt, raw_output = self._call_signature(
                    candidate[comp],
                    group_records,
                    level1_template,
                    self._completion_id(job_idx, component_idx, "map", i),
                )
                group_proposals.append(new_text)
                level1_prompts.append(lm_prompt)
                level1_outputs.append(raw_output)
                total_lm_calls += 1

            proposal.metadata[f"combee:{comp}:level1_prompts"] = level1_prompts
            proposal.metadata[f"combee:{comp}:level1_outputs"] = level1_outputs
            proposal.metadata[f"combee:{comp}:k"] = k

            if not group_proposals:
                self._log(f"ComBEE: component '{comp}' produced no group proposals. Skipping.")
                proposal.metadata[f"combee:{comp}:num_lm_calls"] = total_lm_calls
                proposal.metadata[f"combee:{comp}:mode"] = "no_group_proposals"
                continue

            if len(group_proposals) == 1:
                # Only one group produced a proposal — nothing to aggregate.
                self._log(
                    f"ComBEE: component '{comp}' has a single surviving group proposal; "
                    "skipping the aggregation step and using it directly."
                )
                proposal.new_texts[comp] = group_proposals[0]
                proposal.prompts[comp] = level1_prompts[0]
                proposal.raw_lm_outputs[comp] = level1_outputs[0]
                proposal.metadata[f"combee:{comp}:num_lm_calls"] = 1
                proposal.metadata[f"combee:{comp}:mode"] = "single_group"
                continue

            # --- Level-2 (Reduce): aggregate the k proposals into one instruction.
            # Each proposal is a single-field record so it renders cleanly in the
            # aggregation prompt's <side_info> block.
            agg_records: list[Mapping[str, Any]] = [{"Proposed Instruction Update": prop} for prop in group_proposals]
            final_text, agg_prompt, agg_output = self._call_signature(
                candidate[comp],
                agg_records,
                self.aggregation_prompt_template,
                self._completion_id(job_idx, component_idx, "reduce", 0),
            )
            total_lm_calls += 1
            proposal.new_texts[comp] = final_text
            proposal.prompts[comp] = agg_prompt
            proposal.raw_lm_outputs[comp] = agg_output
            proposal.metadata[f"combee:{comp}:num_lm_calls"] = len(group_proposals) + 1
            proposal.metadata[f"combee:{comp}:mode"] = "map_reduce"

        proposal.metadata["combee:total_lm_calls"] = total_lm_calls
        # Stateless across proposals: the RNG advances, but no reflection
        # context is carried, so this object is its own successor.
        return proposal, self

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ComBEEReflectionLM]:
        """Reflect one job.

        Outside a failed ``reflect_many`` attempt this is a plain, uncached
        call. During the engine's per-job recovery path it replays the matching
        logical call slots, so completed work is reused without treating equal
        prompts from different jobs as the same completion.
        """
        if self._active_job_idx is not None:
            return self._reflect(candidate, reflective_dataset, components_to_update, self._active_job_idx)
        if self._attempt is None:
            return self._reflect(candidate, reflective_dataset, components_to_update, None)

        attempt = self._attempt
        job_idx = attempt.next_fallback_job_idx
        self._active_job_idx = job_idx
        try:
            return self._reflect(candidate, reflective_dataset, components_to_update, job_idx)
        finally:
            self._active_job_idx = None
            # _propose_texts_batch_safe tries every job even when one fails, so
            # consume this slot on both success and failure.
            if self._attempt is attempt:
                attempt.next_fallback_job_idx += 1
                if attempt.next_fallback_job_idx >= attempt.num_jobs:
                    self._attempt = None

    # ------------------------------------------------------------------
    # Batched form (BatchReflectionLM): two waves across all jobs
    # ------------------------------------------------------------------

    def _render_prompt(
        self, current_instruction: str, dataset_with_feedback: Any, prompt_template: str | None
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]]]:
        prompt = InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": current_instruction,
                "dataset_with_feedback": dataset_with_feedback,
                "prompt_template": prompt_template,
            }
        )
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return prompt, messages

    def _batch_complete(
        self,
        prompts: list[Any],
        messages_list: list[list[dict[str, Any]]],
        completion_ids: list[_CompletionId],
    ) -> list[str]:
        """Issue one wave and reuse only exact completed retry slots."""
        if not prompts:
            return []
        if not (len(prompts) == len(messages_list) == len(completion_ids)):
            raise ValueError("prompts, messages_list, and completion_ids must have the same length")

        cached_outputs = [
            self._cached_completion(completion_id, prompt)
            for completion_id, prompt in zip(completion_ids, prompts, strict=True)
        ]
        miss_idx = [i for i, cached in enumerate(cached_outputs) if cached is None]
        if miss_idx:
            batch_complete = getattr(self.lm, "batch_complete", None)
            if callable(batch_complete) and len(miss_idx) > 1:
                fresh = list(cast(Any, batch_complete)([messages_list[i] for i in miss_idx]))
                for i, raw in zip(miss_idx, fresh, strict=True):
                    raw = raw.strip()
                    self._remember_completion(completion_ids[i], prompts[i], raw)
                    cached_outputs[i] = raw
            else:
                for i in miss_idx:
                    raw = self.lm(prompts[i]).strip()
                    self._remember_completion(completion_ids[i], prompts[i], raw)
                    cached_outputs[i] = raw
        if any(raw is None for raw in cached_outputs):
            raise RuntimeError("completion cache was not populated for every requested call")
        return [raw if raw is not None else "" for raw in cached_outputs]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ComBEEReflectionLM]]:
        """Reflect all jobs as one failure-atomic attempt.

        With ``batch_reflection=True``, map calls form one wave and reduce calls
        form another. With ``False``, each job's full map/reduce pass completes
        before the next one starts, matching #307 call order. Both forms retain
        a logical-call retry cache and restore the shuffle stream on failure.
        """
        if self._attempt is None or self._attempt.num_jobs != len(jobs):
            self._attempt = _ReflectionAttempt(num_jobs=len(jobs))
        rng_state = self._rng.getstate()
        try:
            batch_complete = getattr(self.lm, "batch_complete", None)
            results = (
                self._reflect_many_inner(jobs)
                if self._batch_reflection and callable(batch_complete)
                else self._reflect_many_sequential(jobs)
            )
        except Exception:
            self._rng.setstate(rng_state)
            assert self._attempt is not None
            self._attempt.next_fallback_job_idx = 0
            raise
        self._attempt = None
        return results

    def _reflect_many_sequential(
        self, jobs: list[ReflectionJob]
    ) -> list[tuple[ReflectionProposal, ComBEEReflectionLM]]:
        """Strict #307 ordering, wrapped in the same retry transaction."""
        results: list[tuple[ReflectionProposal, ComBEEReflectionLM]] = []
        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            self._active_job_idx = job_idx
            try:
                results.append(self._reflect(candidate, reflective_dataset, components_to_update, job_idx))
            finally:
                self._active_job_idx = None
        return results

    def _reflect_many_inner(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ComBEEReflectionLM]]:
        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={}) for _ in jobs]
        job_call_totals = [0] * len(jobs)

        # --- Plan (consumes RNG in the same order as sequential reflect) ---
        # One plan object per (job, component) OCCURRENCE, in plan order; wave
        # entries reference their plan directly, so duplicates never merge.
        plans: list[dict[str, Any]] = []
        wave1: list[tuple[dict[str, Any], Any, list[dict[str, Any]], _CompletionId]] = []

        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            for component_idx, comp in enumerate(components_to_update):
                if comp not in reflective_dataset or not reflective_dataset.get(comp):
                    self._log(f"Component '{comp}' is not in reflective dataset. Skipping.")
                    continue
                if comp not in candidate:
                    self._log(f"Component '{comp}' is missing from candidate. Skipping.")
                    continue

                records = list(reflective_dataset[comp])
                n = len(records)
                k = max(1, int(n**0.5))
                level1_template = self._resolve_template(comp)

                plan: dict[str, Any] = {
                    "job_idx": job_idx,
                    "component_idx": component_idx,
                    "comp": comp,
                    "k": k,
                    "level1_prompts": [],
                    "level1_outputs": [],
                    "group_proposals": [],
                    "candidate_text": candidate[comp],
                }
                plans.append(plan)

                if k <= 1:
                    self._log(
                        f"ComBEE: component '{comp}' has n={n} reflection records (k=1); "
                        "falling back to standard single-call reflection. Raise "
                        "reflection_minibatch_size to >= 4 to activate ComBEE aggregation."
                    )
                    plan["mode"] = "fallback_single_call"
                    prompt, messages = self._render_prompt(candidate[comp], records, level1_template)
                    completion_id = self._completion_id(job_idx, component_idx, "fallback", 0)
                    assert completion_id is not None
                    wave1.append((plan, prompt, messages, completion_id))
                    continue

                plan["mode"] = "map_reduce"
                augmented: list[Mapping[str, Any]] = records * self.duplication_factor
                self._rng.shuffle(augmented)
                total_aug = len(augmented)
                base_group_size = total_aug // k
                remainder = total_aug % k
                offset = 0
                for i in range(k):
                    size = base_group_size + (1 if i < remainder else 0)
                    group_records = augmented[offset : offset + size]
                    offset += size
                    # Defensive (mirrored from reflect()): unreachable by
                    # construction — total_aug = p*n >= n >= k.
                    if not group_records:
                        continue
                    prompt, messages = self._render_prompt(candidate[comp], group_records, level1_template)
                    completion_id = self._completion_id(job_idx, component_idx, "map", i)
                    assert completion_id is not None
                    wave1.append((plan, prompt, messages, completion_id))

        # --- Wave 1: all map + fallback calls, one batched round ---
        wave1_outputs = self._batch_complete([e[1] for e in wave1], [e[2] for e in wave1], [e[3] for e in wave1])
        for (plan, prompt, _messages, _completion_id), raw in zip(wave1, wave1_outputs, strict=True):
            raw = raw.strip()
            new_instruction = InstructionProposalSignature.output_extractor(raw)["new_instruction"]
            job_call_totals[plan["job_idx"]] += 1
            plan["group_proposals"].append(new_instruction)
            plan["level1_prompts"].append(prompt)
            plan["level1_outputs"].append(raw)

        # --- Between waves: plan reduces for map_reduce components ---
        wave2: list[tuple[dict[str, Any], Any, list[dict[str, Any]], _CompletionId]] = []
        for plan in plans:
            if plan["mode"] != "map_reduce" or len(plan["group_proposals"]) < 2:
                continue
            agg_records: list[Mapping[str, Any]] = [
                {"Proposed Instruction Update": prop} for prop in plan["group_proposals"]
            ]
            prompt, messages = self._render_prompt(
                plan["candidate_text"], agg_records, self.aggregation_prompt_template
            )
            completion_id = self._completion_id(plan["job_idx"], plan["component_idx"], "reduce", 0)
            assert completion_id is not None
            wave2.append((plan, prompt, messages, completion_id))

        # --- Wave 2: all reduce calls, one batched round ---
        wave2_outputs = self._batch_complete([e[1] for e in wave2], [e[2] for e in wave2], [e[3] for e in wave2])
        for (plan, prompt, _messages, _completion_id), raw in zip(wave2, wave2_outputs, strict=True):
            raw = raw.strip()
            plan["reduce_prompt"] = prompt
            plan["reduce_output"] = raw
            plan["final_text"] = InstructionProposalSignature.output_extractor(raw)["new_instruction"]
            job_call_totals[plan["job_idx"]] += 1

        # --- Finalize in plan order (duplicate components overwrite exactly
        # --- as sequential reflect() would) ---
        for plan in plans:
            proposal = proposals[plan["job_idx"]]
            comp = plan["comp"]
            if plan["mode"] == "fallback_single_call":
                proposal.new_texts[comp] = plan["group_proposals"][0]
                proposal.prompts[comp] = plan["level1_prompts"][0]
                proposal.raw_lm_outputs[comp] = plan["level1_outputs"][0]
                proposal.metadata[f"combee:{comp}:num_lm_calls"] = 1
                proposal.metadata[f"combee:{comp}:mode"] = "fallback_single_call"
                continue
            proposal.metadata[f"combee:{comp}:level1_prompts"] = plan["level1_prompts"]
            proposal.metadata[f"combee:{comp}:level1_outputs"] = plan["level1_outputs"]
            proposal.metadata[f"combee:{comp}:k"] = plan["k"]
            n_props = len(plan["group_proposals"])
            if n_props == 0:
                self._log(f"ComBEE: component '{comp}' produced no group proposals. Skipping.")
                proposal.metadata[f"combee:{comp}:num_lm_calls"] = 0
                proposal.metadata[f"combee:{comp}:mode"] = "no_group_proposals"
                continue
            if n_props == 1:
                self._log(
                    f"ComBEE: component '{comp}' has a single surviving group proposal; "
                    "skipping the aggregation step and using it directly."
                )
                proposal.new_texts[comp] = plan["group_proposals"][0]
                proposal.prompts[comp] = plan["level1_prompts"][0]
                proposal.raw_lm_outputs[comp] = plan["level1_outputs"][0]
                proposal.metadata[f"combee:{comp}:num_lm_calls"] = 1
                proposal.metadata[f"combee:{comp}:mode"] = "single_group"
                continue
            proposal.new_texts[comp] = plan["final_text"]
            proposal.prompts[comp] = plan["reduce_prompt"]
            proposal.raw_lm_outputs[comp] = plan["reduce_output"]
            proposal.metadata[f"combee:{comp}:num_lm_calls"] = n_props + 1
            proposal.metadata[f"combee:{comp}:mode"] = "map_reduce"

        for job_idx, proposal in enumerate(proposals):
            proposal.metadata["combee:total_lm_calls"] = job_call_totals[job_idx]
        return [(proposal, self) for proposal in proposals]
