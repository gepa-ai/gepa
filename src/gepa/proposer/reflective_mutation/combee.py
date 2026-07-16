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
``ReflectionProposal.metadata`` under ``combee:``-namespaced keys, so they
reach ``on_proposal_end`` consumers, experiment trackers, and proposal
metadata. ``total_cost``/token totals are exposed by delegation to the wrapped
LM, so ``max_reflection_cost`` works.

Originally contributed by @nuglifeleoji in
https://github.com/gepa-ai/gepa/pull/307; re-hosted onto the ReflectionLM
protocol introduced in #369.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

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


class ComBEEReflectionLM:
    """Map/reduce reflection over large minibatches (ComBEE, arXiv:2604.04247).

    Implements the :class:`~gepa.proposer.reflective_mutation.reflection_lm.ReflectionLM`
    protocol; pass as ``reflection_strategy=`` to :func:`gepa.optimize` or via
    ``ReflectionConfig.reflection_strategy`` in ``optimize_anything``.

    Args:
        lm: The reflection language model — a model name string (resolved via
            :class:`gepa.lm.LM`) or any ``LanguageModel`` callable. Callables
            without cost tracking are wrapped in :class:`gepa.lm.TrackingLM`
            so ``max_reflection_cost`` remains usable.
        reflection_prompt_template: Level-1 (map) prompt template — a string
            applied to all components or a dict mapping component names to
            templates (components without an entry warn once and use the
            default), exactly like the default reflector.
        aggregation_prompt_template: Level-2 (reduce) template. Must contain
            ``<curr_param>`` and ``<side_info>`` placeholders. Defaults to
            :data:`DEFAULT_AGGREGATION_PROMPT_TEMPLATE`.
        duplication_factor: ``p`` — how many times each record is duplicated
            before shuffling (§3.2). Default 2.
        rng: Seeded RNG for the augmented shuffle (default ``Random(0)``), so
            runs are reproducible.
        logger: Optional logger with a ``log(str)`` method.
        batch_reflection: When True (default), multi-proposal iterations batch
            all Level-1 calls into one wave and all Level-2 calls into a
            second (assumes exchangeable LM calls — replies depending only on
            their own prompt, the standard property of stateless completion
            APIs). Set False to enforce strict sequential per-proposal
            ordering (#307-identical), e.g. for order-dependent callables.
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
    ):
        if isinstance(lm, str):
            from gepa.lm import LM

            lm = LM(lm)
        elif not hasattr(lm, "total_cost"):
            from gepa.lm import TrackingLM

            lm = TrackingLM(lm)
        self.lm = lm
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
        # Completion memo for failure transparency: populated while a
        # reflect_many attempt runs, consulted by every LM call (including the
        # engine's per-task retry path through reflect()), cleared at the
        # start and successful end of each reflect_many. A failed wave never
        # forces already-paid completions to be re-purchased.
        self._completion_memo: dict[str, str] = {}
        if not batch_reflection:
            # Opting out makes the sequential-ordering contract ENFORCED, not
            # documented: the engine's dispatcher sees reflect_many as None and
            # runs one reflect() per proposal in task order (matching #307's
            # exact call ordering). Use when the underlying callable is
            # order-dependent (e.g. session-accumulating wrappers).
            self.reflect_many = None  # type: ignore[assignment]
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

    def bind_rng(self, rng: random.Random) -> None:
        """GEPA's wiring calls this with a stream derived from
        ``gepa.optimize(seed=...)`` when the user did not pass an explicit
        ``rng`` — making shuffles seed-sensitive by default while staying
        independent of the candidate-selector stream (unlike #307's shared
        RNG, where shuffles and selection perturbed each other). An explicit
        user-provided ``rng`` always wins."""
        if not self._rng_explicit:
            self._rng = rng

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

    def _memo_key(self, prompt: Any) -> str:
        return prompt if isinstance(prompt, str) else str(prompt)

    def _complete_one(self, prompt: Any) -> str:
        """One completion, via the failure-transparency memo (see ctor)."""
        key = self._memo_key(prompt)
        cached = self._completion_memo.get(key)
        if cached is not None:
            return cached
        raw = self.lm(prompt).strip()
        self._completion_memo[key] = raw
        return raw

    def _call_signature(
        self, current_instruction: str, dataset_with_feedback: Any, prompt_template: str | None
    ) -> tuple[str, str | list[dict[str, Any]], str]:
        # Equivalent to InstructionProposalSignature.run_with_metadata (render
        # -> complete -> strip -> extract), routed through the completion memo.
        prompt, _messages = self._render_prompt(current_instruction, dataset_with_feedback, prompt_template)
        raw_output = self._complete_one(prompt)
        result = InstructionProposalSignature.output_extractor(raw_output)
        return result["new_instruction"], prompt, raw_output

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ComBEEReflectionLM]:
        proposal = ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={})
        total_lm_calls = 0

        for comp in components_to_update:
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
                new_text, lm_prompt, raw_output = self._call_signature(candidate[comp], records, level1_template)
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
                    candidate[comp], group_records, level1_template
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
            agg_records: list[Mapping[str, Any]] = [
                {"Proposed Instruction Update": prop} for prop in group_proposals
            ]
            final_text, agg_prompt, agg_output = self._call_signature(
                candidate[comp], agg_records, self.aggregation_prompt_template
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

    def _batch_complete(self, prompts: list[Any], messages_list: list[list[dict[str, Any]]]) -> list[str]:
        """Issue completions, batched when the LM supports it, through the
        failure-transparency memo: already-paid completions (from a prior
        failed attempt of the same iteration) are reused, and only misses are
        sent. The sequential fallback memoizes incrementally, so a mid-wave
        failure preserves every completed call for the retry."""
        if not prompts:
            return []
        keys = [self._memo_key(p) for p in prompts]
        miss_idx = [i for i, k in enumerate(keys) if k not in self._completion_memo]
        if miss_idx:
            batch_complete = getattr(self.lm, "batch_complete", None)
            if batch_complete is not None and len(miss_idx) > 1:
                fresh = list(batch_complete([messages_list[i] for i in miss_idx]))
                for i, raw in zip(miss_idx, fresh, strict=True):
                    self._completion_memo[keys[i]] = raw.strip()
            else:
                for i in miss_idx:
                    self._completion_memo[keys[i]] = self.lm(prompts[i]).strip()
        return [self._completion_memo[k] for k in keys]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ComBEEReflectionLM]]:
        """Vectorized :meth:`reflect`: all jobs' ComBEE passes in two batched waves.

        Wave 1 batches every Level-1 (map) call and every degenerate-fallback
        call across all jobs/components into one completion round; Wave 2
        batches every Level-2 (reduce) call into a second round. Prompts and
        the RNG stream are identical to sequential :meth:`reflect` calls in job
        order (shuffles happen at plan time, in job/component order), and
        proposals are finalized in plan order (so repeated component names
        overwrite exactly as they would sequentially); only the global call
        ORDER differs. Contract: the underlying LM's calls are assumed
        exchangeable — each reply depends only on its own prompt, the standard
        property of stateless completion APIs. Order-dependent callables
        (e.g. session-accumulating wrappers) are outside ComBEE's contract;
        it declares itself stateless by returning ``self`` as successor. Cost per
        component is unchanged: k map calls + 1 reduce.
        """
        # Failure transparency, two halves: (1) snapshot/restore the RNG so a
        # failed round's retry replays the same shuffle stream — bit-identical
        # results; (2) the completion memo (fresh per iteration, kept across a
        # failure) so the retry reuses every already-paid completion — no
        # double spend. Cleared again on success to bound memory.
        self._completion_memo.clear()
        rng_state = self._rng.getstate()
        try:
            results = self._reflect_many_inner(jobs)
        except Exception:
            self._rng.setstate(rng_state)
            raise
        self._completion_memo.clear()
        return results

    def _reflect_many_inner(
        self, jobs: list[ReflectionJob]
    ) -> list[tuple[ReflectionProposal, ComBEEReflectionLM]]:
        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={}) for _ in jobs]
        job_call_totals = [0] * len(jobs)

        # --- Plan (consumes RNG in the same order as sequential reflect) ---
        # One plan object per (job, component) OCCURRENCE, in plan order; wave
        # entries reference their plan directly, so duplicates never merge.
        plans: list[dict[str, Any]] = []
        wave1: list[tuple[dict[str, Any], Any, list[dict[str, Any]]]] = []

        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            for comp in components_to_update:
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
                    wave1.append((plan, prompt, messages))
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
                    wave1.append((plan, prompt, messages))

        # --- Wave 1: all map + fallback calls, one batched round ---
        wave1_outputs = self._batch_complete([e[1] for e in wave1], [e[2] for e in wave1])
        for (plan, prompt, _messages), raw in zip(wave1, wave1_outputs, strict=True):
            raw = raw.strip()
            new_instruction = InstructionProposalSignature.output_extractor(raw)["new_instruction"]
            job_call_totals[plan["job_idx"]] += 1
            plan["group_proposals"].append(new_instruction)
            plan["level1_prompts"].append(prompt)
            plan["level1_outputs"].append(raw)

        # --- Between waves: plan reduces for map_reduce components ---
        wave2: list[tuple[dict[str, Any], Any, list[dict[str, Any]]]] = []
        for plan in plans:
            if plan["mode"] != "map_reduce" or len(plan["group_proposals"]) < 2:
                continue
            agg_records: list[Mapping[str, Any]] = [
                {"Proposed Instruction Update": prop} for prop in plan["group_proposals"]
            ]
            prompt, messages = self._render_prompt(
                plan["candidate_text"], agg_records, self.aggregation_prompt_template
            )
            wave2.append((plan, prompt, messages))

        # --- Wave 2: all reduce calls, one batched round ---
        wave2_outputs = self._batch_complete([e[1] for e in wave2], [e[2] for e in wave2])
        for (plan, prompt, _messages), raw in zip(wave2, wave2_outputs, strict=True):
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
