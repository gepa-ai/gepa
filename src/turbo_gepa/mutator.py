"""Mutation utilities for LLM-driven candidate generation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Sequence, Set

from .interfaces import Candidate

BatchReflectionRunner = Callable[[Sequence[Dict[str, object]], int], Awaitable[Sequence[str]]]
SpecInductionRunner = Callable[[Sequence[Dict[str, object]], int], Awaitable[Sequence[str]]]
Validator = Callable[[Candidate], None]


def _default_token_validator(max_tokens: int) -> Validator:
    def _validate(candidate: Candidate) -> None:
        if len(candidate.text.split()) > max_tokens:
            raise ValueError("candidate exceeds token budget")

    return _validate


@dataclass
class MutationConfig:
    reflection_batch_size: int
    max_mutations: int
    max_tokens: int


class Mutator:
    """Generate candidate mutations via LLM reflection, spec induction, and temperature exploration."""

    def __init__(
        self,
        config: MutationConfig,
        *,
        validators: Iterable[Validator] | None = None,
        reflection_runner: Callable[[Sequence[Dict[str, Any]], str, Dict[str, Any] | None], Awaitable[Sequence[str]]] | None = None,
        batch_reflection_runner: BatchReflectionRunner | None = None,
        spec_induction_runner: SpecInductionRunner | None = None,
        seed: int | None = None,
        temperature_mutations_enabled: bool = True,
    ) -> None:
        self.config = config
        extra_validators = list(validators or [])
        extra_validators.append(_default_token_validator(config.max_tokens))
        self.validators = extra_validators
        self.single_reflection_runner = reflection_runner
        self.batch_reflection_runner = batch_reflection_runner
        self.spec_induction_runner = spec_induction_runner
        self.temperature_mutations_enabled = temperature_mutations_enabled

    async def propose(
        self,
        parent_contexts: List[Dict[str, object]],
        num_mutations: int,
        task_examples: List[Dict[str, object]] | None = None,
    ) -> List[Candidate]:
        """
        Generate mutated candidates given parent contexts and optional task examples.

        The method blends several strategies in priority order:
            1. Deterministic temperature exploration (cheap, immediate signal)
            2. Batched reflection (preferred) or single-candidate reflection
            3. Specification induction from task examples (if available)
        """
        if not parent_contexts:
            return []

        budget = max(0, num_mutations)
        if self.config.max_mutations:
            budget = min(budget, self.config.max_mutations)
        if budget == 0:
            return []

        proposals: List[Candidate] = []

        # 1) Deterministic temperature exploration
        temp_mutations = self._temperature_mutations(parent_contexts, budget)
        proposals.extend(temp_mutations)
        budget -= len(temp_mutations)
        if budget <= 0:
            return self._filter(proposals)

        # 2) Reflection-driven edits (batched preferred, single-runner fallback)
        reflection_mutations: List[Candidate] = []
        if self.batch_reflection_runner:
            reflection_mutations = await self._generate_incremental_mutations(parent_contexts, budget)
        elif self.single_reflection_runner:
            reflection_mutations = await self._generate_single_reflection_mutations(parent_contexts, budget)

        proposals.extend(reflection_mutations)
        budget -= len(reflection_mutations)
        if budget <= 0:
            return self._filter(proposals)

        # 3) Spec induction (fresh instructions)
        if self.spec_induction_runner and task_examples:
            spec_mutations = await self._generate_spec_induction_mutations(
                task_examples,
                budget,
                parent_contexts,
            )
            proposals.extend(spec_mutations)
            budget -= len(spec_mutations)
            if budget <= 0:
                return self._filter(proposals)

        return self._filter(proposals)[:num_mutations]

    async def _generate_incremental_mutations(
        self,
        parent_contexts: List[Dict[str, object]],
        num_mutations: int,
    ) -> List[Candidate]:
        """Generate mutations by synthesizing ideas from successful parent prompts."""
        # Build contexts for batch reflection
        reflection_contexts = []
        for ctx in parent_contexts:
            candidate = ctx["candidate"]
            failures = ctx.get("failures", []) or []

            # Collect failure traces
            traces = []
            for _example_id, trace_list in failures:
                traces.extend(trace_list)

            # Limit traces per parent to avoid token explosion
            traces = traces[: self.config.reflection_batch_size]

            reflection_contexts.append({
                "prompt": candidate.text,
                "traces": traces,
                "meta": dict(candidate.meta),
            })

        # Single batched reflection call
        mutated_texts = await self.batch_reflection_runner(reflection_contexts, num_mutations)

        # Convert to Candidates with metadata tracking generation method
        proposals: List[Candidate] = []
        for idx, text in enumerate(mutated_texts):
            # Use the best parent's metadata as base
            parent_candidate = self._best_parent_candidate(parent_contexts)

            meta = dict(parent_candidate.meta)
            meta.update({
                "edit": "incremental_reflection",  # Track generation method
                "generation_method": "incremental_reflection",  # Explicit tracking for analysis
                "parent": parent_candidate.fingerprint,
                "proposal_idx": idx,
                "num_parents_seen": len(parent_contexts),
            })
            proposals.append(Candidate(text=text, meta=meta))

        return proposals

    async def _generate_spec_induction_mutations(
        self,
        task_examples: List[Dict[str, object]],
        num_mutations: int,
        parent_contexts: List[Dict[str, object]],
    ) -> List[Candidate]:
        """Generate fresh specifications from task I/O examples (PROMPT-MII style)."""
        # Call spec induction runner
        spec_texts = await self.spec_induction_runner(task_examples, num_mutations)

        # Convert to Candidates with metadata tracking generation method
        proposals: List[Candidate] = []
        parent_candidate = self._best_parent_candidate(parent_contexts)

        for idx, text in enumerate(spec_texts):
            meta = dict(parent_candidate.meta)
            # Remove parent-specific fields since this is a fresh spec
            meta.pop("parent", None)
            meta.pop("parent_objectives", None)

            meta.update({
                "edit": "spec_induction",  # Track generation method
                "generation_method": "spec_induction",  # Explicit tracking for analysis
                "proposal_idx": idx,
                "num_examples_seen": len(task_examples),
            })
            proposals.append(Candidate(text=text, meta=meta))

        return proposals

    def _filter(self, candidates: Iterable[Candidate]) -> List[Candidate]:
        seen: Set[str] = set()
        valid: List[Candidate] = []
        for candidate in candidates:
            fingerprint = candidate.fingerprint
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                for validator in self.validators:
                    validator(candidate)
            except ValueError:
                continue
            valid.append(candidate)
        return valid

    def _best_parent_candidate(self, parent_contexts: Sequence[Dict[str, object]]) -> Candidate:
        if not parent_contexts:
            raise ValueError("parent_contexts must not be empty")

        def score(ctx: Dict[str, object]) -> float:
            candidate = ctx["candidate"]
            meta = candidate.meta or {}
            if isinstance(meta.get("quality"), (int, float)):
                return float(meta["quality"])
            parent_objectives = meta.get("parent_objectives")
            if isinstance(parent_objectives, dict):
                quality = parent_objectives.get("quality")
                if isinstance(quality, (int, float)):
                    return float(quality)
            return 0.0

        return max(parent_contexts, key=score)["candidate"]

    def _temperature_mutations(self, parent_contexts: Sequence[Dict[str, object]], limit: int) -> List[Candidate]:
        if not self.temperature_mutations_enabled or limit <= 0:
            return []
        anchors = [0.0, 0.3, 0.5, 0.7, 1.0]
        mutations: List[Candidate] = []
        for ctx in parent_contexts:
            if len(mutations) >= limit:
                break
            candidate = ctx["candidate"]
            current_temp = candidate.meta.get("temperature")
            temps_to_try: List[float] = []
            if current_temp is None:
                # Seed variants across anchors when no temperature set yet
                temps_to_try.extend(anchors)
            else:
                baseline_temp = float(current_temp)
                for anchor in anchors:
                    if abs(anchor - baseline_temp) > 0.15:
                        temps_to_try.append(anchor)
                temps_to_try.extend([
                    max(0.0, min(1.0, baseline_temp - 0.2)),
                    max(0.0, min(1.0, baseline_temp + 0.2)),
                ])

            seen: Set[float] = set()
            for temp in temps_to_try:
                if len(mutations) >= limit:
                    break
                temp = round(temp, 2)
                if temp in seen:
                    continue
                seen.add(temp)
                meta = dict(candidate.meta)
                meta.update({
                    "temperature": temp,
                    "edit": "temperature_shift",
                    "generation_method": "temperature_shift",
                    "parent": candidate.fingerprint,
                })
                mutations.append(Candidate(text=candidate.text, meta=meta))
        return mutations[:limit]

    async def _generate_single_reflection_mutations(
        self,
        parent_contexts: Sequence[Dict[str, object]],
        limit: int,
    ) -> List[Candidate]:
        if not self.single_reflection_runner or limit <= 0:
            return []

        proposals: List[Candidate] = []
        for ctx in parent_contexts:
            if len(proposals) >= limit:
                break
            candidate = ctx["candidate"]
            failures = ctx.get("failures", []) or []
            traces: List[Dict[str, object]] = []
            for _example_id, trace_list in failures:
                traces.extend(trace_list)
            traces = traces[: self.config.reflection_batch_size]

            try:
                mutated_texts = await self.single_reflection_runner(traces, candidate.text, candidate.meta)
            except TypeError:
                # Support legacy reflection runners that accept only (traces, text)
                mutated_texts = await self.single_reflection_runner(traces, candidate.text, None)  # type: ignore[arg-type]

            if not mutated_texts:
                continue

            for idx, text in enumerate(mutated_texts):
                meta = dict(candidate.meta)
                meta.update({
                    "edit": "reflection",
                    "generation_method": "reflection",
                    "parent": candidate.fingerprint,
                    "proposal_idx": idx,
                })
                proposals.append(Candidate(text=text, meta=meta))
                if len(proposals) >= limit:
                    break
        return proposals
