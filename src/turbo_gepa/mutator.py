"""Mutation utilities for LLM-driven candidate generation."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Sequence

from turbo_gepa.logging.logger import LoggerProtocol, StdOutLogger

from .interfaces import Candidate

BatchReflectionRunner = Callable[[Sequence[dict[str, object]], int], Awaitable[Sequence[str]]]
SpecInductionRunner = Callable[[Sequence[dict[str, object]], int], Awaitable[Sequence[str]]]
Validator = Callable[[Candidate], None]


def _default_token_validator(max_tokens: int) -> Validator:
    def _validate(candidate: Candidate) -> None:
        if len(candidate.text.split()) > max_tokens:
            raise ValueError("candidate exceeds token budget")

    return _validate


def _describe_callable(func: Any) -> str:
    if func is None:
        return "none"
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    module = getattr(func, "__module__", None)
    if name:
        return f"{module}.{name}" if module else name
    return repr(func)


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
        reflection_runner: Callable[[Sequence[dict[str, Any]], str, dict[str, Any] | None], Awaitable[Sequence[str]]]
        | None = None,
        batch_reflection_runner: BatchReflectionRunner | None = None,
        spec_induction_runner: SpecInductionRunner | None = None,
        seed: int | None = None,
        temperature_mutations_enabled: bool = True,
        logger: LoggerProtocol | None = None,
    ) -> None:
        self.config = config
        extra_validators = list(validators or [])
        extra_validators.append(_default_token_validator(config.max_tokens))
        self.validators = extra_validators
        self.single_reflection_runner = reflection_runner
        self.batch_reflection_runner = batch_reflection_runner
        self.spec_induction_runner = spec_induction_runner
        self.temperature_mutations_enabled = temperature_mutations_enabled
        self._reflection_examples: list[dict[str, object]] = []
        self._operator_stats: dict[str, dict[str, int]] = {
            "temperature_shift": {"wins": 0, "trials": 0},
            "incremental_reflection": {"wins": 0, "trials": 0},
            "reflection": {"wins": 0, "trials": 0},
            "spec_induction": {"wins": 0, "trials": 0},
        }
        self._operator_history: dict[str, deque[int]] = {key: deque(maxlen=20) for key in self._operator_stats}
        self._spec_cache: dict[str, deque[str]] = {}
        self._spec_cache_limit = max(4, config.max_mutations or 8)
        self._spec_runner_signature = _describe_callable(spec_induction_runner)
        self.logger: LoggerProtocol = logger or StdOutLogger()

    def set_reflection_examples(self, examples: list[dict[str, object]]) -> None:
        self._reflection_examples = examples

    def set_temperature_mutations_enabled(self, enabled: bool) -> None:
        """Toggle temperature exploration without rebuilding the mutator."""
        self.temperature_mutations_enabled = enabled

    def report_outcome(self, generation_method: str, success: bool) -> None:
        stats = self._operator_stats.setdefault(generation_method, {"wins": 0, "trials": 0})
        stats["trials"] += 1
        if success:
            stats["wins"] += 1
        history = self._operator_history.setdefault(generation_method, deque(maxlen=20))
        history.append(1 if success else 0)

    def _operator_weight(self, generation_method: str) -> float:
        history = self._operator_history.get(generation_method)
        if history and len(history) > 0:
            successes = sum(history)
            trials = len(history)
            # Simple moving-average with Laplace smoothing
            return (successes + 1.0) / (trials + 2.0)
        stats = self._operator_stats.get(generation_method)
        if stats and stats["trials"] > 0:
            return (stats["wins"] + 1.0) / (stats["trials"] + 2.0)
        return 1.0

    async def propose(
        self,
        parent_contexts: list[dict[str, object]],
        num_mutations: int,
        task_examples: list[dict[str, object]] | None = None,
    ) -> list[Candidate]:
        """
        Generate mutated candidates given parent contexts and optional task examples.

        The method blends several strategies in priority order:
            1. Deterministic temperature exploration (cheap, immediate signal)
            2. Batched reflection (preferred) or single-candidate reflection
            3. Specification induction from task examples (if available)
        """
        import asyncio
        import time

        propose_start = time.time()

        if not parent_contexts:
            return []

        total_budget = max(0, num_mutations)
        if self.config.max_mutations:
            total_budget = min(total_budget, self.config.max_mutations)
        if total_budget == 0:
            return []

        reflection_weight = max(
            self._operator_weight("incremental_reflection"),
            self._operator_weight("reflection"),
        )
        spec_weight = self._operator_weight("spec_induction") if (self.spec_induction_runner and task_examples) else 0.0
        temp_weight = self._operator_weight("temperature_shift") if self.temperature_mutations_enabled else 0.0

        spec_quota = 0
        if spec_weight > 0.0:
            total_rs = reflection_weight + spec_weight
            spec_share = spec_weight / total_rs if total_rs > 0 else 0.0
            spec_quota = min(total_budget, max(1, round(total_budget * spec_share)))

        non_spec_budget = total_budget - spec_quota
        proposals: list[Candidate] = []

        # 1) Deterministic temperature exploration
        temp_start = time.time()
        temp_quota = 0
        if non_spec_budget > 0 and self.temperature_mutations_enabled:
            total_tr = temp_weight + reflection_weight
            temp_share = temp_weight / total_tr if total_tr > 0 else 0.0
            temp_quota = min(non_spec_budget, round(non_spec_budget * temp_share))
        temp_mutations = self._temperature_mutations(parent_contexts, temp_quota)
        temp_time = time.time() - temp_start
        proposals.extend(temp_mutations)
        non_spec_budget = max(0, non_spec_budget - len(temp_mutations))

        # 2 & 3) Run reflection and spec induction CONCURRENTLY for speed
        llm_start = time.time()
        reflection_mutations: list[Candidate] = []
        spec_mutations: list[Candidate] = []

        async def run_reflection() -> list[Candidate]:
            if non_spec_budget <= 0:
                return []
            if self.batch_reflection_runner:
                return await self._generate_incremental_mutations(parent_contexts, non_spec_budget)
            if self.single_reflection_runner:
                return await self._generate_single_reflection_mutations(parent_contexts, non_spec_budget)
            return []

        async def run_spec() -> list[Candidate]:
            if not (self.spec_induction_runner and task_examples):
                return []
            spec_budget = min(spec_quota, total_budget - len(proposals))
            if spec_budget <= 0:
                return []
            return await self._generate_spec_induction_mutations(
                task_examples,
                spec_budget,
                parent_contexts,
            )

        reflection_task = asyncio.create_task(run_reflection())
        spec_task = asyncio.create_task(run_spec())

        reflection_mutations, spec_mutations = await asyncio.gather(reflection_task, spec_task)
        proposals.extend(reflection_mutations)
        proposals.extend(spec_mutations)

        llm_time = time.time() - llm_start

        filter_start = time.time()
        filtered = self._filter(proposals)[:num_mutations]
        filter_time = time.time() - filter_start

        propose_total = time.time() - propose_start

        # Log timing breakdown
        self.logger.log("⏱️  Mutator timing breakdown:")
        self.logger.log(f"   Temperature mutations: {temp_time:.2f}s ({len(temp_mutations)} generated)")
        self.logger.log(f"   LLM calls (parallel): {llm_time:.2f}s")
        self.logger.log(f"     - Reflection: {len(reflection_mutations)} mutations")
        self.logger.log(f"     - Spec induction: {len(spec_mutations)} mutations")
        self.logger.log(f"   Filtering: {filter_time:.2f}s")
        self.logger.log(f"   Total propose: {propose_total:.2f}s")

        return filtered

    async def _generate_incremental_mutations(
        self,
        parent_contexts: list[dict[str, object]],
        num_mutations: int,
    ) -> list[Candidate]:
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

            reflection_contexts.append(
                {
                    "prompt": candidate.text,
                    "traces": traces,
                    "meta": dict(candidate.meta),
                }
            )

        mutated_texts = await self._collect_text_batches(
            lambda: self.batch_reflection_runner(reflection_contexts, 1),
            num_mutations,
            max(1, min(self.config.reflection_batch_size, num_mutations)),
        )

        # Convert to Candidates with metadata tracking generation method
        proposals: list[Candidate] = []
        for idx, text in enumerate(mutated_texts):
            # Use the best parent's metadata as base
            parent_candidate = self._best_parent_candidate(parent_contexts)

            meta = dict(parent_candidate.meta)
            meta.update(
                {
                    "edit": "incremental_reflection",  # Track generation method
                    "generation_method": "incremental_reflection",  # Explicit tracking for analysis
                    "parent": parent_candidate.fingerprint,
                    "proposal_idx": idx,
                    "num_parents_seen": len(parent_contexts),
                }
            )
            proposals.append(Candidate(text=text, meta=meta))

        return proposals

    async def _generate_spec_induction_mutations(
        self,
        task_examples: list[dict[str, object]],
        num_mutations: int,
        parent_contexts: list[dict[str, object]],
    ) -> list[Candidate]:
        """Generate fresh specifications from task I/O examples (PROMPT-MII style)."""
        if num_mutations <= 0 or not self.spec_induction_runner:
            return []

        # Build reflection contexts similar to incremental mutations
        # but the spec_induction_runner will use a different prompt template
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

            reflection_contexts.append(
                {
                    "prompt": candidate.text,
                    "traces": traces,
                    "meta": dict(candidate.meta),
                }
            )

        # Cache key based on contexts + examples
        serialized_examples = sorted(json.dumps(ex, sort_keys=True) for ex in task_examples)
        parent_fingerprints = sorted(ctx["candidate"].fingerprint for ctx in parent_contexts) if parent_contexts else []
        cache_payload = {
            "runner": self._spec_runner_signature,
            "examples": serialized_examples,
            "parents": parent_fingerprints,
        }
        cache_key = hashlib.blake2s(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
        cache_queue = self._spec_cache.setdefault(cache_key, deque())

        spec_texts: list[str] = []
        while cache_queue and len(spec_texts) < num_mutations:
            spec_texts.append(cache_queue.popleft())

        remaining = num_mutations - len(spec_texts)
        if remaining > 0:
            prefetch = min(self._spec_cache_limit, max(remaining, remaining * 2))
            # Pass reflection_contexts (with prompts + traces) instead of just task_examples
            new_specs = await self._collect_text_batches(
                lambda: self.spec_induction_runner(reflection_contexts, 1),
                prefetch,
                max(1, min(4, prefetch)),
            )
            for text in new_specs:
                cache_queue.append(text)
            while len(cache_queue) > self._spec_cache_limit:
                cache_queue.popleft()
            while cache_queue and len(spec_texts) < num_mutations:
                spec_texts.append(cache_queue.popleft())

        # Convert to Candidates with metadata tracking generation method
        proposals: list[Candidate] = []
        parent_candidate = self._best_parent_candidate(parent_contexts)

        for idx, text in enumerate(spec_texts):
            meta = dict(parent_candidate.meta)
            meta["parent"] = parent_candidate.fingerprint
            meta.update(
                {
                    "edit": "spec_induction",  # Track generation method
                    "generation_method": "spec_induction",  # Explicit tracking for analysis
                    "proposal_idx": idx,
                    "num_examples_seen": len(task_examples),
                }
            )
            proposals.append(Candidate(text=text, meta=meta))

        return proposals

    async def _collect_text_batches(
        self,
        factory: Callable[[], Awaitable[Sequence[str]]],
        total: int,
        max_concurrency: int,
        early_stop_fraction: float = 0.85,  # Return after 85% of mutations complete
    ) -> list[str]:
        import asyncio
        import time

        if total <= 0:
            return []

        max_concurrency = max(1, min(max_concurrency, total))
        self.logger.log(
            f"🌀 Mutation batch starting: total={total}, max_concurrency={max_concurrency}, early_stop={early_stop_fraction}"
        )
        pending: dict[asyncio.Task[Sequence[str]], float] = {}  # task -> start_time
        results: list[str] = []
        started = 0
        early_stop_target = int(total * early_stop_fraction)
        batch_start_time = time.time()
        mutation_durations: list[float] = []  # Track how long each individual mutation took

        while len(results) < total:
            while started < total and len(pending) < max_concurrency:
                task = asyncio.create_task(factory())
                self.logger.log(f"   + Launched mutation task {started + 1}/{total}")
                pending[task] = time.time()  # Record when this task started
                started += 1

            if not pending:
                break

            # Check if we've hit early stop threshold
            if len(results) >= early_stop_target and early_stop_fraction < 1.0 and len(mutation_durations) >= 3:
                elapsed = time.time() - batch_start_time
                remaining = len(pending)

                # Compute average time per mutation based on completed ones
                avg_duration = sum(mutation_durations) / len(mutation_durations)

                # If we've been waiting significantly longer than expected, cut off stragglers
                # We expect remaining mutations to take avg_duration each
                # Add 2x buffer since stragglers can be slow
                expected_time_for_remaining = avg_duration * 2.0

                # How long have we been waiting since hitting the early stop target?
                # We estimate when we should have hit the target based on avg duration
                # Since tasks run concurrently, the batch should take roughly:
                # (total_mutations / max_concurrency) * avg_duration
                expected_time_to_target = (early_stop_target / max_concurrency) * avg_duration
                time_since_should_have_hit_target = elapsed - expected_time_to_target

                # Early stop if we've been waiting too long for stragglers
                if time_since_should_have_hit_target > expected_time_for_remaining and remaining >= 2:
                    self.logger.log(
                        f"   ⚠️ Mutation early stop: produced={len(results)}/{total} ({len(results) / total * 100:.0f}%), "
                        f"remaining={remaining}, avg_duration={avg_duration:.1f}s, "
                        f"waited={time_since_should_have_hit_target:.1f}s"
                    )
                    # Cancel remaining tasks - they're stragglers
                    for task in pending:
                        task.cancel()
                    break

            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task_start_time = pending.pop(task)
                task_duration = time.time() - task_start_time
                try:
                    batch = task.result()
                    if batch:
                        results.append(batch[0])
                        mutation_durations.append(task_duration)  # Track individual duration
                        self.logger.log(
                            f"   ✅ Mutation task complete in {task_duration:.2f}s "
                            f"(generated {len(batch)} candidates, total so far {len(results)})"
                        )
                except asyncio.CancelledError:
                    pass  # Expected for cancelled tasks

        self.logger.log(f"✅ Mutation batch finished: generated={len(results)} (requested {total})")
        return results[:total]

    def _filter(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        seen: set[str] = set()
        valid: list[Candidate] = []
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

    def _best_parent_candidate(self, parent_contexts: Sequence[dict[str, object]]) -> Candidate:
        if not parent_contexts:
            raise ValueError("parent_contexts must not be empty")

        def score(ctx: dict[str, object]) -> float:
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

    def _temperature_mutations(self, parent_contexts: Sequence[dict[str, object]], limit: int) -> list[Candidate]:
        if not self.temperature_mutations_enabled or limit <= 0:
            return []
        anchors = [0.0, 0.3, 0.5, 0.7, 1.0]
        mutations: list[Candidate] = []
        for ctx in parent_contexts:
            if len(mutations) >= limit:
                break
            candidate = ctx["candidate"]
            current_temp = candidate.meta.get("temperature")
            temps_to_try: list[float] = []
            if current_temp is None:
                # Seed variants across anchors when no temperature set yet
                temps_to_try.extend(anchors)
            else:
                baseline_temp = float(current_temp)
                for anchor in anchors:
                    if abs(anchor - baseline_temp) > 0.15:
                        temps_to_try.append(anchor)
                temps_to_try.extend(
                    [
                        max(0.0, min(1.0, baseline_temp - 0.2)),
                        max(0.0, min(1.0, baseline_temp + 0.2)),
                    ]
                )

            seen: set[float] = set()
            for temp in temps_to_try:
                if len(mutations) >= limit:
                    break
                temp = round(temp, 2)
                if temp in seen:
                    continue
                seen.add(temp)
                meta = dict(candidate.meta)
                meta.update(
                    {
                        "temperature": temp,
                        "edit": "temperature_shift",
                        "generation_method": "temperature_shift",
                        "parent": candidate.fingerprint,
                    }
                )
                mutations.append(Candidate(text=candidate.text, meta=meta))
        return mutations[:limit]

    async def _generate_single_reflection_mutations(
        self,
        parent_contexts: Sequence[dict[str, object]],
        limit: int,
    ) -> list[Candidate]:
        if not self.single_reflection_runner or limit <= 0:
            return []

        proposals: list[Candidate] = []
        for ctx in parent_contexts:
            if len(proposals) >= limit:
                break
            candidate = ctx["candidate"]
            failures = ctx.get("failures", []) or []
            traces: list[dict[str, object]] = []
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
                meta.update(
                    {
                        "edit": "reflection",
                        "generation_method": "reflection",
                        "parent": candidate.fingerprint,
                        "proposal_idx": idx,
                    }
                )
                proposals.append(Candidate(text=text, meta=meta))
                if len(proposals) >= limit:
                    break
        return proposals
