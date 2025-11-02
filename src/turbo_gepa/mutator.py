"""Mutation utilities for LLM-driven candidate generation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Sequence

from turbo_gepa.logging.logger import LogLevel, LoggerProtocol, StdOutLogger

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
        batch_reflection_runner: BatchReflectionRunner | None = None,
        spec_induction_runner: SpecInductionRunner | None = None,
        seed: int | None = None,
        temperature_mutations_enabled: bool = True,
        logger: LoggerProtocol | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.config = config
        extra_validators = list(validators or [])
        extra_validators.append(_default_token_validator(config.max_tokens))
        self.validators = extra_validators
        self.batch_reflection_runner = batch_reflection_runner
        self.spec_induction_runner = spec_induction_runner
        self.temperature_mutations_enabled = temperature_mutations_enabled
        self._reflection_examples: list[dict[str, object]] = []
        self._operator_stats: dict[str, dict[str, float]] = {
            "temperature_shift": {"trials": 0, "delta_sum": 0.0},
            "incremental_reflection": {"trials": 0, "delta_sum": 0.0},
            "spec_induction": {"trials": 0, "delta_sum": 0.0},
        }
        self._operator_history: dict[str, deque[float]] = {key: deque(maxlen=20) for key in self._operator_stats}
        self.logger: LoggerProtocol = logger or StdOutLogger()
        self._metrics = metrics  # For tracking LLM calls in mutation generation

    def set_reflection_examples(self, examples: list[dict[str, object]]) -> None:
        self._reflection_examples = examples

    def set_temperature_mutations_enabled(self, enabled: bool) -> None:
        """Toggle temperature exploration without rebuilding the mutator."""
        self.temperature_mutations_enabled = enabled

    def report_outcome(self, generation_method: str, delta_quality: float) -> None:
        stats = self._operator_stats.setdefault(generation_method, {"trials": 0, "delta_sum": 0.0})
        stats["trials"] += 1
        stats["delta_sum"] += delta_quality
        history = self._operator_history.setdefault(generation_method, deque(maxlen=20))
        history.append(delta_quality)

    def _operator_weight(self, generation_method: str) -> float:
        history = self._operator_history.get(generation_method)
        if history and len(history) > 0:
            avg_delta = sum(history) / len(history)
            return max(0.0, avg_delta) + 0.01
        stats = self._operator_stats.get(generation_method)
        if stats and stats["trials"] > 0:
            avg_delta = stats["delta_sum"] / max(1, stats["trials"])
            return max(0.0, avg_delta) + 0.01
        return 0.01

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

        reflection_weight = self._operator_weight("incremental_reflection")
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

        # Stream mutations as they complete instead of waiting for all
        reflection_task = asyncio.create_task(run_reflection())
        spec_task = asyncio.create_task(run_spec())

        # Use as_completed to stream mutations without waiting for slowest task
        for completed in asyncio.as_completed([reflection_task, spec_task]):
            batch = await completed
            proposals.extend(batch)

        llm_time = time.time() - llm_start

        filter_start = time.time()
        filtered = self._filter(proposals)[:num_mutations]
        filter_time = time.time() - filter_start

        propose_total = time.time() - propose_start

        # Log timing breakdown
        self.logger.log("⏱️  Mutator timing breakdown:")
        self.logger.log(f"   Temperature mutations: {temp_time:.2f}s ({len(temp_mutations)} generated)")
        self.logger.log(f"   LLM calls (parallel): {llm_time:.2f}s")
        self.logger.log(f"     - Incremental reflection: {len(reflection_mutations)} mutations")
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

            context_meta = {k: v for k, v in candidate.meta.items() if k != "temperature"}
            reflection_contexts.append(
                {
                    "prompt": candidate.text,
                    "traces": traces,
                    "meta": context_meta,
                }
            )

        # Track LLM calls for reflection mutations
        import time
        _start_reflection = time.time()
        mutated_texts = await self._collect_text_batches(
            lambda: self.batch_reflection_runner(reflection_contexts, 1),
            num_mutations,
            max(1, min(self.config.reflection_batch_size, num_mutations)),
        )
        _elapsed_reflection = time.time() - _start_reflection

        # Record reflection LLM call in metrics
        if self._metrics is not None:
            # Track one LLM call per mutation generated (they're batched but async)
            for _ in range(len(mutated_texts)):
                self._metrics.record_llm_call("reflection", _elapsed_reflection / max(1, len(mutated_texts)))

        # Convert to Candidates with metadata tracking generation method
        proposals: list[Candidate] = []
        parent_candidates = [ctx["candidate"] for ctx in parent_contexts]
        for idx, text in enumerate(mutated_texts):
            # Rotate through available parents to maintain lineage diversity
            if not parent_candidates:
                parent_candidate = self._best_parent_candidate(parent_contexts)
            else:
                parent_candidate = parent_candidates[idx % len(parent_candidates)]

            meta = {k: v for k, v in parent_candidate.meta.items() if k != "temperature"}
            meta.pop("_sched_key", None)
            meta.update(
                {
                    "edit": "incremental_reflection",  # Track generation method
                    "generation_method": "incremental_reflection",  # Explicit tracking for analysis
                    "operator": "incremental_reflection",  # For metrics tracking
                    "parent": parent_candidate.fingerprint,
                    "parent_sched_key": parent_candidate.meta.get("_sched_key", parent_candidate.fingerprint),
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

            context_meta = {k: v for k, v in candidate.meta.items() if k != "temperature"}
            reflection_contexts.append(
                {
                    "prompt": candidate.text,
                    "traces": traces,
                    "meta": context_meta,
                }
            )

        # Generate spec induction mutations directly via LLM
        import time
        _start_spec = time.time()
        spec_texts = await self._collect_text_batches(
            lambda: self.spec_induction_runner(reflection_contexts, 1),
            num_mutations,
            max(1, min(4, num_mutations)),
        )
        _elapsed_spec = time.time() - _start_spec

        # Record spec_induction LLM call in metrics
        if self._metrics is not None:
            # Track one LLM call per mutation generated
            for _ in range(len(spec_texts)):
                self._metrics.record_llm_call("spec_induction", _elapsed_spec / max(1, len(spec_texts)))

        # Convert to Candidates with metadata tracking generation method
        proposals: list[Candidate] = []
        parent_candidates = [ctx["candidate"] for ctx in parent_contexts]

        for idx, text in enumerate(spec_texts):
            if not parent_candidates:
                parent_candidate = self._best_parent_candidate(parent_contexts)
            else:
                parent_candidate = parent_candidates[idx % len(parent_candidates)]

            meta = {k: v for k, v in parent_candidate.meta.items() if k != "temperature"}
            meta.pop("_sched_key", None)
            meta["parent"] = parent_candidate.fingerprint
            meta["parent_sched_key"] = parent_candidate.meta.get("_sched_key", parent_candidate.fingerprint)
            meta.update(
                {
                    "edit": "spec_induction",  # Track generation method
                    "generation_method": "spec_induction",  # Explicit tracking for analysis
                    "operator": "spec_induction",  # For metrics tracking
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
        result_callback: Callable[[str], None] | None = None,  # Stream results as they arrive
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

            if len(mutation_durations) >= 3:
                import statistics

                mean_duration = statistics.fmean(mutation_durations)
                stdev = statistics.pstdev(mutation_durations) if len(mutation_durations) > 1 else 0.0
                threshold = mean_duration + (2.0 * stdev if stdev > 0 else mean_duration * 2.0)
                now = time.time()
                cancelled = False
                for task, start_time in list(pending.items()):
                    elapsed_task = now - start_time
                    if elapsed_task > threshold:
                        self.logger.log(
                            f"   ⚠️ Mutation early stop: produced={len(results)}/{total} "
                            f"({len(results) / total * 100:.0f}%), remaining={len(pending)}, "
                            f"elapsed={elapsed_task:.1f}s > threshold {threshold:.1f}s"
                        )
                        task.cancel()
                        cancelled = True
                if cancelled and getattr(self, "_metrics", None):
                    self._metrics.record_early_stop("stragglers")  # type: ignore[attr-defined]

            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task_start_time = pending.pop(task)
                task_duration = time.time() - task_start_time
                try:
                    batch = task.result()
                    if batch:
                        text_result = batch[0]
                        results.append(text_result)
                        mutation_durations.append(task_duration)  # Track individual duration
                        self.logger.log(
                            f"   ✅ Mutation task complete in {task_duration:.2f}s "
                            f"(generated {len(batch)} candidates, total so far {len(results)})"
                        )
                        # Stream result immediately via callback if provided
                        if result_callback:
                            result_callback(text_result)
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
                meta.pop("_sched_key", None)
                meta.update(
                    {
                        "temperature": temp,
                        "edit": "temperature_shift",
                        "generation_method": "temperature_shift",
                        "operator": "temperature_shift",  # For metrics tracking
                        "parent": candidate.fingerprint,
                        "parent_sched_key": candidate.meta.get("_sched_key", candidate.fingerprint),
                    }
                )
                mutations.append(Candidate(text=candidate.text, meta=meta))
        return mutations[:limit]
