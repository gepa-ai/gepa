"""Mutation utilities for LLM-driven candidate generation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Sequence

from turbo_gepa.logging.logger import LogLevel, LoggerProtocol, StdOutLogger

from .interfaces import Candidate

StrategyRunner = Callable[
    [Sequence[dict[str, object]], int, list[dict[str, object]] | None],
    Awaitable[Sequence[str]],
]
BatchReflectionRunner = StrategyRunner
SpecInductionRunner = StrategyRunner
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
    # Amortization: number of items to request per LLM call
    mutations_per_call: int = 1
    specs_per_call: int = 1
    objective_key: str = "quality"


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
            "temperature_shift": {"trials": 0, "delta_sum": 0.0, "generated": 0, "promoted": 0},
            "incremental_reflection": {"trials": 0, "delta_sum": 0.0, "generated": 0, "promoted": 0},
            "spec_induction": {"trials": 0, "delta_sum": 0.0, "generated": 0, "promoted": 0},
        }
        self._operator_history: dict[str, deque[float]] = {key: deque(maxlen=20) for key in self._operator_stats}
        self.logger: LoggerProtocol = logger or StdOutLogger()
        self._metrics = metrics  # For tracking LLM calls in mutation generation
        self.objective_key = config.objective_key

    def set_reflection_examples(self, examples: list[dict[str, object]]) -> None:
        self._reflection_examples = examples

    def set_temperature_mutations_enabled(self, enabled: bool) -> None:
        """Toggle temperature exploration without rebuilding the mutator."""
        self.temperature_mutations_enabled = enabled

    def set_mutations_per_call(self, n: int) -> None:
        """Set how many reflection mutations to request per LLM call."""
        try:
            self.config.mutations_per_call = max(1, int(n))
        except Exception:
            self.config.mutations_per_call = 1

    def set_specs_per_call(self, n: int) -> None:
        """Set how many spec induction prompts to request per LLM call."""
        try:
            self.config.specs_per_call = max(1, int(n))
        except Exception:
            self.config.specs_per_call = 1

    def report_outcome(self, generation_method: str, delta_quality: float) -> None:
        stats = self._operator_stats.setdefault(
            generation_method, {"trials": 0, "delta_sum": 0.0, "generated": 0, "promoted": 0}
        )
        stats["trials"] += 1
        stats["delta_sum"] += delta_quality
        if delta_quality > 0:
            stats["promoted"] += 1
        history = self._operator_history.setdefault(generation_method, deque(maxlen=20))
        history.append(delta_quality)

    def _operator_weight(self, generation_method: str) -> float:
        history = self._operator_history.get(generation_method)
        success_rate = 0.0
        avg_delta = 0.0
        sample_size = 0

        if history:
            positives = sum(1 for delta in history if delta > 0)
            sample_size = len(history)
            success_rate = positives / sample_size if sample_size else 0.0
            avg_delta = sum(history) / max(1, sample_size)
        else:
            stats = self._operator_stats.get(generation_method)
            if stats and stats["trials"] > 0:
                sample_size = stats["trials"]
                success_rate = stats["promoted"] / max(1, sample_size)
                avg_delta = stats["delta_sum"] / max(1, sample_size)

        if sample_size >= 6 and success_rate == 0.0:
            return 0.0  # temporary cooldown for consistently failing operators

        # Penalize consistently negative lift more aggressively
        if sample_size >= 8 and avg_delta <= -0.03:
            return 0.001

        weight = success_rate * max(avg_delta, 0.0)
        return max(weight, 0.0) + 0.01

    def _allocate_operator_budget(
        self,
        total_budget: int,
        available_ops: list[str],
    ) -> dict[str, int]:
        """Allocate mutation slots across operators using bandit-style weights."""
        if total_budget <= 0 or not available_ops:
            return {name: 0 for name in available_ops}

        weights: dict[str, float] = {}
        for name in available_ops:
            weights[name] = max(self._operator_weight(name), 0.01)

        weight_sum = sum(weights.values())
        budgets = {name: 0 for name in available_ops}

        if weight_sum <= 0:
            # Evenly distribute when we have no signal yet
            for name in available_ops:
                budgets[name] = total_budget // len(available_ops)
            remainder = total_budget - sum(budgets.values())
            for name in available_ops[:remainder]:
                budgets[name] += 1
            return budgets

        remaining = total_budget
        sorted_ops = sorted(available_ops, key=lambda name: weights[name], reverse=True)
        for idx, name in enumerate(sorted_ops):
            ops_left = len(sorted_ops) - idx
            if remaining <= 0:
                break
            ideal = total_budget * (weights[name] / weight_sum)
            quota = int(round(ideal))
            # Ensure every operator gets at least one slot while budget remains
            min_quota = 1 if remaining >= ops_left else 0
            quota = max(min_quota, quota)
            quota = min(quota, remaining)
            budgets[name] = quota
            remaining -= quota

        # Distribute any leftover (due to rounding) to the heaviest operators
        idx = 0
        while remaining > 0 and sorted_ops:
            name = sorted_ops[idx % len(sorted_ops)]
            budgets[name] += 1
            remaining -= 1
            idx += 1

        return budgets

    async def propose(
        self,
        parent_contexts: list[dict[str, object]],
        num_mutations: int,
        task_examples: list[dict[str, object]] | None = None,
        candidate_sink: Callable[[Candidate], Awaitable[None]] | None = None,
    ) -> list[Candidate]:
        """
        Generate mutated candidates and stream them to orchestrator.

        Args:
            parent_contexts: List of parent candidate contexts
            num_mutations: Number of mutations to generate
            task_examples: Optional task examples for spec induction
            candidate_sink: Async callback to stream candidates as they're created

        The method blends prompt-centric strategies (reflection/spec induction) during Phase 1.
        During a temperature-optimization phase (Phase 2), only deterministic temperature
        mutations are generated.
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

        # Phase 2: exclusively run temperature mutations
        if self.temperature_mutations_enabled:
            temp_start = time.time()
            temp_mutations = self._temperature_mutations(parent_contexts, total_budget)
            for temp_mut in temp_mutations:
                if candidate_sink:
                    await candidate_sink(temp_mut)
            elapsed = time.time() - temp_start
            self.logger.log(
                f"⏱️  Temperature phase generated {len(temp_mutations)} candidates in {elapsed:.2f}s",
                LogLevel.INFO,
            )
            return temp_mutations

        has_reflection = self.batch_reflection_runner is not None
        has_spec = bool(self.spec_induction_runner and task_examples)

        available_ops: list[str] = []
        if has_reflection:
            available_ops.append("incremental_reflection")
        if has_spec:
            available_ops.append("spec_induction")

        operator_budgets = self._allocate_operator_budget(total_budget, available_ops)
        reflection_budget = operator_budgets.get("incremental_reflection", total_budget if not available_ops else 0)
        spec_quota = operator_budgets.get("spec_induction", 0)
        proposals: list[Candidate] = []

        # 2 & 3) Run reflection and spec induction CONCURRENTLY, streaming as they complete
        llm_start = time.time()

        async def run_reflection() -> list[Candidate]:
            if reflection_budget <= 0 or not has_reflection:
                return []
            if self.batch_reflection_runner:
                return await self._generate_incremental_mutations(
                    parent_contexts,
                    reflection_budget,
                    candidate_sink=candidate_sink,
                )
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
                candidate_sink=candidate_sink,
            )

        # Both tasks stream candidates via candidate_sink as they complete
        reflection_task = asyncio.create_task(run_reflection())
        spec_task = asyncio.create_task(run_spec())

        # Wait for both to finish (candidates already streamed)
        for completed in asyncio.as_completed([reflection_task, spec_task]):
            batch = await completed
            proposals.extend(batch)

        llm_time = time.time() - llm_start
        propose_total = time.time() - propose_start

        # Note: Filtering removed - streaming candidates go directly to orchestrator
        # The orchestrator does deduplication via _pending_fingerprints

        # Log timing breakdown
        self.logger.log("⏱️  Mutator timing (STREAMING):")
        self.logger.log(f"   Temperature: 0.00s (0 sent instantly)")
        self.logger.log(f"   LLM calls (parallel): {llm_time:.2f}s")
        self.logger.log(f"     - Total streamed: {len(proposals)} candidates")
        self.logger.log(f"   Total propose: {propose_total:.2f}s")
        self.logger.log(f"   ✅ Candidates streamed to orchestrator during generation")

        return proposals  # Return all for metrics, but candidates already streamed

    async def _generate_incremental_mutations(
        self,
        parent_contexts: list[dict[str, object]],
        num_mutations: int,
        candidate_sink: Callable[[Candidate], Awaitable[None]] | None = None,
    ) -> list[Candidate]:
        """Generate mutations by synthesizing ideas from successful parent prompts.

        Args:
            parent_contexts: List of parent candidate contexts
            num_mutations: Number of mutations to generate
            candidate_sink: Optional async callback to stream candidates as they're created
        """
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
            context_meta["objective_key"] = self.objective_key
            reflection_contexts.append(
                {
                    "prompt": candidate.text,
                    "traces": traces,
                    "meta": context_meta,
                    "candidate": candidate,
                    "failures": failures,
                }
            )

        # Prepare parent candidates for lineage tracking
        parent_candidates = [ctx["candidate"] for ctx in parent_contexts]

        # STREAMING: Create callback that builds Candidate immediately when text arrives
        proposals: list[Candidate] = []

        async def on_text_ready(text: str, idx: int) -> None:
            """Called when each mutation text completes - immediately create Candidate and stream it."""
            # Rotate through available parents to maintain lineage diversity
            if not parent_candidates:
                parent_candidate = self._best_parent_candidate(parent_contexts)
            else:
                parent_candidate = parent_candidates[idx % len(parent_candidates)]

            meta = {k: v for k, v in parent_candidate.meta.items() if k != "temperature"}
            meta.pop("_sched_key", None)
            meta.update(
                {
                    "source": "mutation",
                    "edit": "incremental_reflection",
                    "generation_method": "incremental_reflection",
                    "operator": "incremental_reflection",
                    "parent": parent_candidate.fingerprint,
                    "parent_sched_key": parent_candidate.meta.get("_sched_key", parent_candidate.fingerprint),
                    "proposal_idx": idx,
                    "num_parents_seen": len(parent_contexts),
                }
            )
            candidate = Candidate(text=text, meta=meta)
            proposals.append(candidate)

            # STREAMING: Immediately send to orchestrator
            if candidate_sink:
                await candidate_sink(candidate)

        # Track LLM calls for reflection mutations
        import time
        _start_reflection = time.time()
        per_call = max(1, getattr(self.config, "mutations_per_call", 1))
        mutated_texts = await self._collect_text_batches(
            lambda: self.batch_reflection_runner(reflection_contexts, per_call, None),
            num_mutations,
            max(1, min(self.config.reflection_batch_size, num_mutations)),
            result_callback=on_text_ready,
            items_per_call=per_call,
        )
        _elapsed_reflection = time.time() - _start_reflection

        # Record reflection LLM call in metrics
        if self._metrics is not None:
            for _ in range(len(mutated_texts)):
                self._metrics.record_llm_call("reflection", _elapsed_reflection / max(1, len(mutated_texts)))

        return proposals

    async def _generate_spec_induction_mutations(
        self,
        task_examples: list[dict[str, object]],
        num_mutations: int,
        parent_contexts: list[dict[str, object]],
        candidate_sink: Callable[[Candidate], Awaitable[None]] | None = None,
    ) -> list[Candidate]:
        """Generate fresh specifications from task I/O examples (PROMPT-MII style)."""
        if num_mutations <= 0 or not self.spec_induction_runner:
            return []

        # Build reflection contexts
        reflection_contexts = []
        for ctx in parent_contexts:
            candidate = ctx["candidate"]
            failures = ctx.get("failures", []) or []

            traces = []
            for _example_id, trace_list in failures:
                traces.extend(trace_list)

            traces = traces[: self.config.reflection_batch_size]

            context_meta = {k: v for k, v in candidate.meta.items() if k != "temperature"}
            context_meta["objective_key"] = self.objective_key
            reflection_contexts.append(
                {
                    "prompt": candidate.text,
                    "traces": traces,
                    "meta": context_meta,
                    "candidate": candidate,
                    "failures": failures,
                }
            )

        parent_candidates = [ctx["candidate"] for ctx in parent_contexts]
        proposals: list[Candidate] = []

        async def on_text_ready(text: str, idx: int) -> None:
            """STREAMING: Build Candidate immediately when spec text arrives."""
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
                    "source": "mutation",
                    "edit": "spec_induction",
                    "generation_method": "spec_induction",
                    "operator": "spec_induction",
                    "proposal_idx": idx,
                    "num_examples_seen": len(task_examples),
                }
            )
            candidate = Candidate(text=text, meta=meta)
            proposals.append(candidate)

            # STREAMING: Immediately send to orchestrator
            if candidate_sink:
                await candidate_sink(candidate)

        import time
        _start_spec = time.time()
        per_call = max(1, getattr(self.config, "specs_per_call", 1))
        spec_texts = await self._collect_text_batches(
            lambda: self.spec_induction_runner(reflection_contexts, per_call, task_examples),
            num_mutations,
            max(1, min(4, num_mutations)),
            result_callback=on_text_ready,
            items_per_call=per_call,
        )
        _elapsed_spec = time.time() - _start_spec

        if self._metrics is not None:
            for _ in range(len(spec_texts)):
                self._metrics.record_llm_call("spec_induction", _elapsed_spec / max(1, len(spec_texts)))

        return proposals

    async def _collect_text_batches(
        self,
        factory: Callable[[], Awaitable[Sequence[str]]],
        total: int,
        max_concurrency: int,
        early_stop_fraction: float = 0.85,  # Not used anymore - kept for API compat
        result_callback: Callable[[str, int], Awaitable[None]] | None = None,  # Async callback: (text, index) -> None
        items_per_call: int = 1,
    ) -> list[str]:
        """
        Launch all mutation tasks immediately and stream results as they complete.

        If result_callback is provided, it will be called with (text, index) as soon as each mutation completes.
        This enables the orchestrator to start evaluating candidates before all mutations finish.

        No artificial concurrency limits, no early stopping, no straggler detection.
        Just pure async concurrency - let the LLM provider handle rate limits.
        """
        import asyncio

        if total <= 0:
            return []

        import math as _math
        num_calls = max(1, _math.ceil(total / max(1, items_per_call)))
        self.logger.log(f"🌀 Launching {num_calls} mutation task(s) (target {total}, {items_per_call}/call)")

        # Launch tasks immediately
        tasks = [asyncio.create_task(factory()) for _ in range(num_calls)]

        # Stream results as they complete (don't wait for all)
        results: list[str] = []
        for idx, completed in enumerate(asyncio.as_completed(tasks)):
            try:
                batch = await completed
                if batch:
                    added = 0
                    for text in batch:
                        if len(results) >= total:
                            break
                        results.append(text)
                        if result_callback:
                            await result_callback(text, len(results) - 1)
                        added += 1
            except Exception as e:
                self.logger.log(f"   ⚠️ Mutation task failed: {e}")

        self.logger.log(f"✅ Generated {len(results)}/{total} mutations")
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
            key = self.objective_key
            value = meta.get(key)
            if not isinstance(value, (int, float)):
                value = meta.get("quality")
            if isinstance(value, (int, float)):
                return float(value)
            parent_objectives = meta.get("parent_objectives")
            if isinstance(parent_objectives, dict):
                quality = parent_objectives.get(key)
                if quality is None:
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
