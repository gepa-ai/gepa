"""
High-level orchestrator coordinating selection, evaluation, and mutation.

This module stitches together archive, scheduler, evaluator, and mutator
components. The orchestrator operates within a single island; multi-process
execution is handled by ``islands.spawn_islands`` which can launch several
instances of the loop in parallel.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .archive import Archive, ArchiveEntry
from .cache import DiskCache
from .config import Config
from .evaluator import AsyncEvaluator
from .interfaces import Candidate, EvalResult
from .islands import IslandContext, integrate_in, migrate_out
from turbo_gepa.logging.logger import LoggerProtocol, StdOutLogger
from .mutator import Mutator
from .sampler import InstanceSampler
from .scheduler import BudgetedScheduler, SchedulerConfig
from .stop_governor import StopGovernor, StopGovernorConfig, EpochMetrics, compute_hypervolume_2d


class Orchestrator:
    """Single-island orchestrator loop."""

    def __init__(
        self,
        config: Config,
        evaluator: AsyncEvaluator,
        archive: Archive,
        sampler: InstanceSampler,
        mutator: Mutator,
        cache: DiskCache,
        *,
        island_context: Optional[IslandContext] = None,
        show_progress: bool = True,
        stop_governor: Optional[StopGovernor] = None,
        enable_auto_stop: bool = False,
        example_sampler: Optional[Callable[[int], List[Dict[str, object]]]] = None,
        metrics_callback: Optional[Callable] = None,
        logger: Optional[LoggerProtocol] = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.archive = archive
        self.sampler = sampler
        self.mutator = mutator
        self.cache = cache
        self.island_context = island_context
        self.example_sampler = example_sampler
        self.scheduler = BudgetedScheduler(
            SchedulerConfig(
                shards=config.shards,
                eps_improve=config.eps_improve,
                quantile=config.cohort_quantile,
            )
        )
        self.queue: Deque[Candidate] = deque(maxlen=config.queue_limit)
        self._per_shard_queue: List[Deque[Candidate]] = [deque() for _ in range(len(config.shards))]
        self._pending_fingerprints: Set[str] = set()
        self._inflight_fingerprints: Set[str] = set()
        self._next_shard: int = 0
        self.latest_results: Dict[str, EvalResult] = {}
        self.evaluations_run: int = 0
        self.round_index: int = 0
        self.show_progress = show_progress
        island_id = island_context.island_id if island_context else 0
        self.island_id = island_id
        self.stop_governor = stop_governor if enable_auto_stop else None
        self.enable_auto_stop = enable_auto_stop
        self.total_tokens_spent: int = 0
        self.qd_cells_seen: set[tuple] = set()
        self._shard_cache: Dict[tuple[int, float], List[str]] = {}
        self.metrics_callback = metrics_callback
        self.logger: LoggerProtocol = logger or StdOutLogger()
        self.max_rounds: Optional[int] = None
        self.max_evaluations: Optional[int] = None
        self.rounds_completed: int = 0

        # Streaming mode: buffer for mutations generated in background
        self._mutation_buffer: Deque[Candidate] = deque()
        self._mutation_task: Optional[asyncio.Task] = None
        self.eval_batches_completed: int = 0  # For migration timing

        # Streaming evaluation infrastructure
        self._inflight_tasks: Dict[str, asyncio.Task] = {}  # cand_hash -> task
        self._result_queue: asyncio.Queue[Tuple[Candidate, EvalResult | Exception, int, float]] = asyncio.Queue()
        self._total_inflight: int = 0  # Total evaluations in flight across all shards
        total_cap = self.config.max_total_inflight or self.config.eval_concurrency
        self._effective_concurrency: int = max(1, total_cap)
        self._max_total_inflight: int = self._effective_concurrency
        num_shards = max(1, len(self.config.shards))
        base = max(1, self._max_total_inflight // num_shards)
        remainder = max(0, self._max_total_inflight - base * num_shards)
        self._shard_capacity = [base + (1 if i < remainder else 0) for i in range(num_shards)]
        # Ensure we always honour the total budget even if integer division truncates
        if sum(self._shard_capacity) < self._max_total_inflight:
            deficit = self._max_total_inflight - sum(self._shard_capacity)
            for i in range(deficit):
                self._shard_capacity[i % num_shards] += 1
        self._shard_inflight: List[int] = [0] * num_shards
        self._last_debug_log: float = 0.0
        self._last_budget_log: float = 0.0
        # Global example-level inflight budget (hard cap across candidates)
        self._examples_inflight: int = 0
        self._last_shared: Dict[str, float] = {}
        # Evolution tracking
        self._mutations_requested: int = 0
        self._mutations_generated: int = 0
        self._mutations_enqueued: int = 0
        self._parent_children: defaultdict[str, set[str]] = defaultdict(set)
        self._children_seen: Set[str] = set()
        self._promoted_children: Set[str] = set()
        # Track generation depth for each candidate (fingerprint -> generation number)
        self._candidate_generations: Dict[str, int] = {}

        # Deficit scheduler state for fair rung allocation
        import math
        self._rung_keys: List[str] = [str(shard) for shard in self.config.shards]
        n_rungs = len(self._rung_keys)
        # Equal shares: each rung gets 1/n of launch opportunities
        self._rung_shares: Dict[str, float] = {
            key: 1.0 / n_rungs for key in self._rung_keys
        }
        # Deficit accumulates when a rung doesn't get scheduled
        self._rung_deficit: Dict[str, float] = {
            key: 0.0 for key in self._rung_keys
        }
        # Track inflight per rung (in addition to per-shard tracking)
        self._inflight_by_rung: Dict[str, int] = {
            key: 0 for key in self._rung_keys
        }
        # Per-rung inflight capacity (hard cap to prevent oversubscription)
        self._rung_capacity: Dict[str, int] = {
            key: math.ceil(self._rung_shares[key] * self._max_total_inflight)
            for key in self._rung_keys
        }

    def enqueue(self, candidates: Iterable[Candidate]) -> None:
        for candidate in candidates:
            if not candidate.text.strip():
                continue
            fingerprint = candidate.fingerprint
            if fingerprint in self._pending_fingerprints or fingerprint in self._inflight_fingerprints:
                continue
            if len(self.queue) == self.queue.maxlen and self.queue:
                evicted = self.queue.popleft()
                self._pending_fingerprints.discard(evicted.fingerprint)
                for dq in self._per_shard_queue:
                    try:
                        dq.remove(evicted)
                        break
                    except ValueError:
                        continue
            idx = self.scheduler.current_shard_index(candidate)
            shard_idx = min(idx, len(self._per_shard_queue) - 1)
            self._per_shard_queue[shard_idx].append(candidate)
            self.queue.append(candidate)
            self._pending_fingerprints.add(fingerprint)

    def _get_rung_key(self, candidate: Candidate) -> str:
        """Get rung key string for a candidate based on its current shard.

        Uses scheduler.current_shard_index to ensure consistency with enqueue logic.
        """
        idx = self.scheduler.current_shard_index(candidate)
        shard_fraction = self.config.shards[min(idx, len(self.config.shards) - 1)]
        return str(shard_fraction)

    async def run(
        self,
        seeds: Sequence[Candidate],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        resume: bool = True,  # Enable automatic resume from cache
    ) -> None:
        """Execute the orchestration loop until budgets are exhausted.

        Args:
            seeds: Initial candidates to start optimization
            max_rounds: Maximum number of rounds (None = unlimited)
            max_evaluations: Maximum evaluations (None = unlimited)
            resume: If True, automatically resume from saved state if available
        """
        import time

        if not seeds:
            return

        # Store budgets for metrics reporting
        self.max_rounds = max_rounds
        self.max_evaluations = max_evaluations

        # Attempt to resume from saved state
        resumed = False
        if resume and self.cache.has_state():
            state = self.cache.load_state()
            if state:
                await self._restore_state(state)
                resumed = True
                if self.show_progress:
                    self.logger.log(
                        f"🔄 Resumed from round {self.round_index + 1} ({self.evaluations_run} evaluations)"
                    )

        if not resumed:
            await self._seed_archive(seeds)
            if self.metrics_callback is not None:
                from .metrics import extract_metrics
                self.metrics_callback(extract_metrics(self))

        # Ensure mutation generation task is running
        if self._mutation_task is None or self._mutation_task.done():
            async def _mutation_wrapper():
                try:
                    await self._spawn_mutations(callback=self._mutation_buffer.append)
                except Exception as exc:  # pragma: no cover - debugging aid
                    self.logger.log(f"❌ Mutation task error: {type(exc).__name__}: {exc}")
                    raise
            self._mutation_task = asyncio.create_task(_mutation_wrapper())

        # Window tracking mirrors the legacy batch counter so downstream logging stays intact
        window_size = max(1, self.config.batch_size)
        self.eval_batches_completed = max(self.eval_batches_completed, self.round_index)
        window_id = self.eval_batches_completed
        window_start_evals = self.evaluations_run

        # Simple progress tracking for streaming mode
        loop_iter = 0
        last_progress_display = self.evaluations_run

        while True:
            if max_rounds is not None and window_id >= max_rounds:
                break
            if max_evaluations is not None and self.evaluations_run >= max_evaluations:
                break

            # 1) Move freshly generated mutations into the ready queue
            while self._mutation_buffer:
                self.enqueue([self._mutation_buffer.popleft()])

            # 2) Launch as many evaluations as capacity allows
            launched = await self._stream_launch_ready(window_id, max_evaluations)

            # 3) Drain completed results and update bookkeeping
            drained = await self._stream_drain_results()

            # Simple progress indicator: show activity frequently
            loop_iter += 1
            if self.show_progress:
                # Show update every 5 completions OR every 20 loop iterations (whichever comes first)
                # This ensures we see activity even when evaluations are launching but not yet completing
                should_update = False
                if drained > 0:
                    evals_since_display = self.evaluations_run - last_progress_display
                    if evals_since_display >= 5:
                        should_update = True
                elif loop_iter % 20 == 0 and self._total_inflight > 0:
                    # Show activity indicator even if nothing completed yet
                    should_update = True

                if should_update:
                    pareto = self.archive.pareto_entries()
                    best_q = self._get_best_quality_from_full_shard() if pareto else 0.0
                    self.logger.log(
                        f"🔄 Evaluations: {self.evaluations_run} | Inflight: {self._total_inflight} | "
                        f"ExInflight: {self._examples_inflight}/{self._effective_concurrency} | "
                        f"Queue: {len(self.queue)} | Best: {best_q:.1%}"
                    )
                    last_progress_display = self.evaluations_run
                    if self.metrics_callback is not None:
                        from .metrics import extract_metrics
                        self.metrics_callback(extract_metrics(self))

            # 4) Refresh mutation generation if queues are running low
            if self._mutation_task and self._mutation_task.done():
                self._mutation_task = None
            # Only spawn new mutations if we haven't exceeded the evaluation budget
            should_spawn_mutations = (
                (len(self.queue) + len(self._mutation_buffer)) < self.config.mutation_buffer_min
                and (max_evaluations is None or self.evaluations_run < max_evaluations)
            )
            if should_spawn_mutations:
                if self._mutation_task is None:
                    self._mutation_task = asyncio.create_task(
                        self._spawn_mutations(callback=self._mutation_buffer.append)
                    )

            # 5) Window completion => keep round-indexed metrics in sync
            if (
                not self.queue
                and not self._mutation_buffer
                and (self._mutation_task is not None or self._total_inflight > 0)
            ):
                import time as _time_debug
                now = _time_debug.time()
                if now - self._last_debug_log > 5.0:
                    if self._mutation_task is None:
                        task_state = "idle"
                    elif self._mutation_task.done():
                        try:
                            exc = self._mutation_task.exception()
                        except asyncio.CancelledError:
                            exc = "cancelled"
                        task_state = f"done({exc})"
                    else:
                        task_state = "running"
                    self.logger.log(
                        f"🧭 Debug: queue=0 buffer=0 inflight={self._total_inflight} "
                        f"examples_inflight={self._examples_inflight}/{self._effective_concurrency} "
                        f"mutation_task={task_state} pending_fp={len(self._pending_fingerprints)} "
                        f"inflight_fp={len(self._inflight_fingerprints)}"
                    )
                    self._last_debug_log = now

            evals_in_window = self.evaluations_run - window_start_evals

            if evals_in_window >= window_size:
                window_id += 1
                self.eval_batches_completed = window_id
                self.round_index = window_id
                self.rounds_completed = window_id
                window_start_evals = self.evaluations_run

                # Call metrics callback if provided
                if self.metrics_callback is not None:
                    from .metrics import extract_metrics
                    metrics = extract_metrics(self)
                    self.metrics_callback(metrics)

                # Clear inline progress before showing summary
                if self.show_progress:
                    # logger prints discrete lines; no inline clearing needed
                    pass

                self._shard_cache.clear()

                if self.config.migration_period and window_id % self.config.migration_period == 0:
                    await self._maybe_migrate()

                await self._save_state()

                if self.enable_auto_stop and self.stop_governor is not None:
                    should_stop, debug_info = self._check_stop_governor()
                    if should_stop:
                        if self.show_progress:
                            self.logger.log(f"🛑 Auto-stopping: {debug_info['reason']}")
                        break

            # 6) Target quality guard
            if await self._stream_check_target(window_id):
                break

            # 7) Idle detection – nothing running, nothing queued, mutations finished
            if (
                launched == 0
                and drained == 0
                and self._total_inflight == 0
                and not self.queue
                and not self._mutation_buffer
            ):
                if self._mutation_task is None:
                    break

            # Avoid busy spinning if we neither launched nor drained work
            if launched == 0 and drained == 0:
                await asyncio.sleep(0.01)

        # === Cleanup ===
        # Wait for all inflight evaluations to complete
        if self._inflight_tasks:
            await asyncio.gather(*self._inflight_tasks.values(), return_exceptions=True)

        # Drain any straggling results that finished during cleanup
        await self._stream_drain_results(timeout=0.0)

        # Clean up any pending mutation task
        if self._mutation_task and not self._mutation_task.done():
            self._mutation_task.cancel()
            # Don't wait for it - just let it cancel in the background

        # Final save
        await self._save_state()

        await self.finalize()

        # Clear state only if we reached completion (not if stopped early)
        completed = (
            (max_rounds is not None and window_id >= max_rounds) or
            (max_evaluations is not None and self.evaluations_run >= max_evaluations)
        )
        if completed:
            self.cache.clear_state()

    # === Streaming helpers ===

    def _stream_can_launch(self, candidate: Candidate, max_evaluations: Optional[int]) -> bool:
        if max_evaluations is not None and self.evaluations_run >= max_evaluations:
            return False

        shard_count = len(self._shard_capacity)
        if shard_count == 0:
            return False

        shard_idx = min(self.scheduler.current_shard_index(candidate), shard_count - 1)
        if self._shard_inflight[shard_idx] >= self._shard_capacity[shard_idx]:
            return False
        if self._total_inflight >= self._max_total_inflight:
            return False
        return True

    async def _stream_launch_ready(self, window_id: int, max_evaluations: Optional[int]) -> int:
        """Launch ready candidates using deficit-based scheduling for fair rung allocation."""
        launched = 0
        shard_count = len(self._per_shard_queue)
        if shard_count == 0:
            return launched

        # Try launching until we can't launch anymore
        max_attempts = shard_count * 10  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Accumulate deficit for all rungs (every tick)
            # Clamp deficit for empty queues to prevent infinite credit buildup
            for shard_idx, key in enumerate(self._rung_keys):
                if shard_idx < len(self._per_shard_queue):
                    queue = self._per_shard_queue[shard_idx]
                    if queue:
                        # Queue has work: accumulate deficit normally
                        self._rung_deficit[key] += self._rung_shares[key]
                    else:
                        # Queue empty: reset deficit to non-negative share baseline
                        self._rung_deficit[key] = max(0.0, min(self._rung_deficit[key], self._rung_shares[key]))

            # Calculate total inflight for normalization
            total_inflight = max(1, sum(self._inflight_by_rung.values()))

            # Find best rung by deficit score
            best_score = float('-inf')
            best_shard_idx = None
            best_rung_key = None

            for shard_idx in range(shard_count):
                queue = self._per_shard_queue[shard_idx]
                if not queue:
                    continue

                # Get rung key for this shard (consistent with scheduler)
                shard_fraction = self.config.shards[min(shard_idx, len(self.config.shards) - 1)]
                rung_key = str(shard_fraction)

                # Check per-rung inflight cap before considering this rung
                if self._inflight_by_rung[rung_key] >= self._rung_capacity[rung_key]:
                    continue  # Rung at capacity, skip

                # Calculate deficit score: deficit - (inflight_fraction)
                # Higher score = more "starved" and deserves priority
                inflight_penalty = self._inflight_by_rung[rung_key] / total_inflight
                score = self._rung_deficit[rung_key] - inflight_penalty

                if score > best_score:
                    best_score = score
                    best_shard_idx = shard_idx
                    best_rung_key = rung_key

            # No eligible rung found
            if best_shard_idx is None:
                break

            # Try to launch from best rung
            queue = self._per_shard_queue[best_shard_idx]
            candidate = queue[0]

            if self._stream_can_launch(candidate, max_evaluations):
                # Remove from queue
                queue.popleft()
                try:
                    self.queue.remove(candidate)
                except ValueError:
                    pass

                fingerprint = candidate.fingerprint
                self._pending_fingerprints.discard(fingerprint)
                self._inflight_fingerprints.add(fingerprint)

                # Launch the evaluation
                launched_successfully = await self._stream_launch(candidate, window_id)

                if launched_successfully:
                    # Subtract from deficit (rung got its turn)
                    self._rung_deficit[best_rung_key] -= 1.0
                    launched += 1
                    continue
                else:
                    # Launch failed, put candidate back
                    self._inflight_fingerprints.discard(fingerprint)
                    self._pending_fingerprints.add(fingerprint)
                    queue.appendleft(candidate)
                    if candidate not in self.queue:
                        self.queue.append(candidate)
                    break
            else:
                # Can't launch this candidate (capacity or eval limit reached)
                break

        return launched

    async def _stream_launch(self, candidate: Candidate, window_id: int) -> bool:
        from .cache import candidate_key
        import time

        if len(self._shard_capacity) == 0:
            return False

        shard_idx = min(self.scheduler.current_shard_index(candidate), len(self._shard_capacity) - 1)
        if self._shard_inflight[shard_idx] >= self._shard_capacity[shard_idx]:
            return False
        if self._total_inflight >= self._max_total_inflight:
            return False

        shard_fraction = self.scheduler.shard_fraction_for_index(shard_idx)
        shard_key = (window_id, shard_fraction)
        shard_ids = self._shard_cache.get(shard_key)
        if shard_ids is None:
            shard_size = self._shard_size(shard_fraction)
            shard_ids = self.sampler.sample_shard(window_id, shard_size)
            self._shard_cache[shard_key] = shard_ids

        start_time = time.time()
        cand_hash = candidate_key(candidate)
        self._total_inflight += 1
        self._shard_inflight[shard_idx] += 1
        # Track inflight per rung for deficit scheduler
        rung_key = self._get_rung_key(candidate)
        self._inflight_by_rung[rung_key] += 1

        # Compute fair per-candidate evaluator concurrency so total example-level
        # work stays near the global target without oversubscription.
        # Note: _total_inflight has already been incremented to include this candidate.
        active_candidates = max(1, self._total_inflight)
        per_cand_concurrency = max(1, int(self._effective_concurrency // active_candidates))
        # Never exceed the shard size for this candidate
        per_cand_concurrency = min(per_cand_concurrency, len(shard_ids))
        # Respect hard global example-level cap; if not enough budget, don't launch yet
        if self._examples_inflight + per_cand_concurrency > self._effective_concurrency:
            return False
        # Reserve budget for this candidate
        self._examples_inflight += per_cand_concurrency

        async def runner() -> None:
            try:
                result = await self.evaluator.eval_on_shard(
                    candidate,
                    shard_ids,
                    concurrency=per_cand_concurrency,
                    shard_fraction=shard_fraction,
                    show_progress=self.show_progress,  # Show progress for all evaluations
                )
                await self._result_queue.put((candidate, result, shard_idx, start_time))
            except Exception as exc:  # pragma: no cover - defensive logging path
                await self._result_queue.put((candidate, exc, shard_idx, start_time))
            finally:
                self._shard_inflight[shard_idx] -= 1
                self._total_inflight -= 1
                # Decrement rung inflight for deficit scheduler
                rung_key_inner = self._get_rung_key(candidate)
                self._inflight_by_rung[rung_key_inner] -= 1
                # Release global example-level budget
                self._examples_inflight = max(0, self._examples_inflight - per_cand_concurrency)

        task = asyncio.create_task(runner())
        self._inflight_tasks[cand_hash] = task
        task.add_done_callback(lambda _: self._inflight_tasks.pop(cand_hash, None))
        return True

    async def _stream_drain_results(self, timeout: float = 0.1) -> int:
        import asyncio as _asyncio
        import time as _time

        count = 0
        deadline = _time.time() + timeout

        while True:
            try:
                candidate, payload, shard_idx, start_time = self._result_queue.get_nowait()
            except _asyncio.QueueEmpty:
                remaining = deadline - _time.time()
                if remaining <= 0:
                    break
                try:
                    candidate, payload, shard_idx, start_time = await _asyncio.wait_for(
                        self._result_queue.get(), timeout=remaining
                    )
                except _asyncio.TimeoutError:
                    break
            await self._stream_process_result(candidate, payload, shard_idx, start_time)
            count += 1

        return count

    async def _ingest_result(self, candidate: Candidate, result: EvalResult) -> Tuple[Candidate, str]:
        """Update internal state with a freshly evaluated result."""
        from .cache import candidate_key

        shard_fraction = result.shard_fraction or 0.0
        quality = result.objectives.get(self.config.promote_objective, 0.0)

        generation_method = None
        if isinstance(candidate.meta, dict):
            generation_method = candidate.meta.get("generation_method")

        original_fingerprint = candidate.fingerprint
        meta = dict(candidate.meta)
        prev_fraction = meta.get("quality_shard_fraction", 0.0)
        prev_quality = meta.get("quality", float("-inf"))
        if shard_fraction > prev_fraction or (
            shard_fraction == prev_fraction and quality >= prev_quality
        ):
            meta["quality"] = quality
            meta["quality_shard_fraction"] = shard_fraction
        candidate_with_meta = Candidate(text=candidate.text, meta=meta)

        cand_hash = candidate_key(candidate_with_meta)
        self.latest_results[cand_hash] = result
        self.evaluations_run += 1

        decision = self.scheduler.record(
            candidate_with_meta, result, self.config.promote_objective
        )

        if decision in ("promoted", "completed"):
            await self.archive.insert(candidate_with_meta, result)
            self._record_candidate_promotion(candidate_with_meta)
            await self._maybe_share(candidate_with_meta)

        self._register_failures(result)

        promotions = self.scheduler.promote_ready()
        if promotions:
            self.enqueue(promotions)

        if generation_method and hasattr(self.mutator, "report_outcome"):
            success = decision in ("promoted", "completed")
            self.mutator.report_outcome(generation_method, success)

        # Clear both the original fingerprint (added before launch) and the updated one
        # (metadata changes can alter the fingerprint, so be defensive).
        self._inflight_fingerprints.discard(original_fingerprint)
        if candidate_with_meta.fingerprint != original_fingerprint:
            self._inflight_fingerprints.discard(candidate_with_meta.fingerprint)

        return candidate_with_meta, decision

    async def _maybe_share(self, candidate: Candidate) -> None:
        if not self.island_context:
            return
        import time
        now = time.time()
        shard_fraction = candidate.meta.get("quality_shard_fraction", 0.0)
        key = (candidate.fingerprint, shard_fraction)
        last = self._last_shared.get(key)
        if last is not None and now - last < 10.0:
            return
        elites = [candidate]
        migrate_out(self.island_context, elites)
        if self.show_progress:
            self.logger.log(
                f"🌍 Island {self.island_id}: promoting candidate {candidate.fingerprint[:8]} on shard {shard_fraction:.2f}"
            )
        self._last_shared[key] = now
        incoming = integrate_in(self.island_context)
        if incoming:
            if self.show_progress:
                ids = ", ".join(c.fingerprint[:8] for c in incoming)
                self.logger.log(f"🌐 Island {self.island_id}: received {len(incoming)} candidate(s): {ids}")
            self.enqueue(incoming)

    async def _stream_process_result(
        self,
        candidate: Candidate,
        payload: EvalResult | Exception,
        shard_idx: int,
        start_time: float,
    ) -> None:
        if isinstance(payload, Exception):
            self._inflight_fingerprints.discard(candidate.fingerprint)
            return

        result: EvalResult = payload
        _ = start_time  # placeholder for potential latency calculations
        await self._ingest_result(candidate, result)

    async def _stream_check_target(self, window_id: int) -> bool:
        if self.config.target_quality is None:
            return False

        pareto = self.archive.pareto_entries()
        if not pareto:
            return False

        full_shard = self.config.shards[-1]
        full_entries = [entry for entry in pareto if entry.result.shard_fraction == full_shard]
        if not full_entries:
            return False

        best_quality = max(
            entry.result.objectives.get(self.config.promote_objective, 0.0)
            for entry in full_entries
        )
        if best_quality < self.config.target_quality:
            return False

        if self.show_progress:
            self.logger.log(
                f"🎯 Target quality {self.config.target_quality:.2%} reached! Achieved: {best_quality:.2%}"
            )
            self.logger.log(f"   (Verified on full dataset: {full_shard:.0%} of examples)")
        return True

    # === Streaming Launch Helpers ===

    async def _seed_archive(self, seeds: Sequence[Candidate]) -> None:
        shard_fraction = self.config.shards[0]
        shard_size = self._shard_size(shard_fraction)
        shard = self.sampler.sample_shard(self.round_index, shard_size)

        if self.show_progress:
            self.logger.log(
                f"🌱 Evaluating {len(seeds)} seed prompt(s) on {shard_size} examples ({shard_fraction:.0%} of dataset)..."
            )
            self.logger.log("   This establishes the baseline for optimization.")

        results = await asyncio.gather(
            *[
                self._evaluate_candidate(seed, shard, shard_fraction, show_progress=self.show_progress)
                for seed in seeds
            ]
        )

        if self.show_progress:
            self.logger.log("   ✓ Seed evaluation complete!")

        enriched: List[Candidate] = []
        for seed, result in zip(seeds, results):
            candidate_with_meta, _ = await self._ingest_result(seed, result)
            enriched.append(candidate_with_meta)
            # Mark seeds as generation 0
            self._candidate_generations[candidate_with_meta.fingerprint] = 0

        self.enqueue(enriched)

    def _select_batch(self) -> List[Candidate]:
        batch: List[Candidate] = []
        seen: set[str] = set()
        while self.queue and len(batch) < self.config.batch_size:
            candidate = self.queue.popleft()
            if candidate.fingerprint in seen:
                continue
            batch.append(candidate)
            seen.add(candidate.fingerprint)

        if len(batch) < self.config.batch_size:
            needed = self.config.batch_size - len(batch)
            exploit = (needed + 1) // 2
            explore = needed // 2
            archive_candidates = self.archive.select_for_generation(
                exploit,
                explore,
                objective=self.config.promote_objective,
            )
            for candidate in archive_candidates:
                if len(batch) >= self.config.batch_size:
                    break
                if candidate.fingerprint in seen:
                    continue
                batch.append(candidate)
                seen.add(candidate.fingerprint)
        return batch

    def _get_best_quality_from_full_shard(self) -> float:
        """Get best quality from candidates evaluated on the longest/full shard only.

        This ensures displayed quality represents performance on the full dataset,
        not partial shards which may give inflated scores due to overfitting.

        Falls back to best quality from ANY shard if no full-shard evaluations
        exist yet (e.g., during seed baseline evaluation).
        """
        pareto = self.archive.pareto_entries()
        if not pareto:
            return 0.0

        full_shard = self.config.shards[-1]  # Longest shard (e.g., 1.0 = 100%)

        # Try to get quality from full-shard evaluations first
        full_shard_quality = 0.0
        has_full_shard = False
        for entry in pareto:
            if entry.result.shard_fraction == full_shard:
                has_full_shard = True
                quality = entry.result.objectives.get(self.config.promote_objective, 0.0)
                full_shard_quality = max(full_shard_quality, quality)

        if has_full_shard:
            return full_shard_quality

        # Fallback: Return best quality from ANY shard
        # This ensures we always show something meaningful during early optimization
        best_quality = max(
            entry.result.objectives.get(self.config.promote_objective, 0.0)
            for entry in pareto
        )
        return best_quality

    async def _spawn_mutations(self, callback: Optional[Callable[[Candidate], None]] = None) -> None:
        """Generate mutations using batched reflection for efficiency."""
        # Check if we've exceeded the evaluation budget before spawning mutations
        if self.max_evaluations is not None and self.evaluations_run >= self.max_evaluations:
            return

        entries = self.archive.pareto_entries()
        if not entries:
            self.logger.log("ℹ️  spawn_mutations: archive empty, skipping.")
            return

        num_mutations = self._mutation_budget()
        if num_mutations <= 0:
            self.logger.log(
                f"ℹ️  spawn_mutations: budget={num_mutations}, queue={len(self.queue)}, buffer={len(self._mutation_buffer)} — skipping."
            )
            return

        self.logger.log(
            f"🧭 spawn_mutations: parents={len(entries)} budget={num_mutations} "
            f"queue={len(self.queue)} buffer={len(self._mutation_buffer)}"
        )
        mutations = await self._generate_mutations_batched(entries, num_mutations)
        if not mutations:
            self.logger.log("⚠️  spawn_mutations: runner returned no candidates.")
            return

        self._record_mutation_enqueued(len(mutations))
        if self.show_progress:
            self.logger.log(
                f"   ↳ generated {len(mutations)} mutation(s) "
                f"(total generated={self._mutations_generated}, promoted={len(self._promoted_children)})"
            )
        if callback:
            for candidate in mutations:
                callback(candidate)
        else:
            self.enqueue(mutations)

    async def _generate_mutations_batched(
        self,
        entries: List[ArchiveEntry],
        num_mutations: int,
    ) -> List[Candidate]:
        """Generate mutations using batched reflection."""
        import time
        batch_start = time.time()

        # Smart parent selection: mix of top quality + QD diversity
        # Select 3-5 parents to give reflection LLM rich context
        parent_select_start = time.time()
        num_parents = min(5, max(3, len(entries)))

        # Get top performers by quality
        sorted_entries = sorted(
            entries,
            key=lambda e: e.result.objectives.get(self.config.promote_objective, float("-inf")),
            reverse=True,
        )
        top_quality = sorted_entries[: num_parents // 2 + 1]  # At least half are top quality

        # Add diversity from QD grid
        qd_diverse = self.archive.sample_qd(num_parents - len(top_quality))

        # Combine (deduplicate by fingerprint)
        selected_fingerprints = {e.candidate.fingerprint for e in top_quality}
        selected_entries = list(top_quality)

        for candidate in qd_diverse:
            if candidate.fingerprint not in selected_fingerprints:
                # Find the entry for this candidate
                entry = next((e for e in entries if e.candidate.fingerprint == candidate.fingerprint), None)
                if entry:
                    selected_entries.append(entry)
                    selected_fingerprints.add(candidate.fingerprint)
                if len(selected_entries) >= num_parents:
                    break

        parent_select_time = time.time() - parent_select_start

        # Use cached evaluation results instead of fresh eval (saves 3-5s per round!)
        # We already have traces from the ASHA evaluation that just ran
        reflection_minibatch_size = 5
        reflection_examples: List[Dict[str, object]] = []

        # Sample task examples for spec induction
        task_examples = self._sample_task_examples_for_spec_induction(num_examples=reflection_minibatch_size)

        cache_start = time.time()
        if selected_entries:
            # Get cached traces from best parent's most recent evaluation
            best_parent_fingerprint = selected_entries[0].candidate.fingerprint
            cached_result = self.latest_results.get(best_parent_fingerprint)

            if cached_result and cached_result.traces:
                # Use cached traces from recent ASHA evaluation
                # OG GEPA passes ALL examples (successes + failures), not just failures
                # This shows reflection LLM both what works and what needs fixing
                all_traces = list(cached_result.traces)

                # Prioritize failures (more informative), but include some successes too
                failure_traces = [t for t in all_traces if t.get("quality", 0.0) < 1.0]
                success_traces = [t for t in all_traces if t.get("quality", 0.0) >= 1.0]

                # Take mostly failures (4) + some successes (1) if available
                selected_traces = failure_traces[:4] + success_traces[:1]
                selected_traces = selected_traces[:reflection_minibatch_size]

                for trace in selected_traces:
                    example_input = trace.get("input", "")
                    assistant_output = trace.get("response", "") or trace.get("output", "")
                    expected_answer = trace.get("expected_answer", "")
                    additional_context = trace.get("additional_context")
                    quality = trace.get("quality", 0.0)

                    if quality >= 1.0:
                        feedback = f"The generated response is correct. The response includes the correct answer '{expected_answer}'"
                    else:
                        feedback_parts = [f"The generated response is incorrect. The correct answer is '{expected_answer}'."]
                        feedback_parts.append("Ensure that the correct answer is included in the response exactly as it is.")
                        if additional_context and isinstance(additional_context, dict):
                            context_lines = [f"{k}: {v}" for k, v in additional_context.items()]
                            if context_lines:
                                feedback_parts.append(f"Here is some additional context that might be helpful:\n{chr(10).join(context_lines)}")
                        feedback = " ".join(feedback_parts)

                    reflection_examples.append({
                        "input": example_input,
                        "assistant_output": assistant_output,
                        "expected_answer": expected_answer,
                        "feedback": feedback,
                        "additional_context": additional_context,
                    })

        cache_time = time.time() - cache_start

        # Build parent contexts for mutation (all selected parents)
        # IMPORTANT: Use latest_results instead of entry.result to get the most recent
        # (and typically full-dataset) evaluation, not the archived partial-shard result
        parent_contexts: List[Dict[str, object]] = []
        for entry in selected_entries:
            # Prefer latest_results (most recent eval, usually full dataset)
            # Fall back to entry.result if not found in latest_results
            latest_result = self.latest_results.get(entry.candidate.fingerprint)
            parent_objectives = latest_result.objectives if latest_result else entry.result.objectives

            candidate_with_parent = entry.candidate.with_meta(
                parent_objectives=parent_objectives,
            )
            parent_contexts.append({
                "candidate": candidate_with_parent,
                "failures": [],  # Not used anymore, we have fresh reflection_examples
            })

        if not parent_contexts:
            return []

        # Set reflection examples for the mutator
        if hasattr(self.mutator, "set_reflection_examples"):
            if reflection_examples:
                self.mutator.set_reflection_examples(reflection_examples)
            elif task_examples:
                # If fresh eval failed, use task examples as fallback
                self.mutator.set_reflection_examples(task_examples)

        mutator_start = time.time()
        mutations = await self.mutator.propose(parent_contexts, num_mutations, task_examples=task_examples)
        mutator_time = time.time() - mutator_start
        self._record_mutation_generation(num_mutations, mutations)

        batch_total_time = time.time() - batch_start

        return mutations

    def _record_mutation_generation(self, requested: int, mutations: Sequence[Candidate]) -> None:
        if requested > 0:
            self._mutations_requested += requested
        generated = len(mutations)
        self._mutations_generated += generated
        for candidate in mutations:
            fingerprint = candidate.fingerprint
            self._children_seen.add(fingerprint)
            parent_fp: Optional[str] = None
            if isinstance(candidate.meta, dict):
                parent_fp = candidate.meta.get("parent")
            if parent_fp:
                self._parent_children[parent_fp].add(fingerprint)
                # Track generation: parent's generation + 1
                parent_gen = self._candidate_generations.get(parent_fp, 0)
                self._candidate_generations[fingerprint] = parent_gen + 1
            else:
                # No parent info, assume generation 1 (first mutation)
                self._candidate_generations[fingerprint] = self._candidate_generations.get(fingerprint, 1)

    def _record_mutation_enqueued(self, count: int) -> None:
        if count <= 0:
            return
        self._mutations_enqueued += count

    def _record_candidate_promotion(self, candidate: Candidate) -> None:
        if not isinstance(candidate.meta, dict):
            return
        parent_fp = candidate.meta.get("parent")
        if not parent_fp:
            return
        child_fp = candidate.fingerprint
        self._children_seen.add(child_fp)
        self._parent_children[parent_fp].add(child_fp)
        # Track generation if not already tracked
        if child_fp not in self._candidate_generations:
            parent_gen = self._candidate_generations.get(parent_fp, 0)
            self._candidate_generations[child_fp] = parent_gen + 1
        if child_fp in self.archive.pareto and child_fp not in self._promoted_children:
            self._promoted_children.add(child_fp)

    def evolution_snapshot(self, include_edges: bool = False) -> Dict[str, Any]:
        edges = sum(len(children) for children in self._parent_children.values())
        snapshot: Dict[str, Any] = {
            "mutations_requested": self._mutations_requested,
            "mutations_generated": self._mutations_generated,
            "mutations_enqueued": self._mutations_enqueued,
            "mutations_promoted": len(self._promoted_children),
            "unique_parents": len(self._parent_children),
            "unique_children": len(self._children_seen),
            "evolution_edges": edges,
        }
        if include_edges:
            snapshot["parent_children"] = {
                parent: list(children) for parent, children in self._parent_children.items()
            }
            snapshot["children"] = list(self._children_seen)
            snapshot["promoted_children"] = list(self._promoted_children)
        return snapshot

    def get_candidate_lineage_data(self) -> List[Dict[str, Any]]:
        """Return list of all candidates with generation, quality, and status for visualization."""
        from typing import List, Dict, Any

        data: List[Dict[str, Any]] = []
        promote_objective = self.config.promote_objective

        # Get all pareto candidates
        for entry in self.archive.pareto_entries():
            fp = entry.candidate.fingerprint
            data.append({
                "fingerprint": fp,
                "generation": self._candidate_generations.get(fp, 0),
                "quality": entry.result.objectives.get(promote_objective, 0.0),
                "status": "promoted",
            })

        # Get all QD grid candidates
        for entry in self.archive.qd_grid.values():
            fp = entry.candidate.fingerprint
            # Skip if already in pareto
            if fp in self.archive.pareto:
                continue
            data.append({
                "fingerprint": fp,
                "generation": self._candidate_generations.get(fp, 0),
                "quality": entry.result.objectives.get(promote_objective, 0.0),
                "status": "evaluated",
            })

        # Get in-flight candidates (in queue)
        for candidate in self.queue:
            fp = candidate.fingerprint
            # Get quality from archive if available
            quality = 0.0
            if fp in self.archive.pareto:
                quality = self.archive.pareto[fp].result.objectives.get(promote_objective, 0.0)

            data.append({
                "fingerprint": fp,
                "generation": self._candidate_generations.get(fp, 0),
                "quality": quality,
                "status": "in_flight",
            })

        return data


    def _mutation_budget(self) -> int:
        """Compute how many new mutations we should request this cycle."""
        max_mut = self.config.max_mutations_per_round
        if max_mut <= 0:
            return 0

        ready = len(self.queue) + len(self._mutation_buffer)
        target_ready = max(self._max_total_inflight, self.config.batch_size)
        deficit = target_ready - ready
        if deficit <= 0:
            import time as _time_budget
            now = _time_budget.time()
            if now - self._last_budget_log > 5.0:
                self.logger.log(
                    f"🧭 Debug: mutation budget 0 "
                    f"(ready={ready} target_ready={target_ready} "
                    f"queue={len(self.queue)} buffer={len(self._mutation_buffer)})"
                )
                self._last_budget_log = now
            return 0
        return max(1, min(max_mut, deficit))

    def _sample_task_examples_for_spec_induction(self, num_examples: int = 3) -> List[Dict[str, object]]:
        """Sample a few task examples for spec induction (PROMPT-MII style)."""
        if self.example_sampler:
            # Use adapter-provided sampler (has access to actual example data)
            return self.example_sampler(num_examples)
        else:
            # Fallback: return empty list (spec induction won't run)
            return []

    async def _maybe_migrate(self) -> None:
        if not self.island_context:
            return
        if self.eval_batches_completed == 0:
            return
        # Trigger migration based on evaluation batches completed (streaming mode)
        if self.eval_batches_completed % self.config.migration_period != 0:
            return
        elites = self.archive.select_for_generation(
            self.config.migration_k,
            0,
            objective=self.config.promote_objective,
        )
        if elites:
            if self.show_progress:
                ids = ", ".join(candidate.fingerprint[:8] for candidate in elites)
                self.logger.log(f"🌍 Island {self.island_id}: migrating out {len(elites)} candidate(s): {ids}")
            migrate_out(self.island_context, elites)

        pareto_entries = self.archive.pareto_entries()
        if pareto_entries:
            share_count = min(self.config.migration_k or 1, len(pareto_entries))
            top_candidates = sorted(
                pareto_entries,
                key=lambda entry: entry.result.objectives.get(self.config.promote_objective, float("-inf")),
                reverse=True,
            )[:share_count]
            shared = [entry.candidate for entry in top_candidates]
            if shared:
                self.enqueue(shared)

        incoming = integrate_in(self.island_context)
        if incoming:
            if self.show_progress:
                ids = ", ".join(candidate.fingerprint[:8] for candidate in incoming)
                self.logger.log(f"🌐 Island {self.island_id}: received {len(incoming)} candidate(s): {ids}")
            self.enqueue(incoming)

    async def _evaluate_candidate(
        self,
        candidate: Candidate,
        shard: Sequence[str],
        shard_fraction: float,
        show_progress: bool = False,
    ) -> EvalResult:
        result = await self.evaluator.eval_on_shard(
            candidate,
            shard,
            concurrency=self._effective_concurrency,
            shard_fraction=shard_fraction,
            show_progress=show_progress,
        )
        return result

    def _register_failures(self, result: EvalResult) -> None:
        hard_examples: List[str] = []
        for trace in result.traces:
            example_id = trace.get("example_id") if isinstance(trace, dict) else None
            quality = trace.get("quality") if isinstance(trace, dict) else None
            if example_id is None or quality is None:
                continue
            if quality < result.objectives.get("quality", quality):
                hard_examples.append(example_id)
        if hard_examples:
            self.sampler.register_hard_examples(hard_examples)

    def _shard_size(self, shard_fraction: float) -> int:
        total = max(len(self.sampler.example_ids), 1)
        size = max(1, int(total * shard_fraction))
        return min(size, total)

    def _check_stop_governor(self) -> tuple[bool, Dict]:
        """Update stop governor with current metrics and check if we should stop."""
        if self.stop_governor is None:
            return False, {}

        # Collect current epoch metrics
        pareto_entries = self.archive.pareto_entries()

        if not pareto_entries:
            return False, {"reason": "no_pareto_candidates"}

        # Compute hypervolume (use reference point below all possible values)
        points = [
            (
                entry.result.objectives.get(self.config.promote_objective, 0.0),
                entry.result.objectives.get("neg_cost", -entry.result.objectives.get("tokens", 1000)),
            )
            for entry in pareto_entries
        ]
        # Reference point: (0 quality, -max_tokens) to ensure all points dominate it
        hypervolume = compute_hypervolume_2d(points, reference=(0.0, -self.config.max_tokens))

        # Find best candidate
        best_entry = max(
            pareto_entries,
            key=lambda e: (
                e.result.objectives.get(self.config.promote_objective, 0.0),
                e.result.objectives.get("neg_cost", float('-inf')),
            ),
        )

        # Track QD novelty
        qd_entries = list(self.archive.qd_grid.values())
        current_cells = set()
        for entry in qd_entries:
            # Create a hashable cell identifier from QD features
            cell_id = (
                int(entry.result.objectives.get("quality", 0) * 100),  # Discretize
                len(entry.candidate.text) // 100,  # Length bins
            )
            current_cells.add(cell_id)

        new_cells = current_cells - self.qd_cells_seen
        qd_novelty_rate = len(new_cells) / max(len(current_cells), 1)
        self.qd_cells_seen.update(new_cells)

        # Update token spending
        tokens_this_round = sum(
            self.latest_results.get(e.candidate.fingerprint, e.result).objectives.get("tokens", 0)
            for e in pareto_entries
        )
        self.total_tokens_spent += int(tokens_this_round)

        # Create epoch metrics
        metrics = EpochMetrics(
            round_num=self.round_index,
            hypervolume=hypervolume,
            new_evaluations=self.evaluations_run,
            best_quality=best_entry.result.objectives.get(self.config.promote_objective, 0.0),
            best_cost=best_entry.result.objectives.get("neg_cost", float('-inf')),
            frontier_ids={e.candidate.fingerprint for e in pareto_entries},
            qd_filled_cells=len(current_cells),
            qd_total_cells=self.config.qd_bins_length * self.config.qd_bins_bullets * len(self.config.qd_flags),
            qd_novelty_rate=qd_novelty_rate,
            total_tokens_spent=self.total_tokens_spent,
            shadow_significant=True,  # Shadow shard testing not implemented (optional v2 feature)
        )

        self.stop_governor.update(metrics)
        return self.stop_governor.should_stop()

    async def finalize(self, delta: Optional[float] = None) -> None:
        """Finalize the run before returning results."""
        if hasattr(self, "_save_task") and not self._save_task.done():
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

    # State persistence for resumable optimization

    async def _save_state(self) -> None:
        """Save current orchestrator state to cache for resumability (non-blocking)."""
        # Cancel any previous save that's still running to avoid queue buildup
        if hasattr(self, '_save_task') and not self._save_task.done():
            self._save_task.cancel()

        pareto_candidates = self.archive.pareto_candidates()
        qd_candidates = self.archive.sample_qd(limit=len(self.archive.qd_grid))

        # Run save in background - don't block the optimization loop
        self._save_task = asyncio.create_task(
            asyncio.to_thread(
                self.cache.save_state,
                round_num=self.round_index,
                evaluations=self.evaluations_run,
                pareto_candidates=pareto_candidates,
                qd_candidates=qd_candidates,
                queue=list(self.queue),
            )
        )

    async def _restore_state(self, state: Dict) -> None:
        """Restore orchestrator state from saved checkpoint."""
        self.round_index = state["round"]
        self.evaluations_run = state["evaluations"]

        # Restore archive by re-inserting candidates with concurrency limit
        # to avoid "too many open files" errors when reading from cache
        # Note: We don't have the full EvalResult objects, so we'll need to
        # re-evaluate them (but cache will make this instant)

        # Limit concurrent restorations to avoid file descriptor exhaustion
        # Use a conservative limit since each restoration may open multiple files
        restore_semaphore = asyncio.Semaphore(5)

        async def restore_candidate(candidate: Candidate) -> None:
            async with restore_semaphore:
                # Re-evaluate to get full result (will hit cache)
                shard = self.sampler.sample_shard(self.round_index, len(self.sampler.example_ids))
                result = await self.evaluator.eval_on_shard(
                    candidate,
                    shard,
                    concurrency=self._effective_concurrency,
                    shard_fraction=1.0,
                )
                await self.archive.insert(candidate, result)

        # Restore all candidates with controlled parallelism
        all_candidates = state["pareto"] + state["qd"]
        await asyncio.gather(*(restore_candidate(c) for c in all_candidates))

        # Restore queue
        self.queue = deque(state["queue"], maxlen=self.config.queue_limit)
