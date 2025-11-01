"""
High-level orchestrator coordinating selection, evaluation, and mutation.

This module stitches together archive, scheduler, evaluator, and mutator
components. The orchestrator operates within a single island; multi-island
concurrency is achieved via ``islands.spawn_islands`` using asyncio tasks
in the same process (async ring topology), not separate OS processes.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict, deque
from typing import Any, Callable, Iterable, Sequence

from turbo_gepa.logging.logger import LogLevel, LoggerProtocol, StdOutLogger

from .archive import Archive, ArchiveEntry
from .cache import DiskCache, candidate_key
from .config import Config
from .evaluator import AsyncEvaluator
from .interfaces import Candidate, EvalResult
from .islands import IslandContext, integrate_in, migrate_out
from .metrics import Metrics
from .mutator import Mutator
from .sampler import InstanceSampler
from .scheduler import BudgetedScheduler, SchedulerConfig
from .stop_governor import EpochMetrics, StopGovernor, compute_hypervolume_2d


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
        island_context: IslandContext | None = None,
        show_progress: bool = True,
        stop_governor: StopGovernor | None = None,
        enable_auto_stop: bool = False,
        example_sampler: Callable[[int], list[dict[str, object]]] | None = None,
        metrics_callback: Callable | None = None,
        logger: LoggerProtocol | None = None,
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
                enable_convergence=config.enable_rung_convergence,
                lineage_patience=config.lineage_patience,
                lineage_min_improve=config.lineage_min_improve,
            )
        )
        self._runtime_shards: list[float] = list(self.config.shards)
        self.queue: deque[Candidate] = deque(maxlen=config.queue_limit)
        self._per_shard_queue: list[deque[Candidate]] = [deque() for _ in range(len(self._runtime_shards))]
        self._pending_fingerprints: set[str] = set()
        self._inflight_fingerprints: set[str] = set()
        self._next_shard: int = 0
        self.latest_results: dict[str, EvalResult] = {}
        self.evaluations_run: int = 0
        self.round_index: int = 0
        self.show_progress = show_progress
        island_id = island_context.island_id if island_context else 0
        self.island_id = island_id
        self.stop_governor = stop_governor if enable_auto_stop else None
        self.enable_auto_stop = enable_auto_stop
        self.total_tokens_spent: int = 0
        self.qd_cells_seen: set[tuple] = set()
        self._shard_cache: dict[tuple[int, float], list[str]] = {}
        self.metrics_callback = metrics_callback
        self.logger: LoggerProtocol = logger or StdOutLogger()
        self.max_rounds: int | None = None
        self.max_evaluations: int | None = None
        self.rounds_completed: int = 0

        # Streaming mode: buffer for mutations generated in background
        self._mutation_buffer: deque[Candidate] = deque()
        self._mutation_task: asyncio.Task | None = None
        self.eval_batches_completed: int = 0  # For migration timing

        # Streaming evaluation infrastructure
        self._inflight_tasks: dict[str, asyncio.Task] = {}  # cand_hash -> task
        self._result_queue: asyncio.Queue[tuple[Candidate, EvalResult | Exception, int]] = asyncio.Queue()
        self._total_inflight: int = 0  # Total evaluations in flight across all shards
        total_cap = self.config.max_total_inflight or self.config.eval_concurrency
        self._max_total_inflight: int = max(1, total_cap)
        self._examples_inflight: int = 0
        self._launch_start_times: dict[str, float] = {}
        self._latency_ema: float = 0.0
        self._latency_samples: int = 0
        self._eval_samples: int = 0
        self._timeout_count: int = 0
        self._mutation_throttle: bool = False
        base_mut = self.config.max_mutations_per_round or max(16, self.config.eval_concurrency // 2)
        self._max_mutations_ceiling = base_mut
        self._mutation_min = max(4, base_mut // 4)
        self._rung_launches: list[int] = [0] * len(self._runtime_shards)
        self._rung_promotions: list[int] = [0] * len(self._runtime_shards)
        self._promotion_ema: list[float] = [0.0] * len(self._runtime_shards)
        self._last_debug_log: float = 0.0
        self._last_budget_log: float = 0.0
        self._last_shared: dict[str, float] = {}
        # Evolution tracking
        self._mutations_requested: int = 0
        self._mutations_generated: int = 0
        self._mutations_enqueued: int = 0
        self._parent_children: defaultdict[str, set[str]] = defaultdict(set)
        self._children_seen: set[str] = set()
        self._promoted_children: set[str] = set()
        # Track generation depth for each candidate (fingerprint -> generation number)
        self._candidate_generations: dict[str, int] = {}
        self._lineage_history: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=8))
        self._sched_to_fingerprint: dict[str, str] = {}
        self._promotion_pending: set[str] = set()
        self._last_mutation_attempt: float = 0.0

        # Metrics tracking
        self.metrics = Metrics()
        # Pass metrics to evaluator for cache tracking
        self.evaluator.metrics = self.metrics

        # Scheduler state derived from capacities
        self._recompute_capacities()

    def _resize_metric_list(self, data: list[int], target: int) -> list[int]:
        if len(data) < target:
            data.extend([0] * (target - len(data)))
        elif len(data) > target:
            data = data[:target]
        return data

    def _resize_float_list(self, data: list[float], target: int) -> list[float]:
        if len(data) < target:
            data.extend([0.0] * (target - len(data)))
        elif len(data) > target:
            data = data[:target]
        return data

    def _rebalance_shard_queue(self, target: int) -> None:
        while len(self._per_shard_queue) < target:
            self._per_shard_queue.append(deque())
        while len(self._per_shard_queue) > target:
            extra = self._per_shard_queue.pop()
            if extra and self._per_shard_queue:
                self._per_shard_queue[-1].extend(extra)

    def _recompute_capacities(self) -> None:
        num_shards = max(1, len(self._runtime_shards))
        self._rebalance_shard_queue(num_shards)
        self._effective_concurrency = max(1, min(self._max_total_inflight, self.config.eval_concurrency))
        base = max(1, self._effective_concurrency // num_shards)
        remainder = max(0, self._effective_concurrency - base * num_shards)
        self._shard_capacity = [base + (1 if i < remainder else 0) for i in range(num_shards)]
        if sum(self._shard_capacity) < self._effective_concurrency:
            deficit = self._effective_concurrency - sum(self._shard_capacity)
            for i in range(deficit):
                self._shard_capacity[i % num_shards] += 1
        current_inflight = list(getattr(self, "_shard_inflight", []))
        current_inflight = current_inflight[:num_shards] + [0] * max(0, num_shards - len(current_inflight))
        self._shard_inflight = current_inflight
        self._rung_keys = [str(shard) for shard in self._runtime_shards]
        n_rungs = len(self._rung_keys)
        self._rung_shares = dict.fromkeys(self._rung_keys, 1.0 / max(n_rungs, 1))
        self._rung_deficit = dict.fromkeys(self._rung_keys, 0.0)
        # Preserve existing inflight counts when recomputing capacities
        # This prevents negative counts when candidates finish evaluation after shards changed
        old_inflight = getattr(self, "_inflight_by_rung", {})
        self._inflight_by_rung = {key: old_inflight.get(key, 0) for key in self._rung_keys}
        # Also preserve any old keys with non-zero counts (candidates still in-flight on old shards)
        for old_key, old_count in old_inflight.items():
            if old_key not in self._inflight_by_rung and old_count != 0:
                self._inflight_by_rung[old_key] = old_count
        self._rung_capacity = {
            key: max(1, math.ceil(self._rung_shares[key] * self._effective_concurrency)) for key in self._rung_keys
        }
        self._rung_launches = self._resize_metric_list(self._rung_launches, num_shards)
        self._rung_promotions = self._resize_metric_list(self._rung_promotions, num_shards)
        self._promotion_ema = self._resize_float_list(self._promotion_ema, num_shards)

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
        shard_fraction = self._runtime_shards[min(idx, len(self._runtime_shards) - 1)]
        return str(shard_fraction)

    def _update_eval_metrics(self, candidate: Candidate, result: EvalResult) -> None:
        cand_hash = candidate_key(candidate)
        start = self._launch_start_times.pop(cand_hash, None)
        duration = 0.0
        now = time.time()
        if start is not None:
            duration = max(0.0, now - start)
        if duration > 0:
            if self._latency_samples == 0:
                self._latency_ema = duration
            else:
                alpha = 0.2
                self._latency_ema = (1 - alpha) * self._latency_ema + alpha * duration
            self._latency_samples += 1
        timed_out = False
        for trace in result.traces:
            if isinstance(trace, dict) and trace.get("error") == "timeout":
                timed_out = True
                break
        self._eval_samples += 1
        if timed_out:
            self._timeout_count += 1

    def _current_timeout_ratio(self) -> float:
        return self._timeout_count / max(1, self._eval_samples)

    def _maybe_adjust_shards(self) -> None:
        if not getattr(self.config, "adaptive_shards_enabled", True):
            self._rung_launches = [0] * len(self._rung_launches)
            self._rung_promotions = [0] * len(self._rung_promotions)
            self._promotion_ema = [0.0] * len(self._promotion_ema)
            return

        num_shards = len(self._runtime_shards)
        if num_shards < 2:
            self._rung_launches = [0] * len(self._rung_launches)
            self._rung_promotions = [0] * len(self._rung_promotions)
            self._promotion_ema = [0.0] * len(self._promotion_ema)
            return

        new_shards = list(self._runtime_shards)
        adjusted = False
        target_rate = 0.5
        tolerance = 0.1  # Allow ±10% window before adjusting

        for idx in range(num_shards - 1):
            launches = self._rung_launches[idx] if idx < len(self._rung_launches) else 0
            if launches < 5:
                continue
            promotions = self._rung_promotions[idx] if idx < len(self._rung_promotions) else 0
            rate = promotions / max(1, launches)
            ema = self._promotion_ema[idx]
            if ema <= 0:
                ema = rate
            else:
                ema = 0.5 * ema + 0.5 * rate
            self._promotion_ema[idx] = ema

            if abs(ema - target_rate) <= tolerance:
                continue

            shrink = ema > target_rate + tolerance
            grow = ema < target_rate - tolerance
            if not shrink and not grow:
                continue

            error_fraction = min(0.25, abs(ema - target_rate))  # Cap to avoid huge jumps
            step = 0.12 + error_fraction * 0.4  # Base 12%, larger if far from target
            if shrink:
                scale = max(0.5, 1.0 - step)
            else:
                scale = min(1.5, 1.0 + step)
            new_value = new_shards[idx] * scale

            lower_bound = 0.05 if idx == 0 else new_shards[idx - 1] + 0.03
            if idx == num_shards - 2:
                upper_bound = new_shards[-1] - 0.03
            else:
                upper_bound = 0.97
            new_value = max(lower_bound, min(upper_bound, new_value))
            if abs(new_value - new_shards[idx]) > 1e-4:
                new_shards[idx] = new_value
                adjusted = True

        if adjusted:
            # Ensure strictly increasing order and cap at 1.0 for the last shard
            for i in range(1, num_shards - 1):
                min_allowed = new_shards[i - 1] + 0.03
                if new_shards[i] <= min_allowed:
                    new_shards[i] = min_allowed
            for i in range(num_shards - 2, -1, -1):
                max_allowed = new_shards[i + 1] - 0.03 if i < num_shards - 2 else new_shards[-1] - 0.03
                if new_shards[i] >= max_allowed:
                    new_shards[i] = max(0.05, max_allowed)
            new_shards[-1] = 1.0
            if any(abs(a - b) > 1e-4 for a, b in zip(new_shards, self._runtime_shards)):
                if self.show_progress:
                    before = ", ".join(f"{s:.2f}" for s in self._runtime_shards)
                    after = ", ".join(f"{s:.2f}" for s in new_shards)
                    self.logger.log(f"🔧 Adaptive shards: [{before}] → [{after}]")
                self._runtime_shards = new_shards
                self.config.shards = tuple(new_shards)
                self.scheduler.update_shards(new_shards)
                self._recompute_capacities()

        self._rung_launches = [0] * len(self._runtime_shards)
        self._rung_promotions = [0] * len(self._runtime_shards)

    def _adjust_runtime_parameters(self) -> None:
        if self._latency_samples < 5:
            return
        timeout_ratio = self._current_timeout_ratio()
        high_latency = 5.0
        low_latency = 1.5
        timeout_high = 0.05
        timeout_low = 0.01
        adjusted = False
        if self._latency_ema > high_latency or timeout_ratio > timeout_high:
            new_inflight = max(1, self._max_total_inflight - 1)
            if new_inflight != self._max_total_inflight:
                self._max_total_inflight = new_inflight
                self._recompute_capacities()
            if self.config.max_mutations_per_round:
                self.config.max_mutations_per_round = max(
                    self._mutation_min, self.config.max_mutations_per_round - 1
                )
            adjusted = True
        elif self._latency_ema < low_latency and timeout_ratio < timeout_low:
            new_inflight = min(self._max_total_inflight + 1, self.config.eval_concurrency)
            if new_inflight != self._max_total_inflight:
                self._max_total_inflight = new_inflight
                self._recompute_capacities()
            if self.config.max_mutations_per_round:
                self.config.max_mutations_per_round = min(
                    self.config.max_mutations_per_round + 1, self._max_mutations_ceiling
                )
            adjusted = True
        backlog0 = len(self._per_shard_queue[0]) if self._per_shard_queue else 0
        launches0 = self._rung_launches[0] if self._rung_launches else 0
        promotions0 = self._rung_promotions[0] if self._rung_promotions else 0
        promo_rate0 = promotions0 / max(1, launches0)
        backlog_high = max(3 * self.config.batch_size, 12)
        backlog_low = max(self.config.batch_size, 8)
        if backlog0 > backlog_high and promo_rate0 < 0.2:
            self._mutation_throttle = True
        elif backlog0 < backlog_low:
            self._mutation_throttle = False
        if adjusted:
            # Slowly decay launch counters to keep ratios reactive
            self._rung_launches = [max(0, int(count * 0.8)) for count in self._rung_launches]
            self._rung_promotions = [max(0, int(count * 0.8)) for count in self._rung_promotions]

    async def run(
        self,
        seeds: Sequence[Candidate],
        *,
        max_rounds: int | None = None,
        max_evaluations: int | None = None,
        resume: bool = True,  # Enable automatic resume from cache
    ) -> None:
        """Execute the orchestration loop until budgets are exhausted.

        Args:
            seeds: Initial candidates to start optimization
            max_rounds: Maximum number of rounds (None = unlimited)
            max_evaluations: Maximum evaluations (None = unlimited)
            resume: If True, automatically resume from saved state if available
        """

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

        # DON'T start mutation task here - archive is empty!
        # Let the main loop start it naturally after seeds are evaluated

        # Window tracking mirrors the legacy batch counter so downstream logging stays intact
        window_size = max(1, self.config.batch_size)
        self.eval_batches_completed = max(self.eval_batches_completed, self.round_index)
        window_id = self.eval_batches_completed
        window_start_evals = self.evaluations_run

        # Simple progress tracking for streaming mode
        loop_iter = 0
        last_progress_display = self.evaluations_run
        last_heartbeat = 0  # Track last heartbeat for hung detection

        # Create debug log file for diagnostics
        import time
        _debug_log_path: str | None = None
        _debug_log_file = None
        if self.config.enable_debug_log:
            import os

            _debug_log_path = f".turbo_gepa/debug_{int(time.time())}.log"
            os.makedirs(os.path.dirname(_debug_log_path), exist_ok=True)
            _debug_log_file = open(_debug_log_path, "w", buffering=1)  # Line buffered

        def _debug_log(msg: str):
            """Log to both logger and debug file."""
            if _debug_log_file is not None:
                timestamp = time.strftime("%H:%M:%S")
                full_msg = f"[{timestamp}] {msg}"
                _debug_log_file.write(full_msg + "\n")
                _debug_log_file.flush()
            if self.show_progress:
                self.logger.log(msg)

        _debug_log(f"🚀 Starting main optimization loop (max_evaluations={max_evaluations})")
        if _debug_log_path:
            _debug_log(f"   Debug log: {_debug_log_path}")

        # Track first round start
        self.metrics.start_round()

        while True:
            if max_rounds is not None and window_id >= max_rounds:
                break
            if max_evaluations is not None and self.evaluations_run >= max_evaluations:
                break

            # 1) Move freshly generated mutations into the ready queue
            while self._mutation_buffer:
                self.enqueue([self._mutation_buffer.popleft()])

            # DEBUG: Log critical state before launch attempt
            if loop_iter % 10 == 0 and len(self.queue) > 0 and self._total_inflight == 0:
                _debug_log(f"🔍 MAIN LOOP: queue={len(self.queue)}, inflight={self._total_inflight}, buffer={len(self._mutation_buffer)}")
                _debug_log(f"   Per-shard queues: {[len(q) for q in self._per_shard_queue]}")

            # DEBUG: Log before potentially blocking await
            if loop_iter % 20 == 0:
                _debug_log(f"🔄 Loop iteration {loop_iter}: About to call _stream_launch_ready")

            # 2) Launch as many evaluations as capacity allows
            launched = await self._stream_launch_ready(window_id, max_evaluations)
            if launched > 0 and loop_iter % 5 == 0:
                _debug_log(f"🚀 Launched {launched} candidate(s), total inflight: {self._total_inflight}")

            # DEBUG: Log before drain
            if loop_iter % 20 == 0:
                _debug_log(f"🔄 Loop iteration {loop_iter}: About to call _stream_drain_results")

            # 3) Drain completed results and update bookkeeping
            drained = await self._stream_drain_results()
            if drained > 0 and loop_iter % 5 == 0:
                _debug_log(f"✅ Drained {drained} result(s), total inflight: {self._total_inflight}")

            # Heartbeat: log every 30 seconds even if nothing is happening
            # This proves the system is alive and not hung
            now = time.time()
            if now - last_heartbeat >= 30.0:
                _debug_log(
                    f"💓 HEARTBEAT: Inflight={self._total_inflight}, "
                    f"Queue={len(self.queue)}, ExInflight={self._examples_inflight}, "
                    f"Evals={self.evaluations_run}, LoopIter={loop_iter}"
                )
                last_heartbeat = now

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
            # Only spawn new mutations if we haven't exceeded the evaluation budget
            should_spawn_mutations = (
                len(self.queue) + len(self._mutation_buffer)
            ) < self.config.mutation_buffer_min and (max_evaluations is None or self.evaluations_run < max_evaluations)

            # Debug mutation spawning logic
            if loop_iter % 100 == 0 and len(self.queue) == 0 and self._total_inflight == 0:
                task_status = 'None' if self._mutation_task is None else ('done' if self._mutation_task.done() else 'running')
                _debug_log(
                    f"🔍 MUTATION CHECK: queue={len(self.queue)}, buffer={len(self._mutation_buffer)}, "
                    f"min={self.config.mutation_buffer_min}, should_spawn={should_spawn_mutations}, "
                    f"task={task_status}"
                )

            # Mutation task management with spam prevention
            # Only spawn if: (1) no task exists OR (2) existing task is running
            # Never spawn right after a task completes - wait for cooldown
            if should_spawn_mutations:
                import time as _time_mut
                now = _time_mut.time()

                # If task exists and is done, clear it but record the completion time
                if self._mutation_task is not None and self._mutation_task.done():
                    _debug_log("🔄 Mutation task completed, clearing it")
                    self._mutation_task = None
                    self._last_mutation_attempt = now  # Record completion time to enforce cooldown

                # Only spawn if no task AND cooldown has passed
                elif self._mutation_task is None and (now - self._last_mutation_attempt > 1.0):
                    self._last_mutation_attempt = now
                    _debug_log(
                        f"🧬 Starting mutation generation task (queue+buffer={len(self.queue) + len(self._mutation_buffer)} < {self.config.mutation_buffer_min})"
                    )
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
                # End current round and start new one
                self.metrics.end_round()
                self.metrics.start_round()

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
                self._maybe_adjust_shards()

                if self.enable_auto_stop and self.stop_governor is not None:
                    should_stop, debug_info = self._check_stop_governor()
                    if should_stop:
                        if self.show_progress:
                            self.logger.log(f"🛑 Auto-stopping: {debug_info['reason']}")
                        break

            # 6) Target quality guard
            if await self._stream_check_target(window_id):
                break

            self._adjust_runtime_parameters()

            # 7) Idle detection - nothing running, nothing queued, mutations finished
            if (
                launched == 0
                and drained == 0
                and self._total_inflight == 0
                and not self.queue
                and not self._mutation_buffer
            ):
                if self._mutation_task is None:
                    _debug_log("🛑 IDLE DETECTION: All work complete, exiting loop")
                    break
                elif self._mutation_task.done():
                    _debug_log("🛑 IDLE DETECTION: Mutation task done, exiting loop")
                    self._mutation_task = None
                    break
                else:
                    # Mutation task still running, wait for it
                    if loop_iter % 100 == 0:
                        _debug_log(f"⏳ IDLE: Waiting for mutation task to complete (loop_iter={loop_iter})")

            # Avoid busy spinning when concurrency is saturated or nothing progressed
            if launched == 0:
                if self._examples_inflight >= self._effective_concurrency:
                    await asyncio.sleep(0.005)
                elif drained == 0:
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

        # Print and save comprehensive metrics summary
        metrics_summary = self.metrics.format_summary()
        if self.show_progress:
            self.logger.log("\n" + metrics_summary)

        # Save metrics to .turbo_gepa/metrics/ directory
        import os
        metrics_dir = ".turbo_gepa/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        timestamp = int(time.time())
        metrics_file = f"{metrics_dir}/metrics_{timestamp}.txt"
        with open(metrics_file, "w") as f:
            f.write(metrics_summary)
        if self.show_progress:
            self.logger.log(f"📊 Metrics saved to: {metrics_file}")

        # Clear state only if we reached completion (not if stopped early)
        completed = (max_rounds is not None and window_id >= max_rounds) or (
            max_evaluations is not None and self.evaluations_run >= max_evaluations
        )
        if completed:
            self.cache.clear_state()

        if _debug_log_file is not None:
            _debug_log_file.close()

    # === Streaming helpers ===

    def _stream_can_launch(self, candidate: Candidate, max_evaluations: int | None) -> bool:
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

    async def _stream_launch_ready(self, window_id: int, max_evaluations: int | None) -> int:
        """Launch ready candidates using deficit-based scheduling for fair rung allocation."""
        launched = 0
        shard_count = len(self._per_shard_queue)
        if shard_count == 0:
            return launched

        # Try launching until we can't launch anymore
        max_attempts = shard_count * 10  # Prevent infinite loops
        attempts = 0

        # Debug: Log queue state on first attempt
        if attempts == 0 and len(self.queue) > 0 and self._total_inflight == 0:
            self.logger.log(f"🔍 DEBUG _stream_launch_ready: queue={len(self.queue)} total_inflight={self._total_inflight}")
            for i, q in enumerate(self._per_shard_queue):
                if q:
                    self.logger.log(f"   Per-shard queue[{i}]: {len(q)} candidates")
                    if q:
                        cand = q[0]
                        sched_rung = self.scheduler.current_shard_index(cand)
                        self.logger.log(f"      First candidate rung: {sched_rung}, fp: {cand.fingerprint[:12]}...")

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
            best_score = float("-inf")
            best_shard_idx = None
            best_rung_key = None

            # Debug: Track evaluation of each shard
            debug_info = []

            for shard_idx in range(shard_count):
                queue = self._per_shard_queue[shard_idx]
                queue_len = len(queue) if queue else 0

                if not queue:
                    debug_info.append(f"shard[{shard_idx}]: empty")
                    continue

                # Get rung key for this shard (consistent with scheduler)
                shard_fraction = self._runtime_shards[min(shard_idx, len(self._runtime_shards) - 1)]
                rung_key = str(shard_fraction)

                # Check per-rung inflight cap before considering this rung
                inflight = self._inflight_by_rung.get(rung_key, 0)
                capacity = self._rung_capacity.get(rung_key, 0)
                global_slack = max(0, self._max_total_inflight - self._total_inflight)
                if inflight >= capacity and global_slack > 0:
                    # Borrow unused global capacity when other rungs are idle.
                    capacity += global_slack

                if inflight >= capacity:
                    debug_info.append(
                        f"shard[{shard_idx}]: queue={queue_len}, key={rung_key}, at_capacity({inflight}/{capacity})"
                    )
                    continue  # Rung at capacity, skip

                # Calculate deficit score: deficit - (inflight_fraction)
                # Higher score = more "starved" and deserves priority
                inflight_penalty = inflight / total_inflight
                deficit = self._rung_deficit.get(rung_key, 0.0)
                score = deficit - inflight_penalty

                debug_info.append(f"shard[{shard_idx}]: queue={queue_len}, key={rung_key}, score={score:.3f}, deficit={deficit:.3f}, inflight={inflight}/{capacity}")

                if score > best_score:
                    best_score = score
                    best_shard_idx = shard_idx
                    best_rung_key = rung_key

            # Log debug info if we have queued items but found no eligible rung
            if len(self.queue) > 0 and best_shard_idx is None and attempts == 1:
                self.logger.log(f"❌ DEBUG: No eligible rung found despite queue={len(self.queue)}")
                for info in debug_info:
                    self.logger.log(f"   {info}")
                self.logger.log(f"   _rung_keys: {self._rung_keys}")
                self.logger.log(f"   _rung_capacity: {self._rung_capacity}")
                self.logger.log(f"   _inflight_by_rung: {self._inflight_by_rung}")

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

        # DEBUG: Log if we couldn't launch anything despite having queue items
        if launched == 0 and len(self.queue) > 0 and attempts >= max_attempts:
            self.logger.log(f"⚠️  _stream_launch_ready: launched=0 despite queue={len(self.queue)} after {attempts} attempts")

        return launched

    async def _stream_launch(self, candidate: Candidate, window_id: int) -> bool:
        import time

        if len(self._shard_capacity) == 0:
            return False

        shard_idx = min(self.scheduler.current_shard_index(candidate), len(self._shard_capacity) - 1)
        if self._shard_inflight[shard_idx] >= self._shard_capacity[shard_idx]:
            return False
        if self._total_inflight >= self._max_total_inflight:
            return False

        shard_fraction = self._runtime_shards[min(shard_idx, len(self._runtime_shards) - 1)]
        shard_key = (window_id, shard_fraction)
        shard_ids = self._shard_cache.get(shard_key)
        if shard_ids is None:
            shard_size = self._shard_size(shard_fraction)
            shard_ids = self.sampler.sample_shard(window_id, shard_size)
            self._shard_cache[shard_key] = shard_ids

        # Compute fair per-candidate evaluator concurrency so total example-level
        # work stays near the global target without oversubscription.
        # We simulate the candidate as inflight by adding 1 to _total_inflight for the calculation.
        active_candidates = max(1, self._total_inflight + 1)
        per_cand_concurrency = max(1, int(self._effective_concurrency // active_candidates))
        # Never exceed the shard size for this candidate
        per_cand_concurrency = min(per_cand_concurrency, len(shard_ids))
        # Respect hard global example-level cap; if not enough budget, don't launch yet
        if self._examples_inflight + per_cand_concurrency > self._effective_concurrency:
            return False

        # Track peak example-level concurrency for metrics
        peak_examples = self._examples_inflight + per_cand_concurrency
        self.metrics.update_concurrent_evals(peak_examples)

        time.time()
        cand_hash = candidate_key(candidate)
        self._total_inflight += 1
        self._shard_inflight[shard_idx] += 1
        # Track inflight per rung for deficit scheduler
        # IMPORTANT: Use the shard_idx passed to this function, NOT _get_rung_key()!
        # The candidate may be promoted during evaluation, changing its shard index.
        # We must use the EXACT shard fraction this launch is evaluating on.
        rung_key_at_launch = str(shard_fraction)  # Use the actual shard fraction for this evaluation
        fingerprint_at_launch = candidate.fingerprint
        # Initialize key if not present (can happen if shards were updated mid-flight)
        if rung_key_at_launch not in self._inflight_by_rung:
            self._inflight_by_rung[rung_key_at_launch] = 0
        self._inflight_by_rung[rung_key_at_launch] += 1
        # Reserve budget for this candidate
        self._examples_inflight += per_cand_concurrency
        self._launch_start_times[cand_hash] = time.time()
        if 0 <= shard_idx < len(self._rung_launches):
            self._rung_launches[shard_idx] += 1

        async def runner() -> None:
            eval_start = time.time()
            try:
                if self.show_progress:
                    generation = self._get_generation(candidate)
                    source = "seed" if generation == 0 else f"gen-{generation if generation is not None else '?'}"
                    self.logger.log(
                        f"🔬 Evaluating {source} on shard {shard_idx} ({shard_fraction:.0%} = {len(shard_ids)} examples) "
                        f"[fp={candidate.fingerprint[:12]}... concurrency={per_cand_concurrency}]"
                    )
                result = await self.evaluator.eval_on_shard(
                    candidate,
                    shard_ids,
                    concurrency=per_cand_concurrency,
                    shard_fraction=shard_fraction,
                    show_progress=self.show_progress,  # Show progress for all evaluations
                )
                eval_duration = time.time() - eval_start
                self.metrics.record_evaluation(shard_fraction, eval_duration)
                self.metrics.time_eval_total += eval_duration
                if self.show_progress:
                    quality = result.objectives.get(self.config.promote_objective, 0.0)
                    generation = self._get_generation(candidate)
                    source = "seed" if generation == 0 else f"gen-{generation if generation is not None else '?'}"
                    self.logger.log(
                        f"✅ Completed {source} on shard {shard_idx} ({shard_fraction:.0%}): "
                        f"{quality:.1%} quality [fp={candidate.fingerprint[:12]}...]"
                    )
                await self._result_queue.put((candidate, result, shard_idx))
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.metrics.llm_errors += 1
                if self.show_progress:
                    self.logger.log(
                        f"❌ Evaluation failed for candidate {candidate.fingerprint[:12]}... "
                        f"on shard {shard_fraction}: {type(exc).__name__}"
                    )
                await self._result_queue.put((candidate, exc, shard_idx))
            finally:
                self._shard_inflight[shard_idx] -= 1
                self._total_inflight -= 1
                # Decrement rung inflight using the SAME key we incremented with
                # Don't recalculate - the candidate may have been promoted while evaluating!
                # Use get() with default 0 to avoid KeyError if key was removed during recompute
                if rung_key_at_launch in self._inflight_by_rung:
                    self._inflight_by_rung[rung_key_at_launch] -= 1
                # Release global example-level budget
                self._examples_inflight = max(0, self._examples_inflight - per_cand_concurrency)
                # Clear inflight fingerprint using the captured value from launch time
                # This prevents the candidate from being launched again until this evaluation completes
                self._inflight_fingerprints.discard(fingerprint_at_launch)

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
                candidate, payload, shard_idx = self._result_queue.get_nowait()
            except _asyncio.QueueEmpty:
                remaining = deadline - _time.time()
                if remaining <= 0:
                    break
                try:
                    candidate, payload, shard_idx = await _asyncio.wait_for(self._result_queue.get(), timeout=remaining)
                except _asyncio.TimeoutError:
                    break
            await self._stream_process_result(candidate, payload, shard_idx)
            count += 1

        return count

    async def _ingest_result(self, candidate: Candidate, result: EvalResult) -> tuple[Candidate, str]:
        """Update internal state with a freshly evaluated result."""
        shard_fraction = result.shard_fraction or 0.0
        quality = result.objectives.get(self.config.promote_objective, 0.0)

        generation_method = None
        if isinstance(candidate.meta, dict):
            generation_method = candidate.meta.get("generation_method")

        original_fingerprint = candidate.fingerprint
        meta = dict(candidate.meta)
        if "_sched_key" not in meta:
            meta["_sched_key"] = original_fingerprint
        prev_fraction = meta.get("quality_shard_fraction", 0.0)
        prev_quality = meta.get("quality", float("-inf"))
        if shard_fraction > prev_fraction or (shard_fraction == prev_fraction and quality >= prev_quality):
            meta["quality"] = quality
            meta["quality_shard_fraction"] = shard_fraction

        # Compute improvement signal for mutation operators
        orig_meta = candidate.meta if isinstance(candidate.meta, dict) else {}
        parent_score: float | None = None
        if isinstance(orig_meta, dict):
            parent_score_raw = orig_meta.get("parent_score")
            if isinstance(parent_score_raw, (int, float)):
                parent_score = float(parent_score_raw)
            else:
                parent_objectives = orig_meta.get("parent_objectives")
                if isinstance(parent_objectives, dict):
                    parent_val = parent_objectives.get(self.config.promote_objective)
                    if isinstance(parent_val, (int, float)):
                        parent_score = float(parent_val)
        prev_quality_val = prev_quality if isinstance(prev_quality, (int, float)) and prev_quality != float("-inf") else None
        if isinstance(parent_score, (float, int)):
            delta_quality = quality - float(parent_score)
        elif prev_quality_val is not None:
            delta_quality = quality - float(prev_quality_val)
        else:
            delta_quality = quality

        candidate_with_meta = Candidate(text=candidate.text, meta=meta)

        cand_hash = candidate_key(candidate_with_meta)
        sched_key = candidate_with_meta.meta.get("_sched_key") if isinstance(candidate_with_meta.meta, dict) else None
        if isinstance(sched_key, str):
            self._promotion_pending.discard(sched_key)
            self._sched_to_fingerprint[sched_key] = candidate_with_meta.fingerprint
        else:
            self._promotion_pending.discard(candidate_with_meta.fingerprint)
        self.latest_results[cand_hash] = result
        self.evaluations_run += 1

        prev_idx = min(self.scheduler.current_shard_index(candidate_with_meta), len(self._runtime_shards) - 1)
        decision = self.scheduler.record(candidate_with_meta, result, self.config.promote_objective)

        # Track scheduler decisions in metrics
        if decision == "promoted":
            self.metrics.record_promotion(prev_idx)
        elif decision == "pruned":
            self.metrics.record_pruning()
        elif decision == "completed":
            self.metrics.record_completion()

        if decision in ("promoted", "completed") and 0 <= prev_idx < len(self._rung_promotions):
            self._rung_promotions[prev_idx] += 1

        if isinstance(candidate_with_meta.meta, dict):
            parent_fp = candidate_with_meta.meta.get("parent")
            parent_key = candidate_with_meta.meta.get("parent_sched_key", parent_fp)
            if parent_key:
                self._update_lineage_history(parent_key, candidate_with_meta, result)

                # Track operator performance: delta quality vs parent
                parent_objectives = candidate_with_meta.meta.get("parent_objectives")
                if isinstance(parent_objectives, dict):
                    parent_quality = parent_objectives.get("quality", 0.0)
                    child_quality = result.objectives.get("quality", 0.0)
                    delta_quality = child_quality - parent_quality

                    # Determine which operator was used
                    operator = candidate_with_meta.meta.get("operator", "unknown")
                    self.metrics.record_operator_outcome(operator, delta_quality)

        if decision in ("promoted", "completed"):
            await self.archive.insert(candidate_with_meta, result)
            self._record_candidate_promotion(candidate_with_meta)
            await self._maybe_share(candidate_with_meta)
            # Update archive size metrics
            pareto_size = len(self.archive.pareto)
            qd_size = len(self.archive.qd_grid)
            self.metrics.update_archive_sizes(pareto_size, qd_size)

        self._register_failures(result)

        promotions = self.scheduler.promote_ready()
        if promotions:
            self.logger.log(f"🔍 Got {len(promotions)} promotion(s) from scheduler, enqueuing for next shard...")
            for p in promotions:
                rung = self.scheduler.current_shard_index(p)
                shard_frac = self._runtime_shards[rung] if rung < len(self._runtime_shards) else 1.0
                generation = self._get_generation(p)
                source = "seed" if generation == 0 else f"gen-{generation if generation is not None else '?'}"
                self.logger.log(
                    f"   ⬆️  {source} promoted to shard {rung} ({shard_frac:.0%}) [fp={p.fingerprint[:12]}...]"
                )
            self.enqueue(promotions)
            self.logger.log(f"   📋 Queue after promotion: total={len(self.queue)}, per-shard={[len(q) for q in self._per_shard_queue]}")

        if decision == "promoted":
            if isinstance(sched_key, str):
                self._promotion_pending.add(sched_key)
            else:
                self._promotion_pending.add(candidate_with_meta.fingerprint)

        if generation_method and hasattr(self.mutator, "report_outcome"):
            self.mutator.report_outcome(generation_method, delta_quality)

        # NOTE: Do NOT clear _inflight_fingerprints here! The finally block in _stream_launch
        # is responsible for clearing it after all bookkeeping is complete. Clearing it here
        # would allow the same candidate to be launched twice before the finally block runs.

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
    ) -> None:
        if isinstance(payload, Exception):
            self._launch_start_times.pop(candidate_key(candidate), None)
            meta = candidate.meta if isinstance(candidate.meta, dict) else {}
            sched_key = meta.get("_sched_key")
            if isinstance(sched_key, str):
                self._promotion_pending.discard(sched_key)
            self._promotion_pending.discard(candidate.fingerprint)
            # NOTE: Do NOT clear _inflight_fingerprints here - the finally block handles it
            return

        result: EvalResult = payload
        self._update_eval_metrics(candidate, result)
        await self._ingest_result(candidate, result)

    async def _stream_check_target(self, window_id: int) -> bool:
        if self.config.target_quality is None:
            return False

        pareto = self.archive.pareto_entries()
        if not pareto:
            return False

        full_shard = self._runtime_shards[-1]
        full_entries = [entry for entry in pareto if entry.result.shard_fraction == full_shard]
        if not full_entries:
            return False

        best_quality = max(entry.result.objectives.get(self.config.promote_objective, 0.0) for entry in full_entries)
        if best_quality < self.config.target_quality:
            return False

        if self.show_progress:
            self.logger.log(f"🎯 Target quality {self.config.target_quality:.2%} reached! Achieved: {best_quality:.2%}")
            self.logger.log(f"   (Verified on full dataset: {full_shard:.0%} of examples)")
        return True

    # === Streaming Launch Helpers ===

    async def _seed_archive(self, seeds: Sequence[Candidate]) -> None:
        shard_fraction = self._runtime_shards[0]
        shard_size = self._shard_size(shard_fraction)
        shard = self.sampler.sample_shard(self.round_index, shard_size)

        if self.show_progress:
            self.logger.log(
                f"🌱 Evaluating {len(seeds)} seed prompt(s) on {shard_size} examples ({shard_fraction:.0%} of dataset)..."
            )
            self.logger.log("   This establishes the baseline for optimization.")

        seeds_count = max(1, len(seeds))
        per_seed_budget = self._effective_concurrency // seeds_count if self._effective_concurrency >= seeds_count else 1
        seed_concurrency = max(1, min(len(shard), per_seed_budget or 1))

        results: list[EvalResult] = []
        for seed in seeds:
            result = await self._evaluate_candidate(
                seed,
                shard,
                shard_fraction,
                show_progress=self.show_progress,
                concurrency_override=seed_concurrency,
            )
            results.append(result)

        if self.show_progress:
            self.logger.log("   ✓ Seed evaluation complete!")

        enriched: list[Candidate] = []
        for seed, result in zip(seeds, results, strict=False):
            quality = result.objectives.get(self.config.promote_objective, 0.0)
            if self.show_progress:
                self.logger.log(
                    f"   📊 Seed quality on shard 0: {quality:.1%} "
                    f"(fp={seed.fingerprint[:12]}...)"
                )

            candidate_with_meta, decision = await self._ingest_result(seed, result)
            enriched.append(candidate_with_meta)

            if self.show_progress:
                current_rung = self.scheduler.current_shard_index(candidate_with_meta)
                if decision == "promoted":
                    self.logger.log(
                        f"   ✅ Seed PROMOTED to shard {current_rung} ({self._runtime_shards[current_rung]:.0%}) "
                        f"for full evaluation"
                    )
                else:
                    self.logger.log(
                        f"   ⏸️  Seed decision: {decision}, current_rung={current_rung}"
                    )

            # Mark seeds as generation 0
            seed_key = candidate_with_meta.meta.get("_sched_key") if isinstance(candidate_with_meta.meta, dict) else None
            if not isinstance(seed_key, str):
                seed_key = candidate_with_meta.fingerprint
            self._candidate_generations[seed_key] = 0

        self.enqueue(enriched)

        if self.show_progress:
            self.logger.log(
                f"   🎯 Seeds ready for next evaluation: queue={len(self.queue)}, "
                f"per_shard={[len(q) for q in self._per_shard_queue]}"
            )

    def _select_batch(self) -> list[Candidate]:
        batch: list[Candidate] = []
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

        full_shard = self._runtime_shards[-1]  # Longest shard (e.g., 1.0 = 100%)

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
        best_quality = max(entry.result.objectives.get(self.config.promote_objective, 0.0) for entry in pareto)
        return best_quality

    def _get_generation(self, candidate: Candidate) -> int | None:
        meta = candidate.meta if isinstance(candidate.meta, dict) else {}
        key = meta.get("_sched_key") if isinstance(meta, dict) else None
        lookup_key = key if isinstance(key, str) else candidate.fingerprint
        return self._candidate_generations.get(lookup_key)

    async def _spawn_mutations(self, callback: Callable[[Candidate], None] | None = None) -> None:
        """Generate mutations using batched reflection for efficiency."""
        # Check if we've exceeded the evaluation budget before spawning mutations
        if self.max_evaluations is not None and self.evaluations_run >= self.max_evaluations:
            self.logger.log(
                f"[SPAWN_MUTATIONS] eval budget exceeded ({self.evaluations_run} >= {self.max_evaluations})",
                LogLevel.DEBUG,
            )
            return

        entries = self.archive.pareto_entries()
        if not entries:
            self.logger.log("[SPAWN_MUTATIONS] archive empty, skipping", LogLevel.DEBUG)
            return

        num_mutations = self._mutation_budget()
        if num_mutations <= 0:
            self.logger.log(
                f"[SPAWN_MUTATIONS] budget={num_mutations}, queue={len(self.queue)}, "
                f"buffer={len(self._mutation_buffer)} - skipping",
                LogLevel.DEBUG,
            )
            return

        self.logger.log(
            f"[SPAWN_MUTATIONS] WILL GENERATE {num_mutations} mutations from {len(entries)} parents",
            LogLevel.DEBUG,
        )

        # Log that we're actually going to try generating
        self.logger.log(
            f"🔬 spawn_mutations: WILL GENERATE {num_mutations} mutations from {len(entries)} parents"
        )

        self.logger.log(
            f"🧭 spawn_mutations: parents={len(entries)} budget={num_mutations} "
            f"queue={len(self.queue)} buffer={len(self._mutation_buffer)}"
        )

        # Catch timeouts and other RuntimeErrors from LLM calls gracefully
        try:
            mutations = await self._generate_mutations_batched(entries, num_mutations)
        except RuntimeError as e:
            if "timeout" in str(e).lower():
                self.logger.log(f"⚠️  spawn_mutations: LLM timeout - {e}")
                self.logger.log("   Continuing without mutations from this batch")
                return
            # Re-raise if not a timeout error
            raise
        except Exception as e:
            # Log any other unexpected errors but don't crash the mutation task
            self.logger.log(f"⚠️  spawn_mutations: Unexpected error - {type(e).__name__}: {e}")
            self.logger.log("   Continuing without mutations from this batch")
            return

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
        entries: list[ArchiveEntry],
        num_mutations: int,
    ) -> list[Candidate]:
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

        time.time() - parent_select_start

        # Use cached evaluation results instead of fresh eval (saves 3-5s per round!)
        # We already have traces from the ASHA evaluation that just ran
        reflection_minibatch_size = 5
        reflection_examples: list[dict[str, object]] = []

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
                        feedback_parts = [
                            f"The generated response is incorrect. The correct answer is '{expected_answer}'."
                        ]
                        feedback_parts.append(
                            "Ensure that the correct answer is included in the response exactly as it is."
                        )
                        if additional_context and isinstance(additional_context, dict):
                            context_lines = [f"{k}: {v}" for k, v in additional_context.items()]
                            if context_lines:
                                feedback_parts.append(
                                    f"Here is some additional context that might be helpful:\n{chr(10).join(context_lines)}"
                                )
                        feedback = " ".join(feedback_parts)

                    reflection_examples.append(
                        {
                            "input": example_input,
                            "assistant_output": assistant_output,
                            "expected_answer": expected_answer,
                            "feedback": feedback,
                            "additional_context": additional_context,
                        }
                    )

        time.time() - cache_start

        # Build parent contexts for mutation (all selected parents)
        # IMPORTANT: Use latest_results instead of entry.result to get the most recent
        # (and typically full-dataset) evaluation, not the archived partial-shard result
        parent_contexts: list[dict[str, object]] = []

        # Determine minimum shard for mutation eligibility
        # Don't mutate from candidates only evaluated on the tiny first shard.
        # If no parent has reached the higher shard yet (e.g., full evaluation failed),
        # temporarily fall back to the first shard so optimization continues.
        if len(self._runtime_shards) <= 1:
            min_shard_for_mutation = self._runtime_shards[0]
        else:
            min_shard_for_mutation = self._runtime_shards[1]

        has_eligible_parent = any(
            (entry.result.shard_fraction or 0.0) >= min_shard_for_mutation for entry in selected_entries
        )
        effective_min_shard = min_shard_for_mutation if has_eligible_parent else self._runtime_shards[0]

        for entry in selected_entries:
            # Skip parents that haven't been evaluated on a meaningful shard yet
            # This prevents mutating from 100% scores on tiny samples before seeing real failures
            current_shard = entry.result.shard_fraction or 0.0
            if current_shard < effective_min_shard:
                continue

            # Prefer latest_results (most recent eval, usually full dataset)
            # Fall back to entry.result if not found in latest_results
            latest_result = self.latest_results.get(entry.candidate.fingerprint)
            parent_objectives = latest_result.objectives if latest_result else entry.result.objectives

            candidate_with_parent = entry.candidate.with_meta(
                parent_objectives=parent_objectives,
            )
            parent_meta = candidate_with_parent.meta if isinstance(candidate_with_parent.meta, dict) else {}
            parent_key = parent_meta.get("_sched_key") if isinstance(parent_meta, dict) else None
            if not isinstance(parent_key, str):
                parent_key = entry.candidate.fingerprint

            # Note: We used to skip parents in _promotion_pending, but this caused issues
            # when all parents are pending promotion (e.g., early in optimization).
            # It's safe to mutate from pending parents using their latest evaluation results.

            lineage = list(self._lineage_history.get(parent_key, ()))
            parent_contexts.append(
                {
                    "candidate": candidate_with_parent,
                    "failures": [],  # Not used anymore, we have fresh reflection_examples
                    "lineage": lineage,
                }
            )

        if not parent_contexts:
            if self.show_progress:
                self.logger.log(
                    "⏹️  spawn_mutations: no parents past shard threshold yet; will retry after more evaluations"
                )
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
        mutator_duration = time.time() - mutator_start
        self.metrics.record_mutation_batch(len(mutations), mutator_duration)
        self.metrics.time_mutation_total += mutator_duration
        self._record_mutation_generation(num_mutations, mutations)

        batch_duration = time.time() - batch_start
        self.metrics.time_scheduler_total += batch_duration - mutator_duration  # Other overhead

        return mutations

    def _record_mutation_generation(self, requested: int, mutations: Sequence[Candidate]) -> None:
        if requested > 0:
            self._mutations_requested += requested
        generated = len(mutations)
        self._mutations_generated += generated
        for candidate in mutations:
            meta = candidate.meta if isinstance(candidate.meta, dict) else {}
            child_key = meta.get("_sched_key") if isinstance(meta, dict) else None
            if not isinstance(child_key, str):
                child_key = candidate.fingerprint
            self._children_seen.add(child_key)
            self._sched_to_fingerprint[child_key] = candidate.fingerprint

            parent_key = meta.get("parent_sched_key") if isinstance(meta, dict) else None
            if not parent_key:
                parent_key = meta.get("parent") if isinstance(meta, dict) else None

            if parent_key:
                self._parent_children[parent_key].add(child_key)
                parent_gen = self._candidate_generations.get(parent_key, 0)
                self._candidate_generations[child_key] = parent_gen + 1
            else:
                # No parent info, assume generation 1 (first mutation)
                self._candidate_generations[child_key] = self._candidate_generations.get(child_key, 1)

    def _record_mutation_enqueued(self, count: int) -> None:
        if count <= 0:
            return
        self._mutations_enqueued += count

    def _record_candidate_promotion(self, candidate: Candidate) -> None:
        if not isinstance(candidate.meta, dict):
            return
        parent_fp = candidate.meta.get("parent")
        parent_key = candidate.meta.get("parent_sched_key", parent_fp)
        if not parent_key:
            return
        child_fp = candidate.fingerprint
        child_key = candidate.meta.get("_sched_key", child_fp)
        self._children_seen.add(child_key)
        self._sched_to_fingerprint[child_key] = child_fp
        self._parent_children[parent_key].add(child_key)
        # Track generation if not already tracked
        if child_key not in self._candidate_generations:
            parent_gen = self._candidate_generations.get(parent_key, 0)
            self._candidate_generations[child_key] = parent_gen + 1
        archive_key = candidate_key(candidate)
        if archive_key in self.archive.pareto and child_key not in self._promoted_children:
            self._promoted_children.add(child_key)

    def evolution_snapshot(self, include_edges: bool = False) -> dict[str, Any]:
        edges = sum(len(children) for children in self._parent_children.values())
        snapshot: dict[str, Any] = {
            "mutations_requested": self._mutations_requested,
            "mutations_generated": self._mutations_generated,
            "mutations_enqueued": self._mutations_enqueued,
            "mutations_promoted": len(self._promoted_children),
            "unique_parents": len(self._parent_children),
            "unique_children": len(self._children_seen),
            "evolution_edges": edges,
            "total_evaluations": self.evaluations_run,
        }
        if include_edges:
            snapshot["parent_children"] = {
                self._sched_to_fingerprint.get(parent, parent): [self._sched_to_fingerprint.get(child, child) for child in children]
                for parent, children in self._parent_children.items()
            }
            snapshot["children"] = [self._sched_to_fingerprint.get(child, child) for child in self._children_seen]
            snapshot["promoted_children"] = [self._sched_to_fingerprint.get(child, child) for child in self._promoted_children]
        return snapshot

    def get_candidate_lineage_data(self) -> list[dict[str, Any]]:
        """Return list of all candidates with generation, quality, and status for visualization."""

        data: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        promote_objective = self.config.promote_objective

        # Get all pareto candidates
        for entry in self.archive.pareto_entries():
            meta = entry.candidate.meta if isinstance(entry.candidate.meta, dict) else {}
            key = meta.get("_sched_key") if isinstance(meta, dict) else None
            if not isinstance(key, str):
                key = entry.candidate.fingerprint
            fp = self._sched_to_fingerprint.get(key, entry.candidate.fingerprint)
            seen_keys.add(key)
            data.append(
                {
                    "fingerprint": fp,
                    "generation": self._candidate_generations.get(key, 0),
                    "quality": entry.result.objectives.get(promote_objective, 0.0),
                    "status": "promoted",
                }
            )

        # Get all QD grid candidates
        for entry in self.archive.qd_grid.values():
            meta = entry.candidate.meta if isinstance(entry.candidate.meta, dict) else {}
            key = meta.get("_sched_key") if isinstance(meta, dict) else None
            if not isinstance(key, str):
                key = entry.candidate.fingerprint
            fp = self._sched_to_fingerprint.get(key, entry.candidate.fingerprint)
            # Skip if already in pareto
            if key in seen_keys:
                continue
            seen_keys.add(key)
            data.append(
                {
                    "fingerprint": fp,
                    "generation": self._candidate_generations.get(key, 0),
                    "quality": entry.result.objectives.get(promote_objective, 0.0),
                    "status": "evaluated",
                }
            )

        # Get in-flight candidates (in queue)
        for candidate in self.queue:
            meta = candidate.meta if isinstance(candidate.meta, dict) else {}
            key = meta.get("_sched_key") if isinstance(meta, dict) else None
            if not isinstance(key, str):
                key = candidate.fingerprint
            fp = self._sched_to_fingerprint.get(key, candidate.fingerprint)
            # Get quality from archive if available
            quality = 0.0
            cache_key = candidate_key(candidate)
            latest = self.latest_results.get(cache_key)
            if latest:
                quality = latest.objectives.get(promote_objective, 0.0)

            data.append(
                {
                    "fingerprint": fp,
                    "generation": self._candidate_generations.get(key, 0),
                    "quality": quality,
                    "status": "in_flight",
                }
            )

        return data

    def _mutation_budget(self) -> int:
        """Compute how many new mutations we should request this cycle."""
        max_mut = self.config.max_mutations_per_round or 0
        if max_mut <= 0:
            return 0

        if self._mutation_throttle and max_mut > self._mutation_min:
            max_mut = max(self._mutation_min, max_mut // 2)

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

    def _sample_task_examples_for_spec_induction(self, num_examples: int = 3) -> list[dict[str, object]]:
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
        *,
        concurrency_override: int | None = None,
    ) -> EvalResult:
        concurrency = concurrency_override if concurrency_override is not None else self._effective_concurrency
        result = await self.evaluator.eval_on_shard(
            candidate,
            shard,
            concurrency=concurrency,
            shard_fraction=shard_fraction,
            show_progress=show_progress,
        )
        return result

    def _register_failures(self, result: EvalResult) -> None:
        hard_examples: list[str] = []
        for trace in result.traces:
            example_id = trace.get("example_id") if isinstance(trace, dict) else None
            quality = trace.get("quality") if isinstance(trace, dict) else None
            if example_id is None or quality is None:
                continue
            if quality < result.objectives.get("quality", quality):
                hard_examples.append(example_id)
        if hard_examples:
            self.sampler.register_hard_examples(hard_examples)

    def _update_lineage_history(self, parent_key: str, child: Candidate, result: EvalResult) -> None:
        if not parent_key:
            return
        history = self._lineage_history[parent_key]
        promote_obj = self.config.promote_objective
        child_quality = result.objectives.get(promote_obj, 0.0)
        parent_quality: float | None = None
        meta = child.meta if isinstance(child.meta, dict) else {}
        if isinstance(meta, dict):
            parent_quality = meta.get("parent_score")
            if parent_quality is None:
                parent_obj = meta.get("parent_objectives")
                if isinstance(parent_obj, dict):
                    parent_raw = parent_obj.get(promote_obj)
                    if isinstance(parent_raw, (int, float)):
                        parent_quality = float(parent_raw)

        child_sched = None
        if isinstance(meta, dict):
            child_sched = meta.get("_sched_key")
            if isinstance(child_sched, str):
                self._sched_to_fingerprint[child_sched] = child.fingerprint

        failure_summaries: list[dict[str, object]] = []
        for trace in result.traces:
            if not isinstance(trace, dict):
                continue
            example_id = trace.get("example_id")
            if example_id is None:
                continue
            trace_quality = trace.get("quality")
            if isinstance(trace_quality, (int, float)) and trace_quality >= child_quality:
                continue
            failure_summaries.append(
                {
                    "example_id": example_id,
                    "quality": trace_quality,
                }
            )
            if len(failure_summaries) >= 3:
                break

        entry = {
            "child_fingerprint": child.fingerprint,
            "child_sched_key": child_sched,
            "child_text": child.text,
            "quality": child_quality,
            "shard_fraction": result.shard_fraction or 0.0,
            "tokens": result.objectives.get("tokens", 0),
            "parent_quality": parent_quality,
            "generation_method": meta.get("generation_method") if isinstance(meta, dict) else None,
            "failures": failure_summaries,
        }
        history.appendleft(entry)

    def _shard_size(self, shard_fraction: float) -> int:
        total = max(len(self.sampler.example_ids), 1)
        size = max(1, int(total * shard_fraction))
        return min(size, total)

    def _check_stop_governor(self) -> tuple[bool, dict]:
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
                e.result.objectives.get("neg_cost", float("-inf")),
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
            best_cost=best_entry.result.objectives.get("neg_cost", float("-inf")),
            frontier_ids={e.candidate.fingerprint for e in pareto_entries},
            qd_filled_cells=len(current_cells),
            qd_total_cells=self.config.qd_bins_length * self.config.qd_bins_bullets * len(self.config.qd_flags),
            qd_novelty_rate=qd_novelty_rate,
            total_tokens_spent=self.total_tokens_spent,
        )

        self.stop_governor.update(metrics)
        return self.stop_governor.should_stop()

    async def finalize(self, delta: float | None = None) -> None:
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
        if hasattr(self, "_save_task") and not self._save_task.done():
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

    async def _restore_state(self, state: dict) -> None:
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
