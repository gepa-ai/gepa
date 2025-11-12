"""
High-level orchestrator coordinating selection, evaluation, and mutation.

This module stitches together archive, scheduler, evaluator, and mutator
components. The orchestrator operates within a single island; multi-island
concurrency is achieved via ``islands.spawn_islands`` using asyncio tasks
in the same process (async ring topology), not separate OS processes.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import math
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Iterable, Sequence

from turbo_gepa.logging import build_progress_snapshot
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


def _percentile(samples: Sequence[float], quantile: float) -> float:
    """Return the quantile value using linear interpolation."""

    if not samples:
        return 0.0
    values = sorted(samples)
    if len(values) == 1:
        return float(values[0])
    quantile = max(0.0, min(1.0, quantile))
    pos = (len(values) - 1) * quantile
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(values[int(pos)])
    fraction = pos - lower
    return float(values[lower] + (values[upper] - values[lower]) * fraction)


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
        example_sampler: Callable[[int], list[dict[str, object]]] | None = None,
        metrics_callback: Callable | None = None,
        logger: LoggerProtocol | None = None,
        run_id: str | None = None,
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
                variance_tolerance=config.variance_tolerance,
                shrinkage_alpha=config.shrinkage_alpha,
            )
        )
        self._runtime_shards: list[float] = list(self.config.shards)
        self._final_rung_index: int = max(0, len(self._runtime_shards) - 1)
        self.queue: deque[Candidate] = deque()
        self._pending_fingerprints: set[str] = set()
        self._inflight_fingerprints: set[str] = set()
        self._next_shard: int = 0
        self.latest_results: dict[str, EvalResult] = {}
        self.evaluations_run: int = 0
        self.round_index: int = 0
        self.show_progress = show_progress
        island_id = island_context.island_id if island_context else 0
        self.island_id = island_id
        # Stop governor: always enabled for convergence detection
        self.stop_governor = StopGovernor(config.stop_governor_config)
        self.total_tokens_spent: int = 0
        self._shard_cache: dict[tuple[int, float], list[str]] = {}
        self.metrics_callback = metrics_callback
        self.logger: LoggerProtocol = logger or StdOutLogger()
        self.max_rounds: int | None = None
        self.max_evaluations: int | None = None
        self.rounds_completed: int = 0

        # Streaming mode: buffer for mutations generated in background
        self._mutation_task: asyncio.Task | None = None
        self.eval_batches_completed: int = 0  # For migration timing

        # Streaming evaluation infrastructure
        self._inflight_tasks: dict[str, asyncio.Task] = {}  # cand_hash -> task
        self._result_queue: asyncio.Queue[tuple[Candidate, EvalResult | Exception, int]] = asyncio.Queue()
        self._total_inflight: int = 0  # Total evaluations in flight across all shards
        self._fd_guard_limit: int | None = self._detect_fd_guard()
        self._max_total_inflight: int = self._clamp_inflight(self.config.eval_concurrency)
        self._adaptive_inflight_floor: int = max(1, self.config.max_final_shard_inflight or 1)
        self._examples_inflight: int = 0
        self._launch_start_times: dict[str, float] = {}
        self._latency_ema: float = 0.0
        self._latency_samples: int = 0
        self._latency_history: deque[float] = deque(maxlen=200)
        self._eval_samples: int = 0
        self._timeout_count: int = 0
        self._last_inflight_adjust: float = 0.0
        self._last_finalcap_adjust: float = 0.0
        self._last_effconc_adjust: float = 0.0
        self._mutation_throttle: bool = False
        base_mut = self.config.max_mutations_per_round or max(16, self.config.eval_concurrency // 2)
        self._max_mutations_ceiling = base_mut
        self._mutation_min = max(4, base_mut // 4)
        self._rung_launches: list[int] = [0] * len(self._runtime_shards)
        self._rung_promotions: list[int] = [0] * len(self._runtime_shards)
        self._promotion_ema: list[float] = [0.5] * len(self._runtime_shards)  # Initialize at 0.5 (neutral)
        self._final_success_ema: float = 0.3  # Success rate at final rung
        self._ema_alpha: float = 0.1  # EMA smoothing factor
        self._priority_beta: float = 0.05  # Quality tie-breaker weight
        self._recent_delta_weight: float = 0.5  # Boost for parents with recent improvements
        self._queue_buffer_mult: float = 3.0  # Target queue depth = 3 × concurrency
        # Maintain 3x concurrency buffer to prevent evaluator starvation
        # when mutations temporarily lag behind fast evaluations
        self._mutation_kp: float = 0.2  # Proportional gain for mutation pacing
        self._priority_counter: int = 0  # Tie-breaker for heap ordering
        self._priority_queue: list = []  # Heap: (-priority, -rung_idx, counter, candidate)
        final_fraction = self._runtime_shards[-1] if self._runtime_shards else 1.0
        self._cost_remaining: list[float] = [
            max(1e-6, final_fraction - (self._runtime_shards[i - 1] if i > 0 else 0.0))
            for i in range(len(self._runtime_shards))
        ]
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
        self._candidate_eval_examples: dict[str, list[str]] = {}
        self._inflight_by_rung: defaultdict[int, int] = defaultdict(int)
        self._final_inflight_peak: int = 0
        self._stop_reason: str | None = None
        self._run_started_at: float | None = None
        self._run_id = run_id or uuid.uuid4().hex[:8]

        # Metrics tracking
        self.metrics = Metrics()
        self.metrics.target_quality = self.config.target_quality
        self.metrics.target_shard_fraction = self.config.target_shard_fraction
        # Pass metrics to evaluator for cache tracking
        self.evaluator.metrics = self.metrics

        # Scheduler state derived from capacities
        self._recompute_capacities()

    def _set_stop_reason(self, reason: str) -> None:
        """Record why the orchestration loop decided to stop."""
        if not self._stop_reason:
            self._stop_reason = reason

    def _record_best_quality(self, quality: float, shard_fraction: float | None) -> None:
        shard_fraction = shard_fraction if shard_fraction is not None else 0.0
        if quality > self.metrics.best_quality:
            self.metrics.best_quality = quality
            self.metrics.best_shard_fraction = shard_fraction
        target_shard = self.metrics.target_shard_fraction or 1.0
        if (
            not self.metrics.baseline_recorded
            and shard_fraction + 1e-6 >= target_shard
        ):
            self.metrics.baseline_quality = quality
            self.metrics.baseline_recorded = True
        elapsed = None
        if self._run_started_at is not None:
            elapsed = max(0.0, time.time() - self._run_started_at)
        self.metrics.record_rung_sample(shard_fraction, quality, elapsed)
        self._maybe_record_target_hit(quality, shard_fraction, elapsed)

    def _maybe_record_target_hit(self, quality: float, shard_fraction: float | None, elapsed: float | None) -> None:
        shard_fraction = shard_fraction if shard_fraction is not None else 0.0
        target_quality = self.config.target_quality
        target_shard_fraction = self.config.target_shard_fraction or self._runtime_shards[-1]
        if (
            target_quality is None
            or self.metrics.time_to_target_seconds is not None
            or elapsed is None
        ):
            return
        if shard_fraction + 1e-6 >= target_shard_fraction and quality >= target_quality:
            self.metrics.time_to_target_seconds = elapsed
            if self.show_progress:
                self.logger.log(
                    f"🏎️  Target reached at {elapsed:.1f}s (quality={quality:.3f} @ {shard_fraction:.0%})",
                    LogLevel.WARNING,
                )

    def _log_structured(self, event: str, level: LogLevel = LogLevel.INFO, **fields: Any) -> None:
        payload = {"event": event, "run_id": self._run_id}
        payload.update(fields)
        try:
            message = json.dumps(payload, sort_keys=True)
        except Exception:
            # Fallback to repr if something isn't serializable
            message = json.dumps(
                {k: repr(v) for k, v in payload.items()},
                sort_keys=True,
            )
        self.logger.log(message, level)

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def run_started_at(self) -> float | None:
        return self._run_started_at

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def total_inflight(self) -> int:
        return self._total_inflight

    @property
    def examples_inflight(self) -> int:
        return self._examples_inflight

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

    def _recompute_capacities(self) -> None:
        # SIMPLIFIED: Just use eval_concurrency directly, no artificial limits
        self._effective_concurrency = max(1, self.config.eval_concurrency)
        # Resize metric lists to match current shard count
        num_shards = len(self._runtime_shards)
        self._rung_launches = self._resize_metric_list(self._rung_launches, num_shards)
        self._rung_promotions = self._resize_metric_list(self._rung_promotions, num_shards)
        self._promotion_ema = self._resize_float_list(self._promotion_ema, num_shards)
        if num_shards:
            final_fraction = self._runtime_shards[-1]
            self._cost_remaining = [
                max(1e-6, final_fraction - (self._runtime_shards[i - 1] if i > 0 else 0.0))
                for i in range(num_shards)
            ]
        else:
            self._cost_remaining = []

    def enqueue(self, candidates: Iterable[Candidate]) -> None:
        for candidate in candidates:
            if not candidate.text.strip():
                continue
            if self.config.queue_limit and len(self._priority_queue) >= self.config.queue_limit:
                # Drop lowest-value additions when backlog is already saturated
                continue
            fingerprint = candidate.fingerprint
            if fingerprint in self._pending_fingerprints or fingerprint in self._inflight_fingerprints:
                continue

            # Add to priority queue with computed priority
            rung_idx = self.scheduler.current_shard_index(candidate)
            priority = self._compute_priority(candidate, rung_idx)

            # Use negative priority for max-heap (heapq is min-heap)
            # Secondary sort by negative rung_idx (prefer higher rungs)
            # Tertiary sort by counter for FIFO tie-breaking
            self._priority_counter += 1
            heapq.heappush(self._priority_queue, (-priority, -rung_idx, self._priority_counter, candidate))

            self._pending_fingerprints.add(fingerprint)
            # Keep legacy queue in sync for diagnostics and persistence
            self.queue.append(candidate)

    def _get_rung_key(self, candidate: Candidate) -> str:
        """Get rung key string for a candidate based on its current shard.

        Uses scheduler.current_shard_index to ensure consistency with enqueue logic.
        """
        idx = self.scheduler.current_shard_index(candidate)
        shard_fraction = self._runtime_shards[min(idx, len(self._runtime_shards) - 1)]
        return str(shard_fraction)

    def _update_ema(self, old_value: float, new_value: float) -> float:
        """Update EMA with alpha smoothing."""
        return (1 - self._ema_alpha) * old_value + self._ema_alpha * new_value

    def _compute_priority(self, candidate: Candidate, rung_idx: int) -> float:
        """Compute priority = P_finish(rung_idx) / cost_remaining[rung_idx].

        P_finish = (product of promotion_ema from rung_idx to final) × final_success_ema
        Optional quality tie-breaker: multiply by (1 + beta × normalized_quality)
        """
        # Calculate probability of reaching and succeeding at final rung
        p_finish = self._final_success_ema
        for r in range(rung_idx, len(self._runtime_shards) - 1):
            p_finish *= self._promotion_ema[r]

        # Cost remaining at this rung
        cost_remaining = self._cost_remaining[rung_idx]
        if cost_remaining <= 0:
            cost_remaining = 1e-6  # Avoid division by zero

        # Base priority
        priority = p_finish / cost_remaining

        # Optional quality tie-breaker
        if self._priority_beta > 0 and isinstance(candidate.meta, dict):
            quality = candidate.meta.get("quality")
            if isinstance(quality, (int, float)):
                # Normalize quality to [0, 1] range (assuming quality is already in this range)
                normalized_quality = max(0.0, min(1.0, float(quality)))
                priority *= (1 + self._priority_beta * normalized_quality)

        # Boost parents with strong recent improvements (moving average of delta quality)
        if isinstance(candidate.meta, dict):
            recent_avg = candidate.meta.get("recent_delta_avg")
            if isinstance(recent_avg, (int, float)) and recent_avg > 0:
                boost = max(0.0, min(1.0, recent_avg))
                priority *= (1 + self._recent_delta_weight * boost)

        return priority

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
            self._latency_history.append(duration)
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
        # Adaptive shards always enabled - auto-tune based on promotion rates
        num_shards = len(self._runtime_shards)
        if num_shards < 2:
            self._rung_launches = [0] * len(self._rung_launches)
            self._rung_promotions = [0] * len(self._rung_promotions)
            self._promotion_ema = [0.0] * len(self._promotion_ema)
            return

        new_shards = list(self._runtime_shards)
        adjusted = False
        target_rate = 0.5
        tolerance = 0.05  # Allow ±5% window before adjusting

        for idx in range(num_shards - 1):
            launches = self._rung_launches[idx] if idx < len(self._rung_launches) else 0
            if launches < 3:
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

            # SIMPLE: adjust by fixed 15% in the appropriate direction
            step = 0.15
            scale = 1.0 - step if shrink else 1.0 + step
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
        latencies = list(self._latency_history)
        if len(latencies) < 5:
            return

        import statistics
        now = time.time()

        timeout_ratio = self._current_timeout_ratio()
        lat_mean = statistics.fmean(latencies)
        lat_p95 = _percentile(latencies, 0.95)

        timeout_cap = self.config.eval_timeout_seconds or 60.0
        high_latency = min(max(10.0, 0.75 * timeout_cap), timeout_cap)
        low_latency = 6.0

        backlog0 = len(self._priority_queue)
        launches0 = self._rung_launches[0] if self._rung_launches else 0
        promotions0 = self._rung_promotions[0] if self._rung_promotions else 0
        promo_rate0 = promotions0 / max(1, launches0)
        backlog_high = max(3 * self.config.batch_size, 12)
        backlog_low = max(self.config.batch_size, 8)

        limit = self.config.eval_concurrency
        pressure = backlog0 / max(1, self._max_total_inflight)
        at_cap = self._total_inflight >= max(1, self._max_total_inflight - 1)

        should_expand = (
            self._max_total_inflight < limit
            and pressure > 1.2
            and at_cap
            and timeout_ratio < 0.04
            and lat_p95 < high_latency * 0.8
        )
        should_reduce = (
            timeout_ratio > 0.08
            or lat_p95 > high_latency
            or (lat_mean > high_latency and self._max_total_inflight > self._adaptive_inflight_floor)
        )

        adjusted = False
        if should_expand or should_reduce:
            if now - self._last_inflight_adjust >= 1.0:
                old_limit = self._max_total_inflight
                if should_expand:
                    step = max(1, max(1, old_limit // 3))
                    new_limit = min(limit, old_limit + step)
                else:
                    step = max(1, max(1, old_limit // 4))
                    floor = max(self._adaptive_inflight_floor, 1)
                    new_limit = max(floor, old_limit - step)
                new_limit = self._clamp_inflight(new_limit)
                if new_limit != old_limit:
                    self._max_total_inflight = new_limit
                    self._recompute_capacities()
                    self._last_inflight_adjust = now
                    adjusted = True
                    direction = "↑" if new_limit > old_limit else "↓"
                    if self.show_progress:
                        self.logger.log(
                            (
                                f"⚙️  Adaptive inflight {direction} {old_limit}→{new_limit} "
                                f"| p95={lat_p95:.1f}s mean={lat_mean:.1f}s timeout={timeout_ratio:.1%} backlog={backlog0}"
                            ),
                            LogLevel.WARNING,
                        )
                    if self.config.max_mutations_per_round:
                        if new_limit > old_limit:
                            self.config.max_mutations_per_round = min(
                                self.config.max_mutations_per_round + 1, self._max_mutations_ceiling
                            )
                        else:
                            self.config.max_mutations_per_round = max(
                                self._mutation_min, self.config.max_mutations_per_round - 1
                            )

        if backlog0 > backlog_high and promo_rate0 < 0.2:
            self._mutation_throttle = True
        elif backlog0 < backlog_low:
            self._mutation_throttle = False

        if adjusted:
            # Slowly decay launch counters to keep ratios reactive
            self._rung_launches = [max(0, int(count * 0.8)) for count in self._rung_launches]
            self._rung_promotions = [max(0, int(count * 0.8)) for count in self._rung_promotions]

        # Adaptive final-rung inflight cap: overlap multiple full-shard candidates when tails are low,
        # back off when p95 grows large. Rate-limit changes to one step per second.
        try:
            lat_p50 = _percentile(latencies, 0.50)
        except Exception:
            lat_p50 = lat_mean
        if self.config.max_final_shard_inflight is not None:
            desired_cap = self.config.max_final_shard_inflight
            cap_floor = 2
            cap_ceil = max(2, int(max(1, getattr(self, "_effective_concurrency", self.config.eval_concurrency)) * 0.75))
            increase = lat_p95 < max(1.4 * lat_p50, 1.2 * lat_mean) and backlog0 > backlog_high
            decrease = lat_p95 > max(1.8 * lat_mean, 1.6 * lat_p50)
            now = time.time()
            if now - self._last_finalcap_adjust >= 1.0:
                new_cap = desired_cap
                if increase and desired_cap < cap_ceil:
                    new_cap = min(cap_ceil, desired_cap + 1)
                elif decrease and desired_cap > cap_floor:
                    new_cap = max(cap_floor, desired_cap - 1)
                if new_cap != desired_cap:
                    old = self.config.max_final_shard_inflight
                    self.config.max_final_shard_inflight = new_cap
                    self._last_finalcap_adjust = now
                    if self.show_progress:
                        direction = "↑" if new_cap > old else "↓"
                        self.logger.log(
                            f"⚙️  Adaptive final-rung cap {direction} {old}→{new_cap} | p95={lat_p95:.1f}s p50={lat_p50:.1f}s mean={lat_mean:.1f}s",
                            LogLevel.WARNING,
                        )

        # Adaptive effective concurrency (network-aware throttle)
        # Target: maximize throughput by staying near provider sweet spot.
        if self.config.auto_scale_eval_concurrency:
            floor = max(1, self.config.min_effective_concurrency or 1)
            ceil = max(1, self.config.eval_concurrency)
            old_eff = getattr(self, "_effective_concurrency", ceil)
            eff = old_eff

            lat_p50 = _percentile(latencies, 0.50)
            high_tail = lat_p95 > max(1.8 * lat_mean, 1.5 * lat_p50)
            low_tail = lat_p95 < max(1.4 * lat_mean, 1.3 * lat_p50)
            saturated = self._examples_inflight >= old_eff

            if now - self._last_effconc_adjust >= 1.0:
                if high_tail and saturated and eff > floor:
                    eff = max(floor, eff - 1)
                elif low_tail and backlog0 > backlog_high and eff < ceil:
                    eff = min(ceil, eff + 1)
                if eff != old_eff:
                    self._effective_concurrency = eff
                    self._last_effconc_adjust = now
                    if self.show_progress:
                        direction = "↑" if eff > old_eff else "↓"
                        self.logger.log(
                            f"⚙️  Adaptive effective concurrency {direction} {old_eff}→{eff} | p95={lat_p95:.1f}s p50={lat_p50:.1f}s mean={lat_mean:.1f}s backlog={backlog0}",
                            LogLevel.WARNING,
                        )

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
        self._stop_reason = None

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

        # Track optimization start time for global timeout - BEFORE seed evaluation
        import time
        optimization_start_time = time.time()
        self._run_started_at = optimization_start_time

        if not resumed:
            await self._seed_archive(seeds)
            if self.metrics_callback is not None:
                self.metrics_callback(build_progress_snapshot(self))

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
        if not self.metrics.baseline_recorded:
            baseline = self._get_best_quality_from_full_shard()
            self.metrics.baseline_quality = baseline

        while True:
            if max_rounds is not None and window_id >= max_rounds:
                self._set_stop_reason(f"max_rounds({max_rounds})")
                break
            if max_evaluations is not None and self.evaluations_run >= max_evaluations:
                self._set_stop_reason("max_evaluations")
                break
            # Check convergence flag from scheduler
            if self.scheduler.converged:
                best_full = self._get_best_quality_from_full_shard()
                target_quality = self.config.target_quality
                if target_quality is None or best_full >= target_quality:
                    _debug_log("🛑 CONVERGED: Scheduler detected convergence on final rung")
                    self._set_stop_reason("scheduler_converged")
                    break
                else:
                    _debug_log(
                        "⚠️  Scheduler convergence ignored (best_full_shard=%.3f < target %.3f)",
                        best_full,
                        target_quality,
                    )
                    self.scheduler.reset_final_rung_convergence()
            # Check global timeout
            if self.config.max_optimization_time_seconds is not None:
                elapsed = time.time() - optimization_start_time
                if elapsed >= self.config.max_optimization_time_seconds:
                    _debug_log(f"⏱️  TIMEOUT: Reached max optimization time ({elapsed:.1f}s >= {self.config.max_optimization_time_seconds:.1f}s)")
                    self._set_stop_reason("timeout")
                    # Cancel all in-flight tasks to exit quickly
                    if self._inflight_tasks:
                        _debug_log(f"   Cancelling {len(self._inflight_tasks)} in-flight evaluations...")
                        for task in self._inflight_tasks.values():
                            task.cancel()
                    break

            # DEBUG: Log critical state before launch attempt
            if loop_iter % 10 == 0 and len(self.queue) > 0 and self._total_inflight == 0:
                _debug_log(f"🔍 MAIN LOOP: queue={len(self.queue)}, inflight={self._total_inflight}")
                _debug_log(f"   Priority queue: {len(self._priority_queue)}")

            if loop_iter % 20 == 0 and self._total_inflight > self._effective_concurrency * 2:
                self.logger.log(
                    f"⚠️  Inflight spike: {self._total_inflight} candidates "
                    f"(configured concurrency={self._effective_concurrency}). "
                    f"Queue={len(self.queue)}, priority_queue={len(self._priority_queue)}, "
                    f"examples_inflight={self._examples_inflight}"
                )

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
                        self.metrics_callback(build_progress_snapshot(self))

            # 4) Refresh mutation generation if queues are running low
            # Only spawn new mutations if we haven't exceeded the evaluation budget
            # SIMPLIFIED: Trigger when queue is running low
            # Keep queue depth at 2x concurrency to maintain saturation
            ready_depth = len(self.queue) + len(self._priority_queue)
            soft_target = max(int(self._max_total_inflight * self._queue_buffer_mult), self.config.batch_size)
            hard_target = max(self._max_total_inflight, self.config.batch_size)
            need_depth = ready_depth < soft_target
            need_queue = len(self.queue) < hard_target and self._total_inflight >= max(1, hard_target // 2)
            should_spawn_mutations = (
                (need_depth or need_queue)
                and (max_evaluations is None or self.evaluations_run < max_evaluations)
            )

            # Debug mutation spawning logic
            if loop_iter % 100 == 0 and len(self.queue) == 0 and self._total_inflight == 0:
                task_status = 'None' if self._mutation_task is None else ('done' if self._mutation_task.done() else 'running')
                _debug_log(
                    f"🔍 MUTATION CHECK: queue={len(self.queue)}, priority={len(self._priority_queue)}, "
                    f"target_soft={soft_target}, need_depth={need_depth}, need_queue={need_queue}, "
                    f"task={task_status}"
                )

            # SIMPLIFIED: Stream mutations directly into queue as they complete
            if should_spawn_mutations:
                # If task exists and is done, clear it immediately
                if self._mutation_task is not None and self._mutation_task.done():
                    _debug_log("🔄 Mutation task completed, clearing it")
                    self._mutation_task = None

                # Spawn streaming mutation worker if no task is running
                if self._mutation_task is None:
                    _debug_log(
                        f"🧬 Starting streaming mutation worker (ready={ready_depth}, target={soft_target})"
                    )
                    self._mutation_task = asyncio.create_task(self._streaming_mutation_worker())

            # 5) Window completion => keep round-indexed metrics in sync
            if (
                not self.queue
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
                        f"🧭 Debug: queue=0 inflight={self._total_inflight} "
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
                    self.metrics_callback(build_progress_snapshot(self))

                # Clear inline progress before showing summary
                if self.show_progress:
                    # logger prints discrete lines; no inline clearing needed
                    pass

                self._shard_cache.clear()

                if self.config.migration_period and window_id % self.config.migration_period == 0:
                    await self._maybe_migrate()

                await self._save_state()
                self._maybe_adjust_shards()

                # Stop governor: check convergence after each round
                should_stop, debug_info = self._check_stop_governor()
                if should_stop:
                    if self.show_progress:
                        self.logger.log(f"🛑 Auto-stopping: {debug_info['reason']}")
                    self._set_stop_reason(f"auto_stop:{debug_info['reason']}")
                    break

            # 6) Target quality guard
            if await self._stream_check_target(window_id):
                self._set_stop_reason("target_quality")
                break

            self._adjust_runtime_parameters()

            # 7) Idle detection - nothing running, nothing queued, mutations finished
            # CRITICAL FIX: Also check _priority_queue for promoted candidates awaiting evaluation
            if (
                launched == 0
                and drained == 0
                and self._total_inflight == 0
                and not self.queue
                and not self._priority_queue  # FIX: Don't exit if promoted candidates are waiting
            ):
                if self._mutation_task is None:
                    _debug_log("🛑 IDLE DETECTION: All work complete, exiting loop")
                    self._set_stop_reason("idle")
                    break
                elif self._mutation_task.done():
                    _debug_log("🛑 IDLE DETECTION: Mutation task done, exiting loop")
                    self._mutation_task = None
                    self._set_stop_reason("idle")
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

        if not self._stop_reason:
            self._stop_reason = "completed"

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
            limit = self.config.max_final_shard_inflight or "∞"
            self.logger.log(
                f"📈 Final rung peak inflight: {self._final_inflight_peak}/{limit}",
                LogLevel.WARNING,
            )
        if self.show_progress:
            self.logger.log("\n" + metrics_summary)

        # Save metrics to .turbo_gepa/metrics/ directory
        import os
        metrics_dir = ".turbo_gepa/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        timestamp = int(time.time())
        metrics_file = f"{metrics_dir}/metrics_{timestamp}.txt"
        with open(metrics_file, "w") as f:
            f.write(f"Run ID: {self._run_id}\n")
            f.write(metrics_summary)
        if self.show_progress:
            self.logger.log(f"📊 Metrics saved to: {metrics_file}")
            snapshot = build_progress_snapshot(self)
            reason = snapshot.stop_reason or "completed"
            self.logger.log(
                f"🏁 TurboGEPA finished run={self._run_id} (reason: {reason}) "
                f"| evals={snapshot.evaluations} best={snapshot.best_quality:.3f}@{snapshot.best_quality_shard:.0%} "
                f"| pareto={snapshot.pareto_size}",
                LogLevel.WARNING,
            )
            if snapshot.best_prompt_snippet:
                self.logger.log(f"   best_prompt: {snapshot.best_prompt_snippet}", LogLevel.WARNING)
            self._log_structured(
                "run_complete",
                level=LogLevel.WARNING,
                reason=reason,
                evaluations=snapshot.evaluations,
                best_quality=round(snapshot.best_quality, 6),
                best_quality_shard=snapshot.best_quality_shard,
                pareto_size=snapshot.pareto_size,
                stop_reason=reason,
                best_prompt=snapshot.best_prompt_snippet,
            )

        # Clear state only if we reached completion (not if stopped early)
        completed = (max_rounds is not None and window_id >= max_rounds) or (
            max_evaluations is not None and self.evaluations_run >= max_evaluations
        )
        if completed:
            self.cache.clear_state()

        if _debug_log_file is not None:
            _debug_log_file.close()

    # === Streaming helpers ===

    def _stream_can_launch(self, candidate: Candidate | None, max_evaluations: int | None) -> bool:
        if max_evaluations is not None and self.evaluations_run >= max_evaluations:
            return False

        if candidate is None:
            return True

        limit = self.config.max_final_shard_inflight
        if limit is None:
            return True

        rung_idx = min(self.scheduler.current_shard_index(candidate), len(self._runtime_shards) - 1)
        if rung_idx == self._final_rung_index:
            inflight = self._inflight_by_rung.get(rung_idx, 0)
            if inflight >= limit:
                if self.show_progress:
                    self.logger.log(
                        f"⏸️  Final rung saturated: inflight={inflight}/{limit}, delaying launch",
                        LogLevel.WARNING,
                    )
                return False
        return True

    async def _stream_launch_ready(self, window_id: int, max_evaluations: int | None) -> int:
        """Launch ready candidates using priority-based scheduling."""
        launched = 0

        # Launch from priority queue until we can't anymore
        while self._priority_queue:
            # Pop highest priority candidate
            try:
                neg_priority, neg_rung_idx, counter, candidate = heapq.heappop(self._priority_queue)
            except IndexError:
                break

            if not self._stream_can_launch(candidate, max_evaluations):
                heapq.heappush(self._priority_queue, (neg_priority, neg_rung_idx, counter, candidate))
                break

            priority = -neg_priority
            rung_idx = -neg_rung_idx

            # Check if candidate is already inflight or completed
            fingerprint = candidate.fingerprint
            if fingerprint in self._inflight_fingerprints:
                continue  # Skip, already being evaluated

            # Candidate is about to launch; clear pending bookkeeping and mirrored queue
            self._pending_fingerprints.discard(fingerprint)
            try:
                self.queue.remove(candidate)
            except ValueError:
                # Candidate might have been removed already if enqueued twice; best effort only
                pass

            # Mark as inflight
            self._inflight_fingerprints.add(fingerprint)

            # Launch the evaluation
            launched_successfully = await self._stream_launch(candidate, window_id)

            if launched_successfully:
                launched += 1
            else:
                # Launch failed, put candidate back on queue
                self._inflight_fingerprints.discard(fingerprint)
                self._pending_fingerprints.add(fingerprint)
                self.queue.append(candidate)
                heapq.heappush(self._priority_queue, (neg_priority, neg_rung_idx, counter, candidate))
                break  # Stop trying to launch more

        return launched

    async def _stream_launch(self, candidate: Candidate, window_id: int) -> bool:
        import time

        if len(self._runtime_shards) == 0:
            return False

        shard_idx = min(self.scheduler.current_shard_index(candidate), len(self._runtime_shards) - 1)

        # REMOVED: No orchestrator-level concurrency limits
        # Only eval_concurrency (in evaluator) + straggler detection control throughput

        shard_fraction = self._runtime_shards[min(shard_idx, len(self._runtime_shards) - 1)]
        shard_key = (window_id, shard_fraction)
        shard_ids = self._shard_cache.get(shard_key)
        if shard_ids is None:
            shard_size = self._shard_size(shard_fraction)
            shard_ids = self.sampler.sample_shard(window_id, shard_size)
            self._shard_cache[shard_key] = shard_ids
        # Track assigned examples for coverage accounting
        self._candidate_eval_examples[candidate.fingerprint] = list(shard_ids)

        # Determine per-candidate concurrency and respect a global budget to optimize throughput
        desired = min(self._effective_concurrency, len(shard_ids))
        per_cand_concurrency = desired
        if self.config.global_concurrency_budget:
            available = max(0, self._effective_concurrency - self._examples_inflight)
            if available <= 0:
                # Not enough global budget right now – defer launch
                # Put candidate back on priority queue and try later
                self._pending_fingerprints.add(candidate.fingerprint)
                self.queue.append(candidate)
                rung_idx = self.scheduler.current_shard_index(candidate)
                priority = self._compute_priority(candidate, rung_idx)
                self._priority_counter += 1
                heapq.heappush(self._priority_queue, (-priority, -rung_idx, self._priority_counter, candidate))
                return False
            if available < per_cand_concurrency:
                per_cand_concurrency = max(1, available)
                if self.show_progress:
                    self.logger.log(
                        f"🔧 Global budget clamp: concurrency {desired}→{per_cand_concurrency} (inflight={self._examples_inflight}/{self._effective_concurrency})"
                    )
                self.metrics.record_budget_clamp()

        # Remove example-level oversubscription check entirely - trust straggler detection
        # to handle any tasks that run too long. This maximizes throughput.

        # Track peak example-level concurrency for metrics
        peak_examples = self._examples_inflight + per_cand_concurrency
        self.metrics.update_concurrent_evals(peak_examples)

        time.time()
        cand_hash = candidate_key(candidate)
        self._total_inflight += 1
        # Track inflight per rung for deficit scheduler (removed with priority queue)
        # IMPORTANT: Use the shard_idx passed to this function, NOT _get_rung_key()!
        # The candidate may be promoted during evaluation, changing its shard index.
        # We must use the EXACT shard fraction this launch is evaluating on.
        rung_key_at_launch = str(shard_fraction)  # Use the actual shard fraction for this evaluation
        fingerprint_at_launch = candidate.fingerprint
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
                is_final_shard = shard_idx == len(self._runtime_shards) - 1
                result = await self.evaluator.eval_on_shard(
                    candidate,
                    shard_ids,
                    concurrency=per_cand_concurrency,
                    shard_fraction=shard_fraction,
                    show_progress=self.show_progress,  # Show progress for all evaluations
                    is_final_shard=is_final_shard,
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
                self._total_inflight -= 1
                # Decrement rung inflight using the SAME key we incremented with
                # Don't recalculate - the candidate may have been promoted while evaluating!
                # Use get() with default 0 to avoid KeyError if key was removed during recompute
                # Release global example-level budget
                self._examples_inflight = max(0, self._examples_inflight - per_cand_concurrency)
                current = self._inflight_by_rung.get(shard_idx, 0)
                if current <= 1:
                    self._inflight_by_rung.pop(shard_idx, None)
                else:
                    self._inflight_by_rung[shard_idx] = current - 1
                # Clear inflight fingerprint using the captured value from launch time
                # This prevents the candidate from being launched again until this evaluation completes
                self._inflight_fingerprints.discard(fingerprint_at_launch)

        self._inflight_by_rung[shard_idx] += 1
        if shard_idx == self._final_rung_index:
            current = self._inflight_by_rung[shard_idx]
            if current > self._final_inflight_peak:
                self._final_inflight_peak = current
            if self.show_progress:
                limit = self.config.max_final_shard_inflight or "∞"
                self.logger.log(
                    f"🚀 Final rung launch: inflight={current}/{limit} [fp={candidate.fingerprint[:12]}...]",
                    LogLevel.WARNING,
                )
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

        meta["recent_delta"] = delta_quality
        prev_avg = meta.get("recent_delta_avg")
        if isinstance(prev_avg, (int, float)):
            avg_delta = 0.5 * prev_avg + 0.5 * delta_quality
        else:
            avg_delta = delta_quality
        meta["recent_delta_avg"] = avg_delta
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

        # Update EMA tracking for priority scheduling
        if 0 <= prev_idx < len(self._rung_launches):
            self._rung_launches[prev_idx] += 1

        if decision in ("promoted", "completed") and 0 <= prev_idx < len(self._rung_promotions):
            self._rung_promotions[prev_idx] += 1

            # Update promotion EMA for this rung
            if prev_idx < len(self._promotion_ema):
                promoted = 1.0
                self._promotion_ema[prev_idx] = self._update_ema(self._promotion_ema[prev_idx], promoted)
        elif decision == "pruned" and 0 <= prev_idx < len(self._promotion_ema):
            # Update with failure
            promoted = 0.0
            self._promotion_ema[prev_idx] = self._update_ema(self._promotion_ema[prev_idx], promoted)

        # Update final success EMA when candidates complete final rung
        if decision == "completed":
            at_final_rung = prev_idx >= len(self._runtime_shards) - 1
            if at_final_rung:
                # Completed at final rung = success
                self._final_success_ema = self._update_ema(self._final_success_ema, 1.0)

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
            self.metrics.update_archive_sizes(pareto_size)

        self._register_failures(result)

        promotions = self.scheduler.promote_ready()
        if promotions:
            # Add promoted candidates to priority queue
            for p in promotions:
                rung_idx = self.scheduler.current_shard_index(p)
                priority = self._compute_priority(p, rung_idx)

                # Use negative priority for max-heap (heapq is min-heap)
                # Secondary sort by negative rung_idx (prefer higher rungs)
                # Tertiary sort by counter for FIFO tie-breaking
                self._priority_counter += 1
                heapq.heappush(self._priority_queue, (-priority, -rung_idx, self._priority_counter, p))

                # Optional: log promotion
                if self.show_progress:
                    shard_frac = self._runtime_shards[rung_idx] if rung_idx < len(self._runtime_shards) else 1.0
                    generation = self._get_generation(p)
                    source = "seed" if generation == 0 else f"gen-{generation if generation is not None else '?'}"
                    self.logger.log(
                        f"   ⬆️  {source} promoted to shard {rung_idx} ({shard_frac:.0%}) priority={priority:.4f} [fp={p.fingerprint[:12]}...]"
                    )

            if self.show_progress:
                self.logger.log(f"   📋 Priority queue size after promotion: {len(self._priority_queue)}")

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
        expected_ids = self._candidate_eval_examples.pop(candidate.fingerprint, [])
        shard_fraction = self._runtime_shards[min(shard_idx, len(self._runtime_shards) - 1)]
        if expected_ids:
            seen_ids = set(result.example_ids or [])
            missing = [ex for ex in expected_ids if ex not in seen_ids]
            if missing:
                if self.config.straggler_grace_seconds:
                    grace = self.config.straggler_grace_seconds
                    recovered = await self.evaluator.collect_straggler_results(
                        candidate,
                        missing,
                        timeout=grace,
                    )
                    for recovered_result in recovered:
                        result = result.merge(recovered_result)
                        if recovered_result.example_ids:
                            seen_ids.update(recovered_result.example_ids)
                    if recovered:
                        missing = [ex for ex in expected_ids if ex not in seen_ids]
                        if self.show_progress:
                            self.logger.log(
                                f"📦 Recovered {len(recovered)} straggler example(s) after {grace:.1f}s grace window"
                            )
                missing = [ex for ex in expected_ids if ex not in seen_ids]
            if missing:
                self.evaluator.discard_straggler_results(candidate, missing)
                self.logger.log(
                    f"♻️  Replaying {len(missing)} missing examples on shard {shard_idx} ({shard_fraction:.0%}) "
                    f"for candidate {candidate.fingerprint[:12]}..."
                )
                replay = await self.evaluator.eval_on_shard(
                    candidate,
                    missing,
                    concurrency=min(self._effective_concurrency, max(1, len(missing))),
                    shard_fraction=shard_fraction,
                    show_progress=self.show_progress,
                    is_final_shard=shard_idx == len(self._runtime_shards) - 1,
                )
                result = result.merge(replay)
        self._update_eval_metrics(candidate, result)
        candidate_with_meta, decision = await self._ingest_result(candidate, result)

        shard_fraction = self._runtime_shards[min(shard_idx, len(self._runtime_shards) - 1)]
        quality = result.objectives.get(self.config.promote_objective, 0.0)
        self._record_best_quality(quality, shard_fraction)

        # STREAMING: Trigger mutation worker immediately after successful evaluation
        # This eliminates idle gap between evaluation completion and next mutations
        if decision in ("promoted", "completed"):
            if self._mutation_task is None or self._mutation_task.done():
                self._mutation_task = asyncio.create_task(self._streaming_mutation_worker())

    async def _stream_check_target(self, window_id: int) -> bool:
        if self.config.target_quality is None:
            return False

        pareto = self.archive.pareto_entries()
        if not pareto:
            return False

        full_shard = self._runtime_shards[-1]
        tolerance = 1e-6
        full_entries = [
            entry
            for entry in pareto
            if entry.result.shard_fraction is not None
            and abs(entry.result.shard_fraction - full_shard) <= tolerance
        ]
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
            self._record_best_quality(quality, shard_fraction)
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
                f"priority_queue={len(self._priority_queue)}"
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
        tolerance = 1e-6

        # Try to get quality from full-shard evaluations first
        full_shard_quality = 0.0
        has_full_shard = False
        for entry in pareto:
            shard_fraction = entry.result.shard_fraction
            if shard_fraction is not None and abs(shard_fraction - full_shard) <= tolerance:
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

    async def _streaming_mutation_worker(self) -> None:
        """
        SIMPLIFIED streaming mutation worker.
        Kicks off all mutations concurrently, enqueues each as it completes.
        No waiting, no batching - pure streaming.
        """
        try:
            # Check budget
            if self.max_evaluations is not None and self.evaluations_run >= self.max_evaluations:
                return

            entries = self.archive.pareto_entries()
            if not entries:
                return

            num_mutations = self._mutation_budget()
            if num_mutations <= 0:
                return

            # Mark new generation starting for convergence tracking
            # Track which rungs we're generating from
            rungs_generating = set()
            for entry in entries:
                rung_idx = self.scheduler.current_shard_index(entry.candidate)
                rungs_generating.add(rung_idx)

            # Adaptive mutation quota: bias budget towards higher rungs
            max_rung_idx = max(rungs_generating) if rungs_generating else 0
            rung_factor = 1.0
            if len(self._runtime_shards) > 1:
                if max_rung_idx == 0:
                    rung_factor = 0.6  # Reduce churn when only rung 0 has parents
                else:
                    total_progress_rungs = max(1, len(self._runtime_shards) - 1)
                    progress_ratio = max_rung_idx / total_progress_rungs
                    rung_factor = 1.0 + 0.5 * progress_ratio  # Up to +50% when final rung active

            adjusted_mutations = max(self._mutation_min, int(round(num_mutations * rung_factor)))
            if self.config.max_mutations_per_round:
                adjusted_mutations = min(self.config.max_mutations_per_round, adjusted_mutations)

            if adjusted_mutations != num_mutations and self.show_progress:
                self.logger.log(
                    f"   🎯 Mutation budget adjusted by rung factor ({rung_factor:.2f}): {num_mutations} → {adjusted_mutations}"
                )
            num_mutations = adjusted_mutations

            # Mark generation start for each rung we're mutating from
            for rung_idx in rungs_generating:
                self.scheduler.mark_generation_start(rung_idx)

            self.logger.log(f"🧬 Streaming {num_mutations} mutations from {len(entries)} parents (rungs: {sorted(rungs_generating)})")

            # Generate mutations - they stream directly to queue via candidate_sink
            try:
                mutations = await self._generate_mutations_batched(entries, num_mutations)
            except Exception as e:
                self.logger.log(f"⚠️  Mutation generation failed: {e}")
                return

            if not mutations:
                return

            # Note: Candidates already enqueued via streaming sink - just record metrics
            self._record_mutation_enqueued(len(mutations))
            if self.show_progress:
                self.logger.log(f"   ✅ {len(mutations)} mutations streamed to queue (total enqueued: {self._mutations_enqueued})")
        finally:
            # CRITICAL: Clear task handle so next evaluation can trigger a new worker
            # Without this, the handle points to a done task and new workers won't spawn correctly
            self._mutation_task = None

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
                f"[SPAWN_MUTATIONS] budget={num_mutations}, queue={len(self.queue)} - skipping",
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
            f"🧭 spawn_mutations: parents={len(entries)} budget={num_mutations} queue={len(self.queue)}"
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

        # Parent selection: top quality candidates from Pareto frontier
        # Select 3-5 parents to give reflection LLM rich context
        parent_select_start = time.time()
        num_parents = min(5, max(3, len(entries)))

        # Get top performers by quality
        sorted_entries = sorted(
            entries,
            key=lambda e: (
                e.candidate.meta.get("recent_delta", 0.0),
                e.result.objectives.get(self.config.promote_objective, float("-inf")),
            ),
            reverse=True,
        )
        selected_entries = sorted_entries[:num_parents]

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

            # Derive scheduler key for this parent candidate
            try:
                parent_sched_key = self.scheduler._sched_key(entry.candidate)  # type: ignore[attr-defined]
            except Exception:
                parent_sched_key = entry.candidate.fingerprint

            # Collect per‑rung parent scores from scheduler history, keyed by rung fraction
            parent_rung_scores: dict[float, float] = {}
            try:
                shards = getattr(self.scheduler, "shards", list(self.config.shards))
                rung_scores = getattr(self.scheduler, "_rung_scores", {})  # type: ignore[attr-defined]
                for idx, frac in enumerate(shards):
                    val = rung_scores.get((parent_sched_key, idx))
                    if isinstance(val, (int, float)):
                        parent_rung_scores[float(frac)] = float(val)
            except Exception:
                parent_rung_scores = {}

            candidate_with_parent = entry.candidate.with_meta(
                parent_objectives=parent_objectives,
                parent_rung_scores=parent_rung_scores,
                _sched_key=parent_sched_key,
            )
            parent_meta = candidate_with_parent.meta if isinstance(candidate_with_parent.meta, dict) else {}
            parent_key = parent_sched_key

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

        # STREAMING: Provide sink to enqueue candidates immediately as they're generated
        async def stream_candidate_to_queue(candidate: Candidate) -> None:
            """Called by mutator for each candidate as soon as it's created."""
            # Enqueue immediately - this feeds the priority queue while mutations are still generating
            self.enqueue([candidate])
            if self.show_progress:
                generation = self._get_generation(candidate)
                source = "seed" if generation == 0 else f"gen-{generation if generation is not None else '?'}"
                self.logger.log(f"   ✨ STREAMED {source} to queue (fp={candidate.fingerprint[:12]}...)")

        mutations = await self.mutator.propose(
            parent_contexts,
            num_mutations,
            task_examples=task_examples,
            candidate_sink=stream_candidate_to_queue,
        )
        mutator_duration = time.time() - mutator_start
        self.metrics.record_mutation_batch(len(mutations), mutator_duration)
        self.metrics.time_mutation_total += mutator_duration
        self._record_mutation_generation(num_mutations, mutations)

        batch_duration = time.time() - batch_start
        self.metrics.time_scheduler_total += batch_duration - mutator_duration  # Other overhead

        return mutations  # Still return for metrics tracking, but candidates already enqueued via streaming

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
        if self.show_progress:
            self._log_operator_stats()

    def _log_operator_stats(self) -> None:
        stats = getattr(self.mutator, "_operator_stats", None)
        if not stats:
            return
        summary: list[str] = []
        for op, data in stats.items():
            trials = data.get("trials", 0)
            if not trials:
                continue
            promoted = data.get("promoted", 0)
            delta_sum = data.get("delta_sum", 0.0)
            avg = delta_sum / max(1, trials)
            summary.append(f"{op}: {promoted}/{trials} ({avg:+.3f})")
        if summary:
            self.logger.log("   📈 Operator stats: " + "; ".join(summary[:4]))

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
            "strategy_stats": getattr(self.mutator, "strategy_stats", {}),
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

        # Pareto candidates are already included above

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
        """Compute how many new mutations we should request this cycle based on queue deficit."""
        max_mut = self.config.max_mutations_per_round or 0
        if max_mut <= 0:
            return 0

        if self._mutation_throttle and max_mut > self._mutation_min:
            max_mut = max(self._mutation_min, max_mut // 2)

        current_queue_depth = len(self._priority_queue)
        target_queue_depth = int(self._queue_buffer_mult * self.config.eval_concurrency)
        deficit = target_queue_depth - current_queue_depth

        if deficit <= 0:
            import time as _time_budget

            now = _time_budget.time()
            if now - self._last_budget_log > 5.0:
                self.logger.log(
                    f"🧭 Debug: mutation budget 0 "
                    f"(queue_depth={current_queue_depth} target={target_queue_depth})"
                )
                self._last_budget_log = now
            return 0

        # SIMPLE: Request just enough mutations to fill the deficit (clamped to budget)
        return min(max_mut, max(deficit, self._mutation_min))

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
        is_final_shard = shard_fraction == self._runtime_shards[-1]
        result = await self.evaluator.eval_on_shard(
            candidate,
            shard,
            concurrency=concurrency,
            shard_fraction=shard_fraction,
            show_progress=show_progress,
            is_final_shard=is_final_shard,
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

        # Run save in background - don't block the optimization loop
        self._save_task = asyncio.create_task(
            asyncio.to_thread(
                self.cache.save_state,
                round_num=self.round_index,
                evaluations=self.evaluations_run,
                pareto_candidates=pareto_candidates,
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
                    is_final_shard=True,
                )
                await self.archive.insert(candidate, result)

        # Restore Pareto candidates with controlled parallelism
        pareto_candidates = state.get("pareto", [])
        if pareto_candidates:
            await asyncio.gather(*(restore_candidate(c) for c in pareto_candidates))

        # Restore queue / priority queue
        queued_candidates = state.get("queue", [])
        self.queue = deque()
        self._priority_queue.clear()
        self._pending_fingerprints.clear()
        # Reset counter so restored items preserve relative ordering
        self._priority_counter = 0
        for candidate in queued_candidates:
            if not candidate.text.strip():
                continue
            fingerprint = candidate.fingerprint
            if fingerprint in self._pending_fingerprints or fingerprint in self._inflight_fingerprints:
                continue
            rung_idx = self.scheduler.current_shard_index(candidate)
            priority = self._compute_priority(candidate, rung_idx)
            self._priority_counter += 1
            heapq.heappush(self._priority_queue, (-priority, -rung_idx, self._priority_counter, candidate))
            self._pending_fingerprints.add(fingerprint)
            self.queue.append(candidate)
    def _detect_fd_guard(self) -> int | None:
        try:
            import resource

            soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            guard = int(max(32, soft * 0.5))
            return guard
        except Exception:
            return None

    def _clamp_inflight(self, proposed: int) -> int:
        limit = max(1, proposed)
        if self._fd_guard_limit is not None and limit > self._fd_guard_limit:
            limit = self._fd_guard_limit
            if self.show_progress:
                self.logger.log(
                    f"🔒 FD guard limiting inflight to {limit} (soft limit {self._fd_guard_limit})",
                    LogLevel.WARNING,
                )
        return limit
