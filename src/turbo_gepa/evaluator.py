"""
Async evaluator responsible for running candidate evaluations on shards.

The evaluator coordinates cache lookups, validator checks, and concurrent
execution of user-provided LLM calls. It retains only stdlib dependencies so
we can run quickly in constrained environments.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, Sequence, Any

from turbo_gepa.logging.logger import LoggerProtocol, StdOutLogger, LogLevel

from .cache import DiskCache
from .interfaces import Candidate, EvalResult

if TYPE_CHECKING:
    from .metrics import Metrics

Validator = Callable[[Candidate], None]
MetricsMapper = Callable[[dict[str, float]], dict[str, float]]
TaskRunner = Callable[[Candidate, str], Awaitable[dict[str, float]]]


class AsyncEvaluator:
    """Concurrent evaluator with disk-backed caching."""

    def __init__(
        self,
        cache: DiskCache,
        task_runner: TaskRunner,
        validators: Iterable[Validator] | None = None,
        metrics_mapper: MetricsMapper | None = None,
        verbose_errors: bool = False,
        logger: LoggerProtocol | None = None,
        timeout_seconds: float | None = None,
        min_improve: float = 0.0,
        metrics: Metrics | None = None,
        skip_final_straggler_cutoff: bool = False,
        promote_objective: str = "quality",
        cancel_stragglers_immediately: bool = True,
        replay_stragglers: bool = True,
        min_samples_for_confidence: int = 20,
        target_quality: float | None = None,
        confidence_z: float | None = None,
    ) -> None:
        self.cache = cache
        self.task_runner = task_runner
        self.validators = list(validators or [])
        self.metrics_mapper = metrics_mapper or (lambda metrics: metrics)
        self.verbose_errors = verbose_errors
        self._inflight_examples: int = 0
        self._max_observed_inflight: int = 0
        self.logger: LoggerProtocol = logger or StdOutLogger()
        self.timeout_seconds = timeout_seconds
        self.min_improve = float(min_improve)
        self.metrics = metrics
        self.skip_final_straggler_cutoff = skip_final_straggler_cutoff
        self.promote_objective = promote_objective
        # Track detached stragglers so orchestrator can await their results later
        self.cancel_stragglers_immediately = bool(cancel_stragglers_immediately)
        self.replay_stragglers = bool(replay_stragglers)
        self._use_straggler_futures = not self.cancel_stragglers_immediately and self.replay_stragglers
        self._straggler_results: dict[tuple[str, str], asyncio.Future[EvalResult]] = {}
        self._tail_ratio: float = 1.0
        self.min_samples_for_confidence = max(1, int(min_samples_for_confidence))
        self.target_quality = target_quality
        # Confidence margin (in standard errors) used for lower-bound checks on
        # final-rung early success. When zero or None we fall back to using the
        # plain running mean as before.
        self.confidence_z: float = float(confidence_z) if confidence_z is not None else 0.0

    async def eval_on_shard(
        self,
        candidate: Candidate,
        example_ids: Sequence[str],
        concurrency: int,
        shard_fraction: float | None = None,
        show_progress: bool = False,
        early_stop_fraction: float = 0.9,  # Return after 90% complete
        is_final_shard: bool = False,
        *,
        coverage_target: float | None = None,
        on_partial: Callable[[Candidate, EvalResult], Awaitable[None]] | None = None,
        partial_min_samples: int = 2,
        partial_interval_seconds: float = 1.5,
    ) -> EvalResult:
        """
        Evaluate ``candidate`` on ``example_ids`` with a concurrency cap.

        Cached traces are reused automatically, and only cache misses trigger
        fresh model calls.
        """
        for validator in self.validators:
            validator(candidate)

        import time

        # Compute rung-aware parent baseline for early-stop.
        # Prefer an explicit parent rung score if provided via meta; otherwise
        # shrink the parent's final score toward a global baseline using a
        # rung-dependent alpha (kept in sync with the scheduler defaults).
        def _parent_baseline_for_rung() -> float | None:
            meta = candidate.meta if isinstance(candidate.meta, dict) else None
            if not isinstance(meta, dict):
                return None

            # 1) If caller provided rung-specific parent scores, prefer those.
            parent_rung_scores = meta.get("parent_rung_scores")
            if isinstance(parent_rung_scores, dict) and shard_fraction is not None:
                val = parent_rung_scores.get(shard_fraction)
                if isinstance(val, (int, float)):
                    return float(val)
                # If parent never saw this shard, don't gate; defer to scheduler
                return None

            # 2) Fall back to shrinkage of the final parent score toward a neutral baseline.
            parent_objectives = meta.get("parent_objectives")
            promote_key = getattr(self, "promote_objective", "quality")
            parent_final = None
            if isinstance(parent_objectives, dict):
                parent_final = parent_objectives.get(promote_key)
                if parent_final is None and promote_key != "quality":
                    parent_final = parent_objectives.get("quality")
            if not isinstance(parent_final, (int, float)):
                return None
            if shard_fraction is None:
                return None
            alpha = meta.get("parent_shrinkage_alpha")
            if not isinstance(alpha, (int, float)):
                try:
                    if shard_fraction <= 0:
                        alpha = 0.0
                    else:
                        alpha = shard_fraction ** 0.3
                except Exception:
                    alpha = 0.0
            alpha = max(0.0, min(1.0, float(alpha)))
            baseline_anchor = meta.get("baseline_quality", 0.5)
            if not isinstance(baseline_anchor, (int, float)):
                baseline_anchor = 0.5
            baseline_anchor = float(baseline_anchor)
            parent_final = float(parent_final)
            return baseline_anchor + alpha * (parent_final - baseline_anchor)

        parent_target: float | None = None
        parent_baseline = _parent_baseline_for_rung()
        if isinstance(parent_baseline, (int, float)) and not (is_final_shard and self.skip_final_straggler_cutoff):
            parent_target = float(parent_baseline) + float(self.min_improve)
            if parent_target > 1.0:
                parent_target = 1.0
            if parent_target < 0.0:
                parent_target = 0.0

        semaphore = asyncio.Semaphore(max(concurrency, 1))
        results: list[EvalResult] = []
        completed = 0
        total = len(example_ids)
        batch_start_time = time.time()
        eval_durations: list[float] = []  # Track how long each eval took (excluding cached)
        quality_lock = asyncio.Lock()
        running_quality = 0.0
        running_sq = 0.0  # Sum of squared qualities (approximate, batch-weighted)
        early_stop_flag = False
        early_stop_reason: str | None = None
        straggler_detached_total = 0
        candidate_fp = candidate.fingerprint
        deliver_flags: dict[str, bool] = {ex_id: True for ex_id in example_ids}
        loop = asyncio.get_running_loop()
        latency_ema: dict[str, float] = {}
        latency_samples: dict[str, int] = {}
        # Partial publishing accumulators
        partial_sum: dict[str, float] = {}
        partial_n: int = 0
        partial_example_ids: list[str] = []
        last_partial_publish: float = 0.0
        min_detach_samples = 0

        def _required_detach_samples() -> int:
            nonlocal min_detach_samples
            if min_detach_samples:
                return min_detach_samples
            if total <= 0:
                min_detach_samples = 0
                return 0
            # With coverage floors removed in fast profiles, we allow the
            # straggler detector to operate purely on latency statistics.
            # As soon as at least one example has completed, slow tasks can
            # be detached when they exceed the dynamic threshold.
            min_detach_samples = 0
            return min_detach_samples

        # CI-based early-stop is intentionally disabled for now. Between the
        # parent-target bound below and the shard-level straggler logic we
        # already have sufficient safeguards, and keeping this simple avoids
        # surprising behaviour when the underlying latency distribution shifts.

        def _ensure_straggler_future(example_id: str) -> asyncio.Future[EvalResult] | None:
            if not self._use_straggler_futures:
                return None
            key = (candidate_fp, example_id)
            future = self._straggler_results.get(key)
            if future is None or future.done():
                future = loop.create_future()
                self._straggler_results[key] = future
            return future

        async def _deliver_result(
            result: EvalResult,
            quality_override: float | None,
            example_id: str,
        ) -> None:
            if deliver_flags.get(example_id, True):
                await _register_result(result, quality_override)
                return

            if not self._use_straggler_futures:
                await _register_result(result, quality_override)
                return

            future_key = (candidate_fp, example_id)
            future = self._straggler_results.get(future_key)
            if future is None:
                await _register_result(result, quality_override)
                return
            if not future.done():
                future.set_result(result)
            else:
                await _register_result(result, quality_override)

        async def _register_result(result: EvalResult, quality_override: float | None = None) -> None:
            # Update outer-scope accumulators for partial publishing and gating.
            # We mutate these counters here, so declare them nonlocal.
            nonlocal completed, running_quality, running_sq, early_stop_flag, partial_n, last_partial_publish, early_stop_reason
            async with quality_lock:
                results.append(result)
                completed += result.n_examples
                # Accumulate for partial publishing
                for k, v in result.objectives.items():
                    if isinstance(v, (int, float)):
                        partial_sum[k] = partial_sum.get(k, 0.0) + float(v) * max(1, result.n_examples)
                partial_n += result.n_examples
                if result.example_ids:
                    partial_example_ids.extend(list(result.example_ids))
                q_val = quality_override
                if q_val is None:
                    obj_quality = result.objectives.get(self.promote_objective) if isinstance(result.objectives, dict) else None
                    if isinstance(obj_quality, (int, float)):
                        q_val = float(obj_quality)
                if isinstance(q_val, (int, float)):
                    q_val_f = float(q_val)
                    n = max(1, result.n_examples)
                    running_quality += q_val_f * n
                    running_sq += (q_val_f * q_val_f) * n
                if (
                    not early_stop_flag
                    and parent_target is not None
                    and total > 0
                ):
                    remaining = total - completed
                    if remaining < 0:
                        remaining = 0
                    best_possible = (running_quality + remaining * 1.0) / total
                    if best_possible + 1e-9 < parent_target:
                        early_stop_flag = True
                        # Track early stopping event
                        if self.metrics:
                            self.metrics.record_early_stop("parent_target", is_final_rung=is_final_shard)
                        if show_progress:
                            self.logger.log(
                                f"⚠️ Early stop: candidate {candidate.fingerprint[:12]} cannot beat parent target {parent_target:.1%}"
                            )
                # Simple early-stop for success: once we've seen enough samples
                # and the running mean (or its confidence lower bound on the
                # final rung) clears the configured target_quality, there is no
                # need to wait for remaining examples unless a stricter policy
                # is desired.
                if (
                    not early_stop_flag
                    and self.target_quality is not None
                    and total > 0
                    and completed >= max(1, self.min_samples_for_confidence)
                ):
                    running_mean = running_quality / max(completed, 1)
                    # Default lower bound is the mean itself; for the final rung
                    # we optionally subtract a configurable number of standard
                    # errors derived from verification_speed_bias.
                    lb = running_mean
                    if is_final_shard and self.confidence_z > 0.0 and completed >= 2:
                        try:
                            import math

                            mean_sq = running_sq / max(completed, 1)
                            variance = max(0.0, mean_sq - running_mean * running_mean)
                            if variance > 0.0:
                                se = math.sqrt(variance / max(completed, 1))
                                lb = running_mean - self.confidence_z * se
                        except Exception:
                            # If anything goes wrong computing the bound, fall
                            # back to the plain mean-based check.
                            lb = running_mean
                    if lb >= self.target_quality:
                        early_stop_flag = True
                        early_stop_reason = "target_quality"
                        if self.metrics:
                            self.metrics.record_early_stop("target_quality", is_final_rung=is_final_shard)
                        if show_progress:
                            self.logger.log(
                                f"⚡ Early success: candidate {candidate.fingerprint[:12]} "
                                f"cleared target_quality {self.target_quality:.1%} "
                                f"(mean={running_mean:.1%} on {completed} examples)"
                            )
                # Partial publish
                if on_partial is not None and total > 0:
                    now = time.time()
                    if partial_n >= max(1, partial_min_samples) and (now - last_partial_publish) >= partial_interval_seconds:
                        averaged = {k: v / max(partial_n, 1) for k, v in partial_sum.items()}
                        partial = EvalResult(
                            objectives=averaged,
                            traces=[],
                            n_examples=partial_n,
                            shard_fraction=shard_fraction,
                            example_ids=list(partial_example_ids),
                            coverage_fraction=float(completed) / float(total) if total else 0.0,
                        )
                        try:
                            await on_partial(candidate, partial)
                        except Exception:
                            pass
                        last_partial_publish = now

        def _record_latency(example_id: str, duration: float) -> None:
            prev = latency_ema.get(example_id)
            count = latency_samples.get(example_id, 0)
            alpha = 0.2 if count else 1.0
            base = prev if prev is not None else duration
            latency_ema[example_id] = (1 - alpha) * base + alpha * duration
            latency_samples[example_id] = count + 1

        async def eval_one(example_id: str, task_start_time: float) -> None:
            nonlocal completed

            cached = await self.cache.get(candidate, example_id)
            if cached:
                # Track cache hit
                if self.metrics:
                    self.metrics.record_cache_lookup(hit=True)
                quality_val = None
                if isinstance(cached.objectives, dict):
                    q = cached.objectives.get(self.promote_objective)
                    if q is None:
                        q = cached.objectives.get("quality")
                    if isinstance(q, (int, float)):
                        quality_val = float(q)
                await _deliver_result(cached, quality_val, example_id)
                if show_progress and total > 0:
                    # Debug-level per-example progress; round-level stats are handled elsewhere.
                    try:
                        pct = completed / max(total, 1) * 100
                    except Exception:
                        pct = 0.0
                    self.logger.log(
                        f"Progress: {completed}/{total} examples ({pct:.0f}%)",
                        LogLevel.DEBUG,
                    )
                return

            # Track cache miss
            if self.metrics:
                self.metrics.record_cache_lookup(hit=False)

            try:
                # Per-example launch logging kept at a low level to avoid spam.
                self.logger.log(
                    f"🔄 Starting eval for example {example_id} at t={time.time() - batch_start_time:.1f}s (inflight: {self._inflight_examples})",
                    LogLevel.DEBUG,
                )

                async with semaphore:
                    self._inflight_examples += 1
                    if self._inflight_examples > self._max_observed_inflight:
                        self._max_observed_inflight = self._inflight_examples

                    _start_api = time.time()
                    task = self.task_runner(candidate, example_id)
                    if self.timeout_seconds is not None:
                        metrics = await asyncio.wait_for(task, timeout=self.timeout_seconds)
                    else:
                        metrics = await task
                    _elapsed_api = time.time() - _start_api

                    # Completion timing is useful for debugging but too verbose for high-level progress.
                    self.logger.log(
                        f"✅ Completed eval for example {example_id} in {_elapsed_api:.1f}s at t={time.time() - batch_start_time:.1f}s",
                        LogLevel.DEBUG,
                    )

                # Ensure inflight counter is decremented even if mapper raises
                self._inflight_examples = max(0, self._inflight_examples - 1)
                mapped = self.metrics_mapper(metrics)
                _record_latency(example_id, _elapsed_api)
                # Build a lean trace to avoid heavy I/O; keep only fields used by reflection
                max_len = 2048
                trace: dict[str, object] = {"example_id": example_id}
                # Always keep objective metric and tokens if present
                obj_key = self.promote_objective
                obj_val = metrics.get(obj_key)
                if obj_val is None and obj_key != "quality":
                    obj_val = metrics.get("quality")
                if obj_val is not None:
                    trace[obj_key] = obj_val
                if obj_key != "quality" and "quality" in metrics:
                    trace["quality"] = metrics.get("quality")
                if "tokens" in metrics:
                    trace["tokens"] = metrics.get("tokens")
                # Keep input and expected answer for feedback context
                if "input" in metrics:
                    trace["input"] = metrics.get("input")
                if "expected_answer" in metrics:
                    trace["expected_answer"] = metrics.get("expected_answer")
                if "additional_context" in metrics:
                    trace["additional_context"] = metrics.get("additional_context")
                # Prefer a single output field; if only 'response' exists, map it to 'output'
                raw_output = None
                if "output" in metrics and isinstance(metrics.get("output"), str):
                    raw_output = metrics.get("output")
                elif "response" in metrics and isinstance(metrics.get("response"), str):
                    raw_output = metrics.get("response")
                if isinstance(raw_output, str):
                    out = raw_output
                    if len(out) > max_len:
                        out = out[: max_len] + "…"
                    trace["output"] = out
                result = EvalResult(
                    objectives=mapped,
                    traces=[trace],
                    n_examples=1,
                    shard_fraction=shard_fraction,
                    example_ids=[example_id],
                )
                await self.cache.set(candidate, example_id, result)
                # Track cache write
                if self.metrics:
                    self.metrics.record_cache_write()
                quality_val = None
                if isinstance(mapped, dict):
                    val = mapped.get(self.promote_objective)
                    if val is None:
                        val = mapped.get("quality")
                    if isinstance(val, (int, float)):
                        quality_val = float(val)
                await _deliver_result(result, quality_val, example_id)

                # Track duration for non-cached evals
                eval_duration = time.time() - task_start_time
                eval_durations.append(eval_duration)

                if show_progress:
                    self.logger.log(f"Progress: {completed}/{total} examples ({completed / max(total, 1) * 100:.0f}%)")
            except asyncio.TimeoutError as e:
                self._inflight_examples = max(0, self._inflight_examples - 1)
                timeout_msg = (
                    f"⚠️  Evaluation timed out for example {example_id} "
                    f"after {self.timeout_seconds:.1f}s" if self.timeout_seconds else
                    f"⚠️  Evaluation timed out for example {example_id}"
                )
                if show_progress:
                    self.logger.log(timeout_msg)
                fallback_metrics = {
                    self.promote_objective: 0.0,
                    "quality": 0.0,
                    "neg_cost": 0.0,
                    "tokens": 0.0,
                }
                _record_latency(example_id, self.timeout_seconds or _elapsed_api)
                mapped = self.metrics_mapper(fallback_metrics)
                trace = dict(fallback_metrics)
                trace["example_id"] = example_id
                trace["error"] = "timeout"
                result = EvalResult(
                    objectives=mapped,
                    traces=[trace],
                    n_examples=1,
                    shard_fraction=shard_fraction,
                    example_ids=[example_id],
                )
                # Don't cache timeouts - allow retry next run
                await _deliver_result(result, 0.0, example_id)
            except Exception as e:
                self._inflight_examples = max(0, self._inflight_examples - 1)
                # Handle task runner failures gracefully
                # Return zero scores to avoid crashing the entire batch
                error_msg = f"⚠️  Evaluation failed for example {example_id}: {type(e).__name__}: {str(e)[:100]}"
                if self.verbose_errors:
                    self.logger.log(error_msg)
                elif show_progress:
                    # Always log failures when show_progress is on
                    self.logger.log(error_msg)
                fallback_metrics = {
                    self.promote_objective: 0.0,
                    "quality": 0.0,
                    "neg_cost": 0.0,
                    "tokens": 0.0,
                }
                mapped = self.metrics_mapper(fallback_metrics)
                trace = dict(fallback_metrics)
                trace["example_id"] = example_id
                trace["error"] = str(e)
                result = EvalResult(
                    objectives=mapped,
                    traces=[trace],
                    n_examples=1,
                    shard_fraction=shard_fraction,
                    example_ids=[example_id],
                )
                # Don't cache failed evaluations - allow retry on next run
                await _deliver_result(result, 0.0, example_id)

        # Launch all tasks
        current_time = time.time()
        pending: dict[asyncio.Task, dict[str, Any]] = {}
        for ex_id in example_ids:
            task = asyncio.create_task(eval_one(ex_id, current_time))
            pending[task] = {"start": time.time(), "example_id": ex_id}

        # Monitor tasks and detach stragglers based on runtime statistics
        last_threshold: float | None = None
        while pending:
            if (
                len(eval_durations) >= 2
                and not (self.skip_final_straggler_cutoff and is_final_shard)
                and (
                    # Dynamic coverage trigger: start detaching earlier on small batches
                    # min_cov = max(base, min(0.6, 5/total))
                    # base = 0.25 (partial rungs) or 0.30 (final rung)
                    completed >= max(1, _required_detach_samples())
                )
            ):
                import statistics

                mean_duration = statistics.fmean(eval_durations)
                stdev = statistics.pstdev(eval_durations) if len(eval_durations) > 1 else 0.0
                # Robust quantiles (fallback to mean when samples are too few)
                perc70 = (
                    statistics.quantiles(eval_durations, n=100)[69]
                    if len(eval_durations) >= 5
                    else mean_duration
                )
                perc80 = (
                    statistics.quantiles(eval_durations, n=100)[79]
                    if len(eval_durations) >= 7
                    else max(perc70, mean_duration)
                )

                # Smooth absolute cap derived from observed latencies (no hard-coded tiers)
                # - Partial rungs: tolerate ~2.0x p80
                # - Final rung: a bit more headroom (~2.5x p80)
                dyn_cap = (perc80 * (2.5 if is_final_shard else 2.0))
                # Clamp to a reasonable range to avoid runaway caps on noisy small samples
                dyn_cap = float(max(12.0, min(75.0, dyn_cap)))

                # Core dynamic threshold: exceed typical latency by a healthy margin
                # Use the max of multiple estimators to avoid over-detaching on tight means
                core = max(
                    mean_duration + 1.0 * stdev,
                    perc70 * 1.3,
                    perc80 * 1.15,
                )
                threshold_raw = min(dyn_cap, core)
                # Smooth with a small EMA to reduce jitter
                if last_threshold is None:
                    threshold = threshold_raw
                else:
                    threshold = 0.6 * last_threshold + 0.4 * threshold_raw
                last_threshold = threshold
                # Add a small slack so equal-times don’t trigger detaches due to rounding jitter
                detach_margin = max(0.5, 0.10 * threshold)
                now = time.time()

                # ALWAYS log threshold calculation for visibility (not gated by show_progress)
                self.logger.log(
                    f"🔍 Straggler check: {completed}/{total} complete, "
                    f"threshold={threshold:.1f}s (mean={mean_duration:.1f}s, p70={perc70:.1f}s, p80={perc80:.1f}s, cap={dyn_cap:.1f}s), "
                    f"{len(pending)} tasks pending"
                )

                detached = False
                straggler_count = 0
                for task, info in list(pending.items()):
                    start_time = info["start"]
                    example_id = info["example_id"]
                    elapsed_task = now - start_time
                    ex_latency = latency_ema.get(example_id)
                    if ex_latency is not None:
                        ex_threshold = max(ex_latency * 1.5, threshold)
                    else:
                        ex_threshold = threshold
                    is_straggler = elapsed_task > (ex_threshold + detach_margin)
                    status = "✅ within" if not is_straggler else "⚠️ EXCEEDS"
                    self.logger.log(
                        f"   Task {example_id}: elapsed {elapsed_task:.1f}s {status} threshold {ex_threshold:.1f}s (+{detach_margin:.1f}s slack)"
                    )
                    if is_straggler and deliver_flags.get(example_id, True):
                        straggler_count += 1
                        deliver_flags[example_id] = False
                        if self.cancel_stragglers_immediately:
                            task.cancel()
                        else:
                            _ensure_straggler_future(example_id)
                        pending.pop(task, None)
                        await asyncio.sleep(0)
                        detached = True
                        denom = max(total, 1)
                        self.logger.log(
                            f"⚡ Detaching straggler #{straggler_count} (example {example_id}) at {completed}/{total} "
                            f"({completed / denom * 100:.0f}%), elapsed {elapsed_task:.1f}s > threshold {threshold:.1f}s + slack {detach_margin:.1f}s"
                        )
                if detached and straggler_count:
                    active_total = len(pending) + completed + straggler_count
                    self.logger.log(
                        f"📊 Straggler stats: detached {straggler_count}/{active_total} tasks, "
                        f"mean={mean_duration:.1f}s, stdev={stdev:.1f}s, p80={perc80:.1f}s, threshold={threshold:.1f}s"
                    )
                    straggler_detached_total += straggler_count
                    if self.metrics:
                        self.metrics.record_early_stop("stragglers", is_final_rung=is_final_shard)

            if not pending:
                break

            wait_set = list(pending.keys())
            if not wait_set:
                break

            done, _ = await asyncio.wait(
                wait_set,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                info = pending.pop(task, None)
                try:
                    await task  # Collect result (already added to results by eval_one)
                except asyncio.CancelledError:
                    pass  # Expected for cancelled tasks
                except Exception:
                    pass  # Already handled in eval_one

            if early_stop_flag:
                for task in pending:
                    task.cancel()
                break

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # No explicit progress bar cleanup when using logger

        # Log batch completion metrics
        batch_duration = time.time() - batch_start_time
        if eval_durations:
            import statistics
            mean_dur = statistics.fmean(eval_durations)
            stdev_dur = statistics.pstdev(eval_durations) if len(eval_durations) > 1 else 0.0
            try:
                p50 = statistics.quantiles(eval_durations, n=100)[49]
                p95 = statistics.quantiles(eval_durations, n=100)[94]
            except Exception:
                p50 = mean_dur
                p95 = max(mean_dur, max(eval_durations))
            self._tail_ratio = float(p95 / max(p50, 1e-3)) if p50 else float("inf")
            if show_progress:
                self.logger.log(
                    f"⏱️  Batch complete: {len(results)}/{total} examples in {batch_duration:.1f}s "
                    f"(mean={mean_dur:.1f}s, stdev={stdev_dur:.1f}s, throughput={len(results)/batch_duration:.1f} ex/s)"
                )

        totals: dict[str, float] = {}
        traces: list[dict[str, float]] = []
        example_trace_ids: list[str] = []
        n_examples = 0
        for result in results:
            totals = _accumulate(totals, result.objectives, weight=result.n_examples)
            traces.extend(result.traces)
            if result.example_ids:
                example_trace_ids.extend(result.example_ids)
            n_examples += result.n_examples

        averaged = {k: v / max(n_examples, 1) for k, v in totals.items()}
        coverage_ratio = completed / max(total, 1)
        shard_key = shard_fraction if shard_fraction is not None else 0.0

        if self.metrics:
            self.metrics.record_shard_outcome(
                shard_fraction,
                coverage_ratio,
                straggler_detached_total,
                batch_duration,
            )

        if (coverage_ratio < 0.999 or straggler_detached_total > 0) and total > 0:
            self.logger.log(
                f"📏 Shard outcome {shard_key:.0%}: coverage={coverage_ratio:.1%} "
                f"(completed {completed}/{total}), stragglers_detached={straggler_detached_total}, "
                f"duration={batch_duration:.1f}s"
            )
        return EvalResult(
            objectives=averaged,
            traces=traces,
            n_examples=n_examples,
            shard_fraction=shard_fraction,
            example_ids=example_trace_ids,
        )

    @property
    def inflight_examples(self) -> int:
        """Current number of example-level evaluations running."""
        return self._inflight_examples

    @property
    def max_observed_inflight(self) -> int:
        """Highest concurrent example-level evaluations seen since instantiation."""
        return self._max_observed_inflight

    @property
    def tail_latency_ratio(self) -> float:
        return max(1.0, float(self._tail_ratio))

    async def collect_straggler_results(
        self,
        candidate: Candidate,
        example_ids: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> list[EvalResult]:
        """Wait for detached straggler evaluations to finish and return their results.

        Args:
            candidate: Candidate whose evaluations produced the stragglers
            example_ids: Example identifiers we are still missing
            timeout: Optional time budget (seconds). ``None`` waits indefinitely,
                ``0`` performs a non-blocking poll.

        Returns:
            List of EvalResult objects (one per completed straggler example).
        """

        if not example_ids or not self._use_straggler_futures:
            return []

        fingerprint = candidate.fingerprint
        futures: list[asyncio.Future[EvalResult]] = []
        keys: list[tuple[str, str]] = []

        for example_id in example_ids:
            key = (fingerprint, example_id)
            future = self._straggler_results.get(key)
            if future is None:
                continue
            futures.append(future)
            keys.append(key)

        if not futures:
            return []

        wait_timeout: float | None
        if timeout is None:
            wait_timeout = None
        else:
            wait_timeout = max(0.0, timeout)

        if wait_timeout == 0.0:
            done = {future for future in futures if future.done()}
            pending = {future for future in futures if not future.done()}
        else:
            done, pending = await asyncio.wait(futures, timeout=wait_timeout)

        results: list[EvalResult] = []
        for future in done:
            if future.cancelled():
                continue
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging only
                self.logger.log(f"⚠️  Straggler future failed: {exc}")
                continue
            results.append(result)

        # Clean up completed futures so they are not awaited twice
        for key in keys:
            future = self._straggler_results.get(key)
            if future is not None and future.done():
                self._straggler_results.pop(key, None)

        if results:
            self.logger.log(
                f"📦 Collected {len(results)} completed stragglers for candidate {fingerprint[:12]}"
            )

        # Pending futures remain registered for the next grace window
        return results

    def discard_straggler_results(self, candidate: Candidate, example_ids: Sequence[str]) -> None:
        """Forget pending straggler futures for the given candidate/example IDs."""

        if not example_ids or not self._use_straggler_futures:
            return

        fingerprint = candidate.fingerprint
        for example_id in example_ids:
            key = (fingerprint, example_id)
            if key in self._straggler_results:
                self._straggler_results.pop(key, None)


def _accumulate(
    base: dict[str, float],
    update: dict[str, float],
    weight: int = 1,
) -> dict[str, float]:
    merged = dict(base)
    for key, value in update.items():
        # Skip non-numeric values (e.g., example_id, output strings)
        if not isinstance(value, (int, float)):
            continue
        merged[key] = merged.get(key, 0.0) + value * weight
    return merged
