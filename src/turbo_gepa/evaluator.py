"""
Async evaluator responsible for running candidate evaluations on shards.

The evaluator coordinates cache lookups, validator checks, and concurrent
execution of user-provided LLM calls. It retains only stdlib dependencies so
we can run quickly in constrained environments.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, Sequence

from turbo_gepa.logging.logger import LoggerProtocol, StdOutLogger

from .cache import DiskCache
from .interfaces import Candidate, EvalResult

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

    async def eval_on_shard(
        self,
        candidate: Candidate,
        example_ids: Sequence[str],
        concurrency: int,
        shard_fraction: float | None = None,
        show_progress: bool = False,
        early_stop_fraction: float = 0.9,  # Return after 90% complete
    ) -> EvalResult:
        """
        Evaluate ``candidate`` on ``example_ids`` with a concurrency cap.

        Cached traces are reused automatically, and only cache misses trigger
        fresh model calls.
        """
        for validator in self.validators:
            validator(candidate)

        import time

        semaphore = asyncio.Semaphore(max(concurrency, 1))
        results: list[EvalResult] = []
        completed = 0
        total = len(example_ids)
        early_stop_target = int(total * early_stop_fraction)
        batch_start_time = time.time()
        eval_durations: list[float] = []  # Track how long each eval took (excluding cached)

        async def eval_one(example_id: str, task_start_time: float) -> None:
            nonlocal completed

            cached = await self.cache.get(candidate, example_id)
            if cached:
                results.append(cached)
                completed += 1
                if show_progress:
                    self.logger.log(f"Progress: {completed}/{total} examples ({completed / total * 100:.0f}%)")
                return

            try:
                # Log when we're about to start an API call
                if show_progress:
                    self.logger.log(f"🔄 Starting eval for example {example_id} (inflight: {self._inflight_examples})")

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

                    if show_progress:
                        self.logger.log(f"✅ Completed eval for example {example_id} in {_elapsed_api:.1f}s")

                # Ensure inflight counter is decremented even if mapper raises
                self._inflight_examples = max(0, self._inflight_examples - 1)
                mapped = self.metrics_mapper(metrics)
                trace = dict(metrics)
                trace["example_id"] = example_id
                result = EvalResult(
                    objectives=mapped,
                    traces=[trace],
                    n_examples=1,
                    shard_fraction=shard_fraction,
                    example_ids=[example_id],
                )
                await self.cache.set(candidate, example_id, result)
                results.append(result)
                completed += 1

                # Track duration for non-cached evals
                eval_duration = time.time() - task_start_time
                eval_durations.append(eval_duration)

                if show_progress:
                    self.logger.log(f"Progress: {completed}/{total} examples ({completed / total * 100:.0f}%)")
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
                    "quality": 0.0,
                    "neg_cost": 0.0,
                    "tokens": 0.0,
                }
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
                results.append(result)
                completed += 1
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
                results.append(result)

        # Launch all tasks
        current_time = time.time()
        tasks = {asyncio.create_task(eval_one(ex_id, current_time)): ex_id for ex_id in example_ids}
        pending = set(tasks.keys())

        # Wait for early_stop_fraction to complete, then move on
        while pending:
            # Check if we've hit early stop threshold (dynamic version)
            if completed >= early_stop_target and early_stop_fraction < 1.0 and len(eval_durations) >= 5:
                elapsed = time.time() - batch_start_time
                remaining = len(pending)

                # Compute average eval duration (only non-cached evals)
                avg_duration = sum(eval_durations) / len(eval_durations)

                # Expected time for remaining evals (with 2x buffer for variance)
                # Since we run with concurrency limit, estimate based on parallel execution
                expected_time_for_remaining = avg_duration * 2.0

                # How long should we have taken to reach this point?
                # Estimate: (completed / concurrency) * avg_duration
                expected_time_to_target = (early_stop_target / concurrency) * avg_duration
                time_since_should_have_hit_target = elapsed - expected_time_to_target

                # Early stop if we've been waiting too long for stragglers
                if time_since_should_have_hit_target > expected_time_for_remaining and remaining >= 2:
                    if show_progress:
                        self.logger.log(
                            f"⚡ Early stop: {completed}/{total} complete ({completed / total * 100:.0f}%), cancelling {remaining} stragglers"
                        )
                        self.logger.log(
                            f"      Avg eval duration: {avg_duration:.1f}s, waited {time_since_should_have_hit_target:.1f}s past target..."
                        )
                    # Cancel remaining tasks - they're stragglers
                    for task in pending:
                        task.cancel()
                    break

            # Wait for next batch to complete (with timeout to prevent indefinite blocking)
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED, timeout=5.0)

            # If nothing completed in 5 seconds, log status
            if not done and show_progress:
                self.logger.log(
                    f"⏳ Waiting for tasks: {len(pending)} pending, {completed}/{total} completed"
                )
                continue

            for task in done:
                try:
                    await task  # Collect result (already added to results by eval_one)
                except asyncio.CancelledError:
                    pass  # Expected for cancelled tasks
                except Exception:
                    pass  # Already handled in eval_one

        # No explicit progress bar cleanup when using logger

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
