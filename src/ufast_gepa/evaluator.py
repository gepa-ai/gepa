"""
Async evaluator responsible for running candidate evaluations on shards.

The evaluator coordinates cache lookups, validator checks, and concurrent
execution of user-provided LLM calls. It retains only stdlib dependencies so
we can run quickly in constrained environments.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, Iterable, List, Sequence

from .cache import DiskCache
from .interfaces import Candidate, EvalResult

Validator = Callable[[Candidate], None]
MetricsMapper = Callable[[Dict[str, float]], Dict[str, float]]
TaskRunner = Callable[[Candidate, str], Awaitable[Dict[str, float]]]


class AsyncEvaluator:
    """Concurrent evaluator with disk-backed caching."""

    def __init__(
        self,
        cache: DiskCache,
        task_runner: TaskRunner,
        validators: Iterable[Validator] | None = None,
        metrics_mapper: MetricsMapper | None = None,
    ) -> None:
        self.cache = cache
        self.task_runner = task_runner
        self.validators = list(validators or [])
        self.metrics_mapper = metrics_mapper or (lambda metrics: metrics)

    async def eval_on_shard(
        self,
        candidate: Candidate,
        example_ids: Sequence[str],
        concurrency: int,
        shard_fraction: float | None = None,
    ) -> EvalResult:
        """
        Evaluate ``candidate`` on ``example_ids`` with a concurrency cap.

        Cached traces are reused automatically, and only cache misses trigger
        fresh model calls.
        """
        for validator in self.validators:
            validator(candidate)

        semaphore = asyncio.Semaphore(max(concurrency, 1))
        results: List[EvalResult] = []

        async def eval_one(example_id: str) -> None:
            cached = await self.cache.get(candidate, example_id)
            if cached:
                results.append(cached)
                return

            async with semaphore:
                metrics = await self.task_runner(candidate, example_id)
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

        await asyncio.gather(*(eval_one(example) for example in example_ids))

        totals: Dict[str, float] = {}
        traces: List[Dict[str, float]] = []
        example_trace_ids: List[str] = []
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


def _accumulate(
    base: Dict[str, float],
    update: Dict[str, float],
    weight: int = 1,
) -> Dict[str, float]:
    merged = dict(base)
    for key, value in update.items():
        # Skip non-numeric values (e.g., example_id, output strings)
        if not isinstance(value, (int, float)):
            continue
        merged[key] = merged.get(key, 0.0) + value * weight
    return merged
