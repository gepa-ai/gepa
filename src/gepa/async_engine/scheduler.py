# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Per-stage buffers and worker pools, plus the adaptive worker controller.

The head thread owns every buffer. A stage's executor only runs stage bodies;
completions come back through one thread-safe queue that the head drains.
"""

from __future__ import annotations

import queue
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from statistics import median
from typing import Any

from gepa.async_engine.config import PIPELINE_STAGES, StageConfig
from gepa.async_engine.items import WorkItem


@dataclass
class Completion:
    stage: str
    items: list[WorkItem]
    result: Any
    error: BaseException | None
    service_seconds: float


class StagePool:
    """One stage: a bounded input buffer and a pool of workers with a movable concurrency cap."""

    def __init__(
        self,
        name: str,
        resource: str,
        config: StageConfig,
        completions: queue.Queue[Completion],
        priority: Callable[[WorkItem], float] | None = None,
    ):
        """``priority`` ranks buffered items when a batch is taken: higher first, arrival order on ties."""
        self.name = name
        self.priority = priority
        self.resource = resource
        self.config = config
        self.workers = config.workers
        self.buffer: deque[WorkItem] = deque()
        self.inflight_tasks = 0
        self.inflight_items = 0
        self._completions = completions
        assert config.max_workers is not None
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix=f"gepa-{name}")
        # Rolling window for the adaptive controller.
        self.window_completed_items = 0

    # -- buffer ---------------------------------------------------------

    def has_room(self) -> bool:
        assert self.config.max_queue is not None
        return len(self.buffer) < self.config.max_queue

    def put(self, item: WorkItem) -> None:
        item.enqueued_at = time.monotonic()
        self.buffer.append(item)

    def take_batch(self) -> list[WorkItem]:
        n = min(self.config.batch_size, len(self.buffer))
        priority = self.priority
        if priority is None or n == len(self.buffer) == 1:
            return [self.buffer.popleft() for _ in range(n)]
        ranked = sorted(self.buffer, key=lambda it: (-priority(it), it.enqueued_at))
        chosen = ranked[:n]
        self.buffer = deque(it for it in self.buffer if not any(it is c for c in chosen))
        return chosen

    # -- workers --------------------------------------------------------

    def can_dispatch(self) -> bool:
        return bool(self.buffer) and self.inflight_tasks < self.workers

    def submit(self, items: list[WorkItem], fn: Callable[[], Any]) -> None:
        self.inflight_tasks += 1
        self.inflight_items += len(items)
        started = time.monotonic()
        for it in items:
            it.wait_seconds[self.name] = it.wait_seconds.get(self.name, 0.0) + (started - it.enqueued_at)

        def _done(fut: Future[Any]) -> None:
            elapsed = time.monotonic() - started
            err = fut.exception()
            self._completions.put(Completion(self.name, items, None if err else fut.result(), err, elapsed))

        self._executor.submit(fn).add_done_callback(_done)

    def on_complete(self, completion: Completion) -> None:
        self.inflight_tasks -= 1
        self.inflight_items -= len(completion.items)
        self.window_completed_items += len(completion.items)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class WorkerController:
    """Moves stage worker caps toward balanced production rates.

    Every ``interval`` seconds the controller compares each pipeline stage's
    production rate (items completed per second in the window) with the median
    across stages. A stage that is slow *and has work waiting* is a bottleneck
    and gains a worker. A stage that is fast *and is filling a full downstream
    buffer* is overproducing and loses one. Each stage changes by at most one
    worker per adjustment and stays within its configured bounds.
    """

    def __init__(self, pools: dict[str, StagePool], interval: float):
        self.pools = pools
        self.interval = interval
        self._last = time.monotonic()

    def maybe_adjust(self) -> list[tuple[str, int, int]]:
        now = time.monotonic()
        window = now - self._last
        if window < self.interval:
            return []
        self._last = now
        stages = [s for s in PIPELINE_STAGES if s in self.pools]
        rates = {s: self.pools[s].window_completed_items / window for s in stages}
        for s in stages:
            self.pools[s].window_completed_items = 0
        active = [r for r in rates.values() if r > 0]
        if not active:
            return []
        mid = median(active)
        changes: list[tuple[str, int, int]] = []
        for i, s in enumerate(stages):
            pool = self.pools[s]
            cfg = pool.config
            assert cfg.max_workers is not None
            downstream = self.pools[stages[i + 1]] if i + 1 < len(stages) else None
            starved_for_workers = bool(pool.buffer) and pool.inflight_tasks >= pool.workers
            if rates[s] < 0.5 * mid and starved_for_workers and pool.workers < cfg.max_workers:
                changes.append((s, pool.workers, pool.workers + 1))
                pool.workers += 1
            elif (
                rates[s] > 2.0 * mid
                and downstream is not None
                and not downstream.has_room()
                and pool.workers > cfg.min_workers
            ):
                changes.append((s, pool.workers, pool.workers - 1))
                pool.workers -= 1
        return changes
