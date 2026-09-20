# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Append-only event log and run statistics. Written by the head thread only."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass
class StageStats:
    dispatched_tasks: int = 0
    completed_tasks: int = 0
    completed_items: int = 0
    failed_tasks: int = 0
    busy_seconds: float = 0.0
    wait_seconds: float = 0.0
    max_queue_depth: int = 0
    worker_changes: list[tuple[float, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        items = max(self.completed_items, 1)
        return {
            "dispatched_tasks": self.dispatched_tasks,
            "completed_tasks": self.completed_tasks,
            "completed_items": self.completed_items,
            "failed_tasks": self.failed_tasks,
            "busy_seconds": round(self.busy_seconds, 4),
            "mean_service_seconds": round(self.busy_seconds / max(self.completed_tasks, 1), 4),
            "mean_wait_seconds": round(self.wait_seconds / items, 4),
            "max_queue_depth": self.max_queue_depth,
            "worker_changes": [(round(t, 3), k) for t, k in self.worker_changes],
        }


class EventLog:
    """JSONL log of every sample, dispatch, completion, routing decision and commit.

    The log is the record of what ran when. Per-stage timing, queue depths and
    staleness gaps can all be recomputed from it.
    """

    def __init__(self, run_dir: str | None, enabled: bool = True):
        self._t0 = time.monotonic()
        self._fh: TextIO | None = None
        if enabled and run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            self._fh = open(os.path.join(run_dir, "async_events.jsonl"), "a", encoding="utf-8")
        self.stages: dict[str, StageStats] = {}
        self.outcomes: Counter[str] = Counter()
        self.gaps_at_screen: list[int] = []
        self.gaps_at_validate: list[int] = []
        self.gaps_at_commit: list[int] = []
        self.patched = 0
        self.patch_discards = 0
        # Time-weighted concurrency accounting.
        self._last_tick = self._t0
        self._inflight_seconds = 0.0
        self._overlap_seconds = 0.0
        self.max_busy_stages = 0

    def now(self) -> float:
        return time.monotonic() - self._t0

    def stage(self, name: str) -> StageStats:
        return self.stages.setdefault(name, StageStats())

    def tick(self, inflight_by_stage: dict[str, int]) -> None:
        """Accumulate time-weighted in-flight counts since the previous tick."""
        t = time.monotonic()
        dt = t - self._last_tick
        self._last_tick = t
        busy = [s for s, n in inflight_by_stage.items() if n > 0]
        self._inflight_seconds += dt * sum(inflight_by_stage.values())
        if len(busy) >= 2:
            self._overlap_seconds += dt
        self.max_busy_stages = max(self.max_busy_stages, len(busy))

    def emit(self, kind: str, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {"t": round(self.now(), 4), "kind": kind, **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def summary(self, *, total_metric_calls: int, commits: int, orders: int) -> dict[str, Any]:
        wall = max(self.now(), 1e-9)

        def hist(values: list[int]) -> dict[str, int]:
            return {str(k): v for k, v in sorted(Counter(values).items())}

        return {
            "wall_seconds": round(wall, 4),
            "orders": orders,
            "commits": commits,
            "total_metric_calls": total_metric_calls,
            "orders_per_minute": round(60.0 * orders / wall, 3),
            "commits_per_minute": round(60.0 * commits / wall, 3),
            "mean_tasks_in_flight": round(self._inflight_seconds / wall, 3),
            "stage_overlap_fraction": round(self._overlap_seconds / wall, 3),
            "max_busy_stages": self.max_busy_stages,
            "outcomes": dict(self.outcomes),
            "staleness_at_screen": hist(self.gaps_at_screen),
            "staleness_at_validate": hist(self.gaps_at_validate),
            "staleness_at_commit": hist(self.gaps_at_commit),
            "patched": self.patched,
            "patch_discards": self.patch_discards,
            "stages": {name: s.as_dict() for name, s in self.stages.items()},
        }
