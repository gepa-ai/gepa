"""
Structured logging helpers for TurboGEPA.

Logging relies on JSONL lines so downstream tooling can consume events
without bespoke parsers. All write operations are serialized via asyncio
locks to remain safe under high concurrency.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple


class EventLogger:
    """Append-only JSONL logger with async-safe writes."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        atexit.register(self.close)

    def close(self) -> None:
        if self._handle and not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    async def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Write a structured event."""
        record = {
            "ts": time.time(),
            "event": event_type,
            **payload,
        }
        async with self._lock:
            await asyncio.to_thread(self._append_record, record)

    def _append_record(self, record: Dict[str, Any]) -> None:
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()


class SummaryLogger:
    """Tracks metrics and periodically emits summary statistics."""

    def __init__(self, logger: EventLogger, interval: int = 10) -> None:
        self.logger = logger
        self.interval = interval
        self.cache_hits = 0
        self.cache_misses = 0
        self.eval_latencies: Deque[float] = deque(maxlen=1000)
        self.prune_counts: Dict[int, int] = defaultdict(int)
        self.promote_counts: Dict[int, int] = defaultdict(int)
        self.objectives_history: Deque[Dict[str, float]] = deque(maxlen=1000)
        self.queue_depths: Deque[int] = deque(maxlen=100)
        self.shard_utilizations: Dict[float, int] = defaultdict(int)
        self.last_summary_round = 0

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def record_eval_latency(self, latency_ms: float) -> None:
        """Record evaluation latency in milliseconds."""
        self.eval_latencies.append(latency_ms)

    def record_prune(self, shard_index: int) -> None:
        """Record a candidate being pruned at a specific shard."""
        self.prune_counts[shard_index] += 1

    def record_promote(self, shard_index: int) -> None:
        """Record a candidate being promoted from a specific shard."""
        self.promote_counts[shard_index] += 1

    def record_objectives(self, objectives: Dict[str, float]) -> None:
        """Record objective values from an evaluation."""
        self.objectives_history.append(dict(objectives))

    def record_queue_depth(self, depth: int) -> None:
        """Record current queue depth."""
        self.queue_depths.append(depth)

    def record_shard_usage(self, shard_fraction: float) -> None:
        """Record usage of a specific shard size."""
        self.shard_utilizations[shard_fraction] += 1

    async def maybe_emit_summary(self, round_index: int) -> None:
        """Emit a summary if the interval has elapsed."""
        if round_index - self.last_summary_round >= self.interval:
            await self.emit_summary(round_index)
            self.last_summary_round = round_index

    async def emit_summary(self, round_index: int) -> None:
        """Emit a summary of collected metrics."""
        total_cache_accesses = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0.0
        )

        avg_latency = (
            statistics.mean(self.eval_latencies) if self.eval_latencies else 0.0
        )
        p50_latency = (
            statistics.median(self.eval_latencies) if self.eval_latencies else 0.0
        )
        p95_latency = (
            statistics.quantiles(self.eval_latencies, n=20)[18]
            if len(self.eval_latencies) > 20
            else 0.0
        )

        avg_queue_depth = (
            statistics.mean(self.queue_depths) if self.queue_depths else 0.0
        )

        # Calculate prune rate at shard 0 (first shard)
        shard_0_prunes = self.prune_counts.get(0, 0)
        shard_0_promotes = self.promote_counts.get(0, 0)
        shard_0_total = shard_0_prunes + shard_0_promotes
        prune_rate_shard_0 = (
            shard_0_prunes / shard_0_total if shard_0_total > 0 else 0.0
        )

        # Compute objectives histogram
        objectives_summary = {}
        if self.objectives_history:
            keys = set()
            for obj in self.objectives_history:
                keys.update(obj.keys())
            for key in keys:
                values = [obj.get(key, 0.0) for obj in self.objectives_history if key in obj]
                if values:
                    objectives_summary[key] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                    }

        summary = {
            "round": round_index,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "latency_ms": {
                "mean": avg_latency,
                "p50": p50_latency,
                "p95": p95_latency,
            },
            "queue_depth_avg": avg_queue_depth,
            "prune_rate_shard_0": prune_rate_shard_0,
            "prune_counts": dict(self.prune_counts),
            "promote_counts": dict(self.promote_counts),
            "shard_utilization": dict(self.shard_utilizations),
            "objectives": objectives_summary,
        }

        await self.logger.log("summary", summary)


class ProgressLogger(EventLogger):
    """EventLogger that also prints progress to stdout with performance tracking."""

    def __init__(self, path: str, display_progress: bool = True) -> None:
        super().__init__(path)
        self.display_progress = display_progress
        self.eval_count = 0
        self.last_best_score = 0.0
        self.score_history: Deque[float] = deque(maxlen=20)  # Last 20 scores
        self.best_history: List[Tuple[int, float]] = []  # (eval_num, score) when improved
        self.start_time = time.time()

    async def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Write event and optionally display progress."""
        await super().log(event_type, payload)

        if not self.display_progress:
            return

        # Print key events to stdout
        if event_type == "eval_done":
            self.eval_count += 1
            score = payload.get("score", 0.0)
            decision = payload.get("decision", "")

            # Track score history
            self.score_history.append(score)

            # Update best score if improved
            if score > self.last_best_score:
                improvement = score - self.last_best_score
                self.last_best_score = score
                self.best_history.append((self.eval_count, score))

                # Calculate trend if we have history
                trend = ""
                if len(self.best_history) >= 2:
                    recent_improvements = len([s for _, s in self.best_history[-5:]])
                    if recent_improvements >= 3:
                        trend = " 📈 Trending up!"

                print(f"✨ Eval {self.eval_count}: New best! {score:.2%} (+{improvement:.2%}){trend}")

            elif self.eval_count % 10 == 0:
                # Show progress every 10 evals with moving average
                recent_avg = statistics.mean(self.score_history) if self.score_history else 0.0
                elapsed = time.time() - self.start_time
                evals_per_sec = self.eval_count / elapsed if elapsed > 0 else 0

                print(f"⏳ Eval {self.eval_count}: Current {score:.2%}, Best {self.last_best_score:.2%}, "
                      f"Avg(last 20) {recent_avg:.2%}, Rate {evals_per_sec:.1f}/s")

        elif event_type == "summary":
            round_num = payload.get("round", 0)
            objectives = payload.get("objectives", {})
            quality = objectives.get("quality", {})
            if quality:
                best = quality.get('max', 0)
                avg = quality.get('mean', 0)
                elapsed = time.time() - self.start_time

                # Show improvement trajectory
                trajectory = ""
                if len(self.best_history) >= 2:
                    first_best = self.best_history[0][1]
                    total_improvement = best - first_best
                    trajectory = f", Improvement {total_improvement:+.2%}"

                print(f"\n📊 Round {round_num + 1} Summary - Best: {best:.2%}, Avg: {avg:.2%}, "
                      f"Evals: {self.eval_count}, Time: {elapsed:.0f}s{trajectory}\n")


def build_logger(base_dir: str, name: str, display_progress: bool = False) -> EventLogger:
    """Create a logger under ``base_dir`` with a filename ``{name}.jsonl``."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    path = Path(base_dir) / f"{name}.jsonl"
    if display_progress:
        return ProgressLogger(str(path), display_progress=True)
    return EventLogger(str(path))
