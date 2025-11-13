"""Human-friendly progress reporting for TurboGEPA runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .logger import LogLevel, LoggerProtocol

if TYPE_CHECKING:  # pragma: no cover
    from turbo_gepa.orchestrator import Orchestrator


def _truncate_prompt(text: str, max_chars: int = 180) -> str:
    """Return a single-line snippet for logs."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1]}…"


@dataclass
class ProgressSnapshot:
    """Lightweight snapshot of the orchestrator state for logging."""

    timestamp: float
    elapsed: float
    run_id: str
    round_index: int
    evaluations: int
    pareto_size: int
    best_quality: float
    best_quality_shard: float
    best_prompt_snippet: Optional[str]
    queue_size: int
    inflight_candidates: int
    inflight_examples: int
    target_quality: Optional[float]
    target_shard_fraction: float
    target_reached: bool
    stop_reason: Optional[str]
    promotion_attempts: dict[int, int]
    promotion_promoted: dict[int, int]
    promotion_pruned: dict[int, int]


def build_progress_snapshot(orchestrator: Orchestrator) -> ProgressSnapshot:
    """Collect a ProgressSnapshot from the orchestrator."""

    pareto_entries = orchestrator.archive.pareto_entries()
    best_quality = 0.0
    best_shard = 0.0
    snippet: Optional[str] = None
    if pareto_entries:
        promote_obj = orchestrator.config.promote_objective
        full_shard = orchestrator._runtime_shards[-1] if orchestrator._runtime_shards else 1.0
        tolerance = 1e-6
        full_entries = [
            entry
            for entry in pareto_entries
            if entry.result.shard_fraction is not None
            and abs(entry.result.shard_fraction - full_shard) <= tolerance
        ]
        preferred = full_entries or pareto_entries
        best_entry = max(
            preferred,
            key=lambda entry: (
                entry.result.objectives.get(promote_obj, 0.0),
                entry.result.objectives.get("neg_cost", float("-inf")),
            ),
        )
        best_quality = best_entry.result.objectives.get(promote_obj, 0.0)
        best_shard = best_entry.result.shard_fraction or 0.0
        snippet = _truncate_prompt(best_entry.candidate.text)

    target = orchestrator.config.target_quality
    target_shard = float(getattr(orchestrator.config, "target_shard_fraction", 1.0) or 1.0)
    tol = 1e-6
    # Consider target reached only when both quality and shard criteria are met
    target_reached = (
        target is not None
        and best_quality >= target
        and (best_shard + tol) >= target_shard
    )
    run_started_at = orchestrator.run_started_at or time.time()
    metrics = getattr(orchestrator, "metrics", None)
    attempts = dict(getattr(metrics, "promotion_attempts_by_rung", {})) if metrics else {}
    promoted = dict(getattr(metrics, "promotions_by_rung", {})) if metrics else {}
    pruned = dict(getattr(metrics, "promotion_pruned_by_rung", {})) if metrics else {}

    return ProgressSnapshot(
        timestamp=time.time(),
        elapsed=max(0.0, time.time() - run_started_at),
        round_index=orchestrator.round_index,
        evaluations=orchestrator.evaluations_run,
        run_id=orchestrator.run_id,
        pareto_size=len(pareto_entries),
        best_quality=best_quality,
        best_quality_shard=best_shard,
        best_prompt_snippet=snippet,
        queue_size=len(orchestrator.queue),
        inflight_candidates=orchestrator.total_inflight,
        inflight_examples=orchestrator.examples_inflight,
        target_quality=target,
        target_shard_fraction=target_shard,
        target_reached=target_reached,
        stop_reason=orchestrator.stop_reason,
        promotion_attempts=attempts,
        promotion_promoted=promoted,
        promotion_pruned=pruned,
    )


class ProgressReporter:
    """Emit concise, readable log lines for TurboGEPA progress."""

    def __init__(self, logger: LoggerProtocol, *, log_prompts: bool = False) -> None:
        self.logger = logger
        self.log_prompts = log_prompts
        self._last_round = -1
        self._prev_best = 0.0
        self._last_improvement_ts: Optional[float] = None

    def __call__(self, snapshot: ProgressSnapshot) -> None:
        """Record a snapshot (matches the metrics_callback signature)."""
        if snapshot.round_index == self._last_round:
            # Still in the same round; avoid spamming logs.
            return
        self._last_round = snapshot.round_index

        elapsed = max(snapshot.elapsed, 1e-6)
        eval_rate = snapshot.evaluations / elapsed

        best_delta = snapshot.best_quality - self._prev_best
        if best_delta > 1e-9:
            self._last_improvement_ts = snapshot.timestamp
            self._prev_best = snapshot.best_quality
        time_since_improve = None
        if self._last_improvement_ts is not None:
            time_since_improve = snapshot.timestamp - self._last_improvement_ts

        target_block = ""
        if snapshot.target_quality is not None:
            status = "✓" if snapshot.target_reached else f"{snapshot.target_quality:.2f}"
            if not snapshot.target_reached and snapshot.target_quality > 0:
                progress_pct = snapshot.best_quality / snapshot.target_quality
                target_block = f" target={status} progress={progress_pct:.0%}"
            else:
                target_block = f" target={status}"

        delta_str = f"{best_delta:+.3f}" if abs(best_delta) >= 5e-4 else "0.000"
        since_str = (
            f"{time_since_improve:.0f}s ago" if time_since_improve is not None and time_since_improve >= 1.0 else "just now"
        )

        rung_stats = ""
        if snapshot.promotion_attempts:
            parts = []
            for rung in sorted(snapshot.promotion_attempts):
                attempts = snapshot.promotion_attempts.get(rung, 0)
                promoted = snapshot.promotion_promoted.get(rung, 0)
                pruned = snapshot.promotion_pruned.get(rung, 0)
                rate = promoted / attempts if attempts else 0.0
                parts.append(f"{rung}:{promoted}/{attempts} ({rate:.0%})")
            rung_stats = " promotions=[" + ", ".join(parts[:4]) + "]"

        self.logger.log(
            (
                f"[TurboGEPA] run={snapshot.run_id} round={snapshot.round_index} "
                f"evals={snapshot.evaluations} "
                f"speed={eval_rate:.2f}/s "
                f"best={snapshot.best_quality:.3f}@{snapshot.best_quality_shard:.0%} "
                f"(Δ={delta_str}, last_improve={since_str}) "
                f"pareto={snapshot.pareto_size} "
                f"queue={snapshot.queue_size} inflight={snapshot.inflight_candidates}"
                f"{target_block}{rung_stats}"
            ),
            LogLevel.WARNING,
        )

        if self.log_prompts and snapshot.best_prompt_snippet:
            self.logger.log(f"   ↳ prompt: {snapshot.best_prompt_snippet}", LogLevel.WARNING)

        if snapshot.stop_reason:
            self.logger.log(f"   stop_reason={snapshot.stop_reason}", LogLevel.WARNING)

        structured = {
            "event": "progress",
            "run_id": snapshot.run_id,
            "round": snapshot.round_index,
            "evaluations": snapshot.evaluations,
            "eval_rate_per_sec": eval_rate,
            "best_quality": snapshot.best_quality,
            "best_quality_shard": snapshot.best_quality_shard,
            "best_quality_delta": best_delta,
            "time_since_improvement": time_since_improve,
            "pareto_size": snapshot.pareto_size,
            "queue": snapshot.queue_size,
            "inflight_candidates": snapshot.inflight_candidates,
            "target_quality": snapshot.target_quality,
            "target_shard": snapshot.target_shard_fraction,
            "target_reached": snapshot.target_reached,
            "stop_reason": snapshot.stop_reason,
            "promotion_attempts": snapshot.promotion_attempts,
            "promotion_promoted": snapshot.promotion_promoted,
            "promotion_pruned": snapshot.promotion_pruned,
        }
        self.logger.log(json.dumps(structured), LogLevel.WARNING)
