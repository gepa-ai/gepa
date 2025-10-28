"""
Budgeted scheduler implementing an ASHA-style successive halving policy.

The current implementation provides transparent hooks for plugging in custom
promotion heuristics while keeping default behavior light-weight.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .cache import candidate_key
from .interfaces import Candidate, EvalResult
from .stop_governor import EpochMetrics, StopGovernor, StopGovernorConfig

logger = logging.getLogger(__name__)


@dataclass
class Rung:
    """Tracks candidates evaluated on a specific shard."""

    shard_fraction: float
    results: dict[str, deque[float]] = field(default_factory=dict)
    max_history: int = 64

    def update(self, key: str, score: float) -> None:
        history = self.results.setdefault(key, deque(maxlen=self.max_history))
        history.append(score)

    def summary(self, key: str) -> float:
        values = self.results.get(key, [])
        return statistics.fmean(values) if values else float("-inf")


@dataclass
class SchedulerConfig:
    shards: Sequence[float]
    eps_improve: float
    quantile: float
    enable_convergence: bool = False
    lineage_patience: int = 0
    lineage_min_improve: float = 0.01


@dataclass
class _ConvergenceState:
    governor: StopGovernor
    evals: int = 0
    tokens: float = 0.0
    last_debug: dict[str, Any] | None = None


class BudgetedScheduler:
    """Manage shard promotion and pruning for asynchronous evaluations."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.rungs = [Rung(shard) for shard in config.shards]
        self._candidate_levels: dict[str, int] = {}
        self._pending_promotions: list[Candidate] = []
        self._parent_scores: dict[str, float] = {}
        self._convergence: dict[str, dict[int, _ConvergenceState]] = {}
        self._convergence_config = StopGovernorConfig(
            alpha=0.5,
            hysteresis_window=3,
            stop_threshold=0.2,
            max_no_improvement_epochs=4,
        )
        self._lineage_failures: dict[tuple[str, int], int] = {}
        self._lineage_seen_children: set[tuple[str, int]] = set()

    def _sched_key(self, candidate: Candidate) -> str:
        meta = candidate.meta if isinstance(candidate.meta, dict) else None
        if meta:
            key = meta.get("_sched_key")
            if isinstance(key, str):
                return key
        return candidate_hash(candidate)

    def current_shard_index(self, candidate: Candidate) -> int:
        return self._candidate_levels.get(self._sched_key(candidate), 0)

    def _get_convergence_state(self, key: str, rung_idx: int) -> _ConvergenceState:
        cand_states = self._convergence.setdefault(key, {})
        state = cand_states.get(rung_idx)
        if state is None:
            state = _ConvergenceState(governor=StopGovernor(replace(self._convergence_config)))
            cand_states[rung_idx] = state
        return state

    def _clear_convergence(self, key: str, rung_idx: int | None = None) -> None:
        states = self._convergence.get(key)
        if not states:
            return
        if rung_idx is None:
            self._convergence.pop(key, None)
            return
        states.pop(rung_idx, None)
        if not states:
            self._convergence.pop(key, None)

    def _apply_convergence(
        self,
        candidate: Candidate,
        rung_idx: int,
        final_rung_index: int,
        score: float,
        result: EvalResult,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Update convergence tracker and decide whether to promote."""
        key = self._sched_key(candidate)
        state = self._get_convergence_state(key, rung_idx)
        state.evals += 1
        state.tokens += result.objectives.get("tokens", 0.0)

        metrics = EpochMetrics(
            round_num=state.evals,
            hypervolume=score,
            new_evaluations=state.evals,
            best_quality=score,
            best_cost=result.objectives.get("neg_cost", float("-inf")),
            frontier_ids={f"{key}:{rung_idx}"},
            qd_filled_cells=1,
            qd_total_cells=1,
            qd_novelty_rate=0.0,
            total_tokens_spent=int(state.tokens),
        )
        state.governor.update(metrics)
        should_stop, debug = state.governor.should_stop()
        if should_stop:
            state.last_debug = debug
        if should_stop and rung_idx < final_rung_index:
            return True, debug
        return False, debug

    def current_shard_fraction(self, candidate: Candidate) -> float:
        idx = self.current_shard_index(candidate)
        return self.rungs[idx].shard_fraction

    def record(self, candidate: Candidate, result: EvalResult, objective_key: str) -> str:
        """Record fresh metrics and queue promotions when the candidate excels."""
        decision = "pending"
        score = result.objective(objective_key, default=None)
        if score is None:
            return decision

        idx = self.current_shard_index(candidate)
        final_rung_index = len(self.rungs) - 1
        rung = self.rungs[idx]
        sched_key = self._sched_key(candidate)
        rung.update(sched_key, score)

        force_promote = False
        force_reason: str | None = None
        convergence_debug: dict[str, Any] | None = None
        if self.config.enable_convergence:
            force_promote, convergence_debug = self._apply_convergence(
                candidate,
                idx,
                final_rung_index,
                score,
                result,
            )
            if force_promote and convergence_debug and convergence_debug.get("reason"):
                force_reason = f"convergence:{convergence_debug['reason']}"

        parent_objectives = candidate.meta.get("parent_objectives") if isinstance(candidate.meta, dict) else None
        parent_score = None
        if isinstance(parent_objectives, dict):
            parent_score = parent_objectives.get(objective_key)
        elif "parent_score" in candidate.meta:
            parent_score = candidate.meta.get("parent_score")
        parent_fp = candidate.meta.get("parent") if isinstance(candidate.meta, dict) else None

        if (
            parent_fp
            and self.config.lineage_patience > 0
            and parent_score is not None
            and idx < final_rung_index
        ):
            child_seen_key = (sched_key, idx)
            if child_seen_key not in self._lineage_seen_children:
                self._lineage_seen_children.add(child_seen_key)
                lineage_key = (parent_fp, idx)
                delta = score - parent_score
                if delta >= self.config.lineage_min_improve:
                    self._lineage_failures.pop(lineage_key, None)
                else:
                    failures = self._lineage_failures.get(lineage_key, 0) + 1
                    self._lineage_failures[lineage_key] = failures
                    if failures >= self.config.lineage_patience and not force_promote:
                        force_promote = True
                        force_reason = f"lineage:{failures}"
                        self._lineage_failures.pop(lineage_key, None)

        if force_promote:
            if force_reason:
                logger.debug("   ⚡ ASHA: Promoted via %s", force_reason)
            elif convergence_debug and convergence_debug.get("reason"):
                logger.debug(
                    "   ⚡ ASHA: Promoted via convergence (%s)",
                    convergence_debug["reason"],
                )
            else:
                logger.debug("   ⚡ ASHA: Promoted via convergence")
            self._candidate_levels[sched_key] = min(idx + 1, final_rung_index)
            self._pending_promotions.append(candidate)
            self._parent_scores[sched_key] = score
            if sched_key in rung.results:
                del rung.results[sched_key]
            self._clear_convergence(sched_key, idx)
            if parent_fp and self.config.lineage_patience > 0:
                self._lineage_failures.pop((parent_fp, idx), None)
            self._lineage_seen_children.discard((sched_key, idx))
            return "promoted"

        # Check parent comparison: prune if worse, promote if better
        if parent_score is not None:
            if score < parent_score + self.config.eps_improve:
                # Worse than parent - prune immediately
                self._parent_scores[sched_key] = score
                logger.debug(
                    "   ❌ ASHA: Pruned (worse than parent: %s < %s + %s)",
                    f"{score:.1%}",
                    f"{parent_score:.1%}",
                    f"{self.config.eps_improve:.2%}",
                )
                self._clear_convergence(sched_key)
                self._lineage_seen_children.discard((sched_key, idx))
                return "pruned"
            elif idx < final_rung_index:
                # Better than parent - promote immediately (skip quantile check)
                logger.debug(
                    "   ⬆️  ASHA: PROMOTED! (better than parent: %s >= %s + %s, rung %s -> %s)",
                    f"{score:.1%}",
                    f"{parent_score:.1%}",
                    f"{self.config.eps_improve:.2%}",
                    idx,
                    idx + 1,
                )
                self._candidate_levels[sched_key] = idx + 1
                self._pending_promotions.append(candidate)
                self._parent_scores[sched_key] = score
                self._clear_convergence(sched_key, idx)
                if parent_fp and self.config.lineage_patience > 0:
                    self._lineage_failures.pop((parent_fp, idx), None)
                self._lineage_seen_children.discard((sched_key, idx))
                return "promoted"

        if idx >= final_rung_index:
            logger.debug("   ✅ ASHA: Completed at final rung (score=%s)", f"{score:.1%}")
            self._clear_convergence(sched_key, idx)
            if parent_fp and self.config.lineage_patience > 0:
                self._lineage_failures.pop((parent_fp, idx), None)
            self._lineage_seen_children.discard((sched_key, idx))
            return "completed"  # already at max shard

        # Always promote perfect scores (1.0) to verify on full dataset
        should_promote = score >= 1.0

        if not should_promote:
            threshold = self._promotion_threshold(rung)
            if threshold == float("-inf"):
                logger.debug(
                    "   ⏸️  ASHA: Pending (no threshold yet, rung has %s candidates)",
                    len(rung.results),
                )
                return decision
            should_promote = score >= threshold

            if should_promote:
                logger.debug(
                    "   ⬆️  ASHA: PROMOTED! (score=%s >= threshold=%s, rung %s -> %s)",
                    f"{score:.1%}",
                    f"{threshold:.1%}",
                    idx,
                    idx + 1,
                )
            else:
                logger.debug(
                    "   ❌ ASHA: Pruned (score=%s < threshold=%s, rung=%s, %s candidates)",
                    f"{score:.1%}",
                    f"{threshold:.1%}",
                    idx,
                    len(rung.results),
                )

        if should_promote:
            self._candidate_levels[sched_key] = idx + 1
            self._pending_promotions.append(candidate)
            self._parent_scores[sched_key] = score
            # Keep promoted candidates in rung.results for threshold calculation
            # This allows future candidates to compare against ALL evaluated candidates on this shard
            self._clear_convergence(sched_key, idx)
            if parent_fp and self.config.lineage_patience > 0:
                self._lineage_failures.pop((parent_fp, idx), None)
            self._lineage_seen_children.discard((sched_key, idx))
            decision = "promoted"
        else:
            decision = "pruned"
            # Keep pruned candidates in rung.results too (already there, just don't delete)
            self._clear_convergence(sched_key)
            self._lineage_seen_children.discard((sched_key, idx))
        return decision

    def shard_fraction_for_index(self, index: int) -> float:
        index = max(0, min(index, len(self.rungs) - 1))
        return self.rungs[index].shard_fraction

    def promote_ready(self) -> list[Candidate]:
        """Return candidates ready for the next shard."""
        ready = list(self._pending_promotions)
        self._pending_promotions.clear()
        return ready

    def _promotion_threshold(self, rung: Rung) -> float:
        """Calculate the moving quantile score threshold for promotion.

        Always promotes at least the top candidate(s), even if all have same score.
        This prevents the chicken-and-egg problem where all candidates are equally
        bad (e.g., all 0%) but we still need to keep the best ones for reflection.
        """
        samples = [statistics.fmean(list(values)) for values in rung.results.values() if values]
        if not samples:
            return float("-inf")
        if len(samples) == 1:
            # Single candidate on this rung - allow it to advance based on its own score.
            return samples[0]

        # Calculate rank for quantile-based promotion (top 40% by default)
        quantile_rank = max(int(len(samples) * (1 - self.config.quantile)), 0)

        # Always promote at least top 1-2 candidates, even if quantile would prune all
        # This ensures reflection has something to work with when all candidates are similar
        min_to_promote = min(2, len(samples))  # Keep at least 1-2 best
        rank = min(quantile_rank, len(samples) - min_to_promote)

        samples.sort(reverse=True)
        threshold_score = samples[rank]

        # Only add eps_improve if there's actually variation in scores
        # When all candidates are tied (e.g., all 0%), don't add epsilon or we'll prune everyone
        if len(set(samples)) > 1:
            # Multiple distinct scores - require improvement over threshold
            return threshold_score + self.config.eps_improve
        else:
            # All tied - just use the score (allows ties to promote)
            return threshold_score


def candidate_hash(candidate: Candidate) -> str:
    return candidate_key(candidate)
