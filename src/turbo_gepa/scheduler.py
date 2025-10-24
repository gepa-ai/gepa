"""
Budgeted scheduler implementing an ASHA-style successive halving policy.

The current implementation provides transparent hooks for plugging in custom
promotion heuristics while keeping default behavior light-weight.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Sequence

from .cache import candidate_key
from .interfaces import Candidate, EvalResult


@dataclass
class Rung:
    """Tracks candidates evaluated on a specific shard."""

    shard_fraction: float
    results: Dict[str, Deque[float]] = field(default_factory=dict)
    max_history: int = 64

    def update(self, candidate: Candidate, score: float) -> None:
        key = candidate_hash(candidate)
        history = self.results.setdefault(key, deque(maxlen=self.max_history))
        history.append(score)

    def summary(self, candidate: Candidate) -> float:
        values = self.results.get(candidate_hash(candidate), [])
        return statistics.fmean(values) if values else float("-inf")


@dataclass
class SchedulerConfig:
    shards: Sequence[float]
    eps_improve: float
    quantile: float


class BudgetedScheduler:
    """Manage shard promotion and pruning for asynchronous evaluations."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.rungs = [Rung(shard) for shard in config.shards]
        self._candidate_levels: Dict[str, int] = {}
        self._pending_promotions: List[Candidate] = []
        self._parent_scores: Dict[str, float] = {}

    def current_shard_index(self, candidate: Candidate) -> int:
        return self._candidate_levels.get(candidate_hash(candidate), 0)

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
        rung = self.rungs[idx]
        rung.update(candidate, score)
        cand_hash = candidate_hash(candidate)
        parent_objectives = candidate.meta.get("parent_objectives") if isinstance(candidate.meta, dict) else None
        parent_score = None
        if isinstance(parent_objectives, dict):
            parent_score = parent_objectives.get(objective_key)
        elif "parent_score" in candidate.meta:
            parent_score = candidate.meta.get("parent_score")
        if parent_score is not None and score < parent_score + self.config.eps_improve:
            self._parent_scores[cand_hash] = score
            return "pruned"
        if idx >= len(self.rungs) - 1:
            return "completed"  # already at max shard
        threshold = self._promotion_threshold(rung)
        if threshold == float("-inf"):
            return decision
        if score >= threshold:
            self._candidate_levels[cand_hash] = idx + 1
            self._pending_promotions.append(candidate)
            self._parent_scores[cand_hash] = score

            # Rung cleanup: Remove this candidate's results from the previous rung
            # to keep ASHA signal clean and avoid confusion about which shard's
            # results are current. The candidate will be re-evaluated on the next shard.
            if cand_hash in rung.results:
                del rung.results[cand_hash]

            decision = "promoted"
        else:
            decision = "pruned"
        return decision

    def shard_fraction_for_index(self, index: int) -> float:
        index = max(0, min(index, len(self.rungs) - 1))
        return self.rungs[index].shard_fraction

    def promote_ready(self) -> List[Candidate]:
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
            # Single candidate on this rung – allow it to advance based on its own score.
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
