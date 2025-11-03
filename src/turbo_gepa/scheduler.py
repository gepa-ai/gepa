"""
SIMPLIFIED scheduler: Promotion based ONLY on parent-child comparison.

No cohort quantiles, no convergence detection, no lineage tracking.
Just: is child better than parent? → promote : prune
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Sequence

from .cache import candidate_key
from .interfaces import Candidate, EvalResult

logger = logging.getLogger(__name__)




@dataclass
class SchedulerConfig:
    """SIMPLIFIED scheduler config with evolution-based convergence."""
    shards: Sequence[float]
    eps_improve: float
    patience_generations: int = 3  # Generations without improvement before convergence


class BudgetedScheduler:
    """
    SIMPLIFIED scheduler: Parent-child comparison + generation-based convergence.

    Rules:
    - Seeds (no parent): Always promote
    - Mutations: Promote if score >= parent_score + eps_improve, else prune
    - Convergence: Track generations without improvement per rung
      - Generation = one mutation round (orchestrator decides when round starts)
      - Mid-rung: After N generations without improvement → force promote best
      - Final rung: After N generations without improvement → mark converged
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.shards = list(config.shards)  # Just store the fractions directly
        self._candidate_levels: dict[str, int] = {}
        self._pending_promotions: list[Candidate] = []
        self._parent_scores: dict[str, float] = {}

        # Generation-based convergence tracking
        # rung_idx -> (generations_without_improvement, improvement_this_generation)
        self._rung_generations: dict[int, tuple[int, bool]] = {
            i: (0, False) for i in range(len(self.shards))
        }
        self._best_on_rung: dict[int, tuple[Candidate, float]] = {}  # rung_idx -> (candidate, score)
        self.converged = False  # Flag for final rung convergence

    def _sched_key(self, candidate: Candidate) -> str:
        meta = candidate.meta if isinstance(candidate.meta, dict) else None
        if meta:
            key = meta.get("_sched_key")
            if isinstance(key, str):
                return key
        return candidate_hash(candidate)

    def current_shard_index(self, candidate: Candidate) -> int:
        return self._candidate_levels.get(self._sched_key(candidate), 0)

    def mark_generation_start(self, rung_idx: int) -> None:
        """
        Call this when orchestrator starts a new generation of mutations for a rung.
        This completes the previous generation and checks for convergence.
        """
        if rung_idx not in self._rung_generations:
            return

        stagnant_gens, improved_this_gen = self._rung_generations[rung_idx]

        if improved_this_gen:
            # Had improvement → reset counter
            self._rung_generations[rung_idx] = (0, False)
        else:
            # No improvement → increment counter
            new_count = stagnant_gens + 1
            self._rung_generations[rung_idx] = (new_count, False)

            # Check for convergence
            if new_count >= self.config.patience_generations:
                final_rung_index = len(self.shards) - 1

                if rung_idx >= final_rung_index:
                    # FINAL RUNG: Mark converged
                    self.converged = True
                    logger.debug(
                        "   🛑 CONVERGED on final rung after %d generations without improvement",
                        new_count,
                    )
                else:
                    # MID-RUNG: Force promote best
                    if rung_idx in self._best_on_rung:
                        best_cand, best_score = self._best_on_rung[rung_idx]
                        best_key = self._sched_key(best_cand)
                        self._candidate_levels[best_key] = rung_idx + 1
                        self._pending_promotions.append(best_cand)
                        self._rung_generations[rung_idx] = (0, False)  # Reset
                        logger.debug(
                            "   🚀 FORCE PROMOTED best on rung %d after %d stagnant generations (score=%s)",
                            rung_idx,
                            new_count,
                            f"{best_score:.1%}",
                        )

    def update_shards(self, shards: Sequence[float]) -> None:
        """Update rung configuration while preserving candidate levels."""
        self.config = replace(self.config, shards=tuple(shards))
        self.shards = list(self.config.shards)
        max_idx = max(len(self.shards) - 1, 0)
        for key, level in list(self._candidate_levels.items()):
            if level > max_idx:
                self._candidate_levels[key] = max_idx
        self._pending_promotions.clear()
        # Reset convergence tracking
        self._rung_generations = {i: (0, False) for i in range(len(self.shards))}
        self._best_on_rung.clear()
        self.converged = False

    def current_shard_fraction(self, candidate: Candidate) -> float:
        idx = self.current_shard_index(candidate)
        return self.shards[idx]

    def record(self, candidate: Candidate, result: EvalResult, objective_key: str) -> str:
        """
        SIMPLIFIED: Parent-child comparison + track improvements per generation.

        Rules:
        1. Seeds (no parent): Always promote
        2. Mutations: Compare child vs parent
           - Better → promote, mark improvement_this_generation = True
           - Worse → prune
        3. Orchestrator calls mark_generation_start() when starting new round
        """
        score = result.objective(objective_key, default=None)
        if score is None:
            return "pending"

        idx = self.current_shard_index(candidate)
        final_rung_index = len(self.shards) - 1
        sched_key = self._sched_key(candidate)

        # Track score and best candidate on this rung
        self._parent_scores[sched_key] = score
        if idx not in self._best_on_rung or score > self._best_on_rung[idx][1]:
            self._best_on_rung[idx] = (candidate, score)

        # Check if at final rung
        if idx >= final_rung_index:
            logger.debug("   ✅ ASHA: Completed at final rung (score=%s)", f"{score:.1%}")
            return "completed"

        # Extract parent score
        parent_objectives = candidate.meta.get("parent_objectives") if isinstance(candidate.meta, dict) else None
        parent_score = None
        if isinstance(parent_objectives, dict):
            parent_score = parent_objectives.get(objective_key)
        elif isinstance(candidate.meta, dict) and "parent_score" in candidate.meta:
            parent_score = candidate.meta.get("parent_score")

        # SEED: No parent → always promote
        if parent_score is None:
            logger.debug(
                "   🌱 ASHA: PROMOTED! (seed, rung %s -> %s, score=%s)",
                idx,
                idx + 1,
                f"{score:.1%}",
            )
            self._candidate_levels[sched_key] = idx + 1
            self._pending_promotions.append(candidate)
            # Mark improvement on this rung
            if idx in self._rung_generations:
                gens, _ = self._rung_generations[idx]
                self._rung_generations[idx] = (gens, True)
            return "promoted"

        # MUTATION: Compare child vs parent
        improved = score >= parent_score + self.config.eps_improve

        # Special case: if parent hit ceiling (100%), promote equal scores since no room for improvement
        at_ceiling = parent_score >= 0.999  # Allow for floating point imprecision
        if at_ceiling and score >= parent_score - 0.001:  # Child matches or nearly matches parent at ceiling
            improved = True

        if improved:
            # Better than parent → promote
            logger.debug(
                "   ⬆️  ASHA: PROMOTED! (improved: %s >= %s + %s, rung %s -> %s)",
                f"{score:.1%}",
                f"{parent_score:.1%}",
                f"{self.config.eps_improve:.2%}",
                idx,
                idx + 1,
            )
            self._candidate_levels[sched_key] = idx + 1
            self._pending_promotions.append(candidate)
            # Mark improvement on this rung
            if idx in self._rung_generations:
                gens, _ = self._rung_generations[idx]
                self._rung_generations[idx] = (gens, True)
            return "promoted"
        else:
            # Worse than parent → prune
            logger.debug(
                "   ❌ ASHA: Pruned (no improvement: %s < %s + %s)",
                f"{score:.1%}",
                f"{parent_score:.1%}",
                f"{self.config.eps_improve:.2%}",
            )
            return "pruned"

    def shard_fraction_for_index(self, index: int) -> float:
        index = max(0, min(index, len(self.shards) - 1))
        return self.shards[index]

    def promote_ready(self) -> list[Candidate]:
        """Return candidates ready for the next shard."""
        ready = list(self._pending_promotions)
        self._pending_promotions.clear()
        return ready


def candidate_hash(candidate: Candidate) -> str:
    return candidate_key(candidate)
