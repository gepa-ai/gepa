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
    # Rung-specific epsilon margins for fair comparison (higher margin at smaller rungs due to variance)
    eps_per_rung: dict[float, float] | None = None  # If None, uses eps_improve for all rungs
    # Shrinkage coefficients for estimating parent@rung_i from parent@final when unavailable
    shrinkage_alpha: dict[float, float] | None = None  # If None, uses {0.2: 0.7, 0.5: 0.85, 1.0: 1.0}


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

        # Per-rung score tracking: (candidate_key, rung_idx) -> score
        self._rung_scores: dict[tuple[str, int], float] = {}

        # Rung-specific epsilon (defaults to uniform eps_improve if not provided)
        if config.eps_per_rung is None:
            # Use uniform eps_improve for all rungs (empty dict signals uniform behavior)
            self.eps_per_rung = {}
        else:
            self.eps_per_rung = dict(config.eps_per_rung)

        # Shrinkage alpha for fallback (defaults if not provided)
        if config.shrinkage_alpha is None:
            self.shrinkage_alpha = {0.2: 0.7, 0.5: 0.85, 1.0: 1.0}
        else:
            self.shrinkage_alpha = dict(config.shrinkage_alpha)

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

    def _get_epsilon_for_rung(self, rung_fraction: float) -> float:
        """Get epsilon for a rung, interpolating if exact match not found."""
        # Try exact match first
        if rung_fraction in self.eps_per_rung:
            return self.eps_per_rung[rung_fraction]

        # Find closest rung fraction with interpolation
        sorted_rungs = sorted(self.eps_per_rung.keys())
        if not sorted_rungs:
            return self.config.eps_improve

        # If smaller than smallest rung, use smallest rung's epsilon
        if rung_fraction <= sorted_rungs[0]:
            return self.eps_per_rung[sorted_rungs[0]]

        # If larger than largest rung, use largest rung's epsilon
        if rung_fraction >= sorted_rungs[-1]:
            return self.eps_per_rung[sorted_rungs[-1]]

        # Interpolate between two nearest rungs
        for i in range(len(sorted_rungs) - 1):
            lower_rung = sorted_rungs[i]
            upper_rung = sorted_rungs[i + 1]
            if lower_rung <= rung_fraction <= upper_rung:
                # Linear interpolation
                lower_eps = self.eps_per_rung[lower_rung]
                upper_eps = self.eps_per_rung[upper_rung]
                weight = (rung_fraction - lower_rung) / (upper_rung - lower_rung)
                return lower_eps + weight * (upper_eps - lower_eps)

        # Fallback (shouldn't reach here)
        return self.config.eps_improve

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
        RUNG-AWARE: Parent-child comparison at same rung + track improvements per generation.

        Rules:
        1. Seeds (no parent): Always promote
        2. Mutations: Compare child@rung_i vs parent@rung_i (fair comparison)
           - Better → promote, mark improvement_this_generation = True
           - Worse → prune
           - Uses rung-specific epsilon margins (larger at small rungs due to variance)
           - Falls back to shrinkage estimate if parent@rung_i unavailable
        3. Orchestrator calls mark_generation_start() when starting new round
        """
        score = result.objective(objective_key, default=None)
        if score is None:
            return "pending"

        idx = self.current_shard_index(candidate)
        final_rung_index = len(self.shards) - 1
        sched_key = self._sched_key(candidate)
        rung_fraction = self.shards[idx]

        # Track score at this rung for future comparisons
        self._rung_scores[(sched_key, idx)] = score
        self._parent_scores[sched_key] = score
        if idx not in self._best_on_rung or score > self._best_on_rung[idx][1]:
            self._best_on_rung[idx] = (candidate, score)

        # Check if at final rung
        if idx >= final_rung_index:
            logger.debug("   ✅ ASHA: Completed at final rung (score=%s)", f"{score:.1%}")
            return "completed"

        # Extract parent info
        parent_objectives = candidate.meta.get("parent_objectives") if isinstance(candidate.meta, dict) else None
        parent_sched_key = candidate.meta.get("parent_sched_key") if isinstance(candidate.meta, dict) else None

        # SEED: No parent → always promote
        if parent_objectives is None or not isinstance(parent_objectives, dict):
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

        # MUTATION: Rung-aware parent comparison
        # Try to get parent's score at this same rung
        parent_score_at_rung = None
        if parent_sched_key:
            parent_score_at_rung = self._rung_scores.get((parent_sched_key, idx))

        # Determine comparison mode for fallback logic
        using_rung_specific_fallback = len(self.eps_per_rung) > 0

        # Fallback: Estimate parent score at this rung
        if parent_score_at_rung is None:
            parent_final_score = parent_objectives.get(objective_key)
            if parent_final_score is not None:
                if using_rung_specific_fallback:
                    # Variance tolerance mode: Use shrinkage to estimate parent@rung_i
                    # b_parent(i) = (1 - alpha) * baseline + alpha * parent_final
                    alpha = self.shrinkage_alpha.get(rung_fraction, 0.5)
                    global_baseline = 0.5  # Conservative baseline
                    parent_score_at_rung = (1 - alpha) * global_baseline + alpha * parent_final_score
                    logger.debug(
                        "   📉 Using shrinkage fallback: parent@final=%.1f%% → parent@rung_%d≈%.1f%% (α=%.2f)",
                        parent_final_score * 100,
                        idx,
                        parent_score_at_rung * 100,
                        alpha,
                    )
                else:
                    # Minimum improvement mode: Use parent's final score directly (conservative)
                    parent_score_at_rung = parent_final_score
                    logger.debug(
                        "   📊 Using parent's final score: %.1f%% (no shrinkage in min-improve mode)",
                        parent_final_score * 100,
                    )
            else:
                # No parent score at all - treat as seed
                logger.debug("   🌱 ASHA: No parent score found, treating as seed")
                self._candidate_levels[sched_key] = idx + 1
                self._pending_promotions.append(candidate)
                return "promoted"

        # Get rung-specific epsilon - interpolate if exact match not found
        eps = self._get_epsilon_for_rung(rung_fraction)

        # Determine comparison mode based on whether rung-specific thresholds are configured
        # - If eps_per_rung is empty: eps_improve is MINIMUM IMPROVEMENT required
        # - If eps_per_rung is set: epsilon is VARIANCE TOLERANCE (can be slightly worse)
        using_rung_specific = len(self.eps_per_rung) > 0

        # Compare child vs parent at same rung
        if using_rung_specific:
            # Variance tolerance mode: child >= parent - eps (tolerate being slightly worse)
            improved = score >= parent_score_at_rung - eps
        else:
            # Minimum improvement mode: child >= parent + eps (require improvement)
            improved = score >= parent_score_at_rung + eps

        # Special case: if parent hit ceiling (100%), promote equal scores
        at_ceiling = parent_score_at_rung >= 0.999
        child_at_ceiling = score >= 0.999
        if at_ceiling and child_at_ceiling:
            # Both at ceiling → promote (can't improve beyond 100%)
            improved = True
        elif using_rung_specific and at_ceiling:
            # Variance tolerance mode: allow near-ceiling scores
            if score >= parent_score_at_rung - 0.001:
                improved = True

        if improved:
            # Better than or near parent → promote
            if using_rung_specific:
                logger.debug(
                    "   ⬆️  ASHA: PROMOTED! (score: %s >= %s - %s [variance tol], rung %s -> %s)",
                    f"{score:.1%}",
                    f"{parent_score_at_rung:.1%}",
                    f"{eps:.2%}",
                    idx,
                    idx + 1,
                )
            else:
                logger.debug(
                    "   ⬆️  ASHA: PROMOTED! (score: %s >= %s + %s [min improve], rung %s -> %s)",
                    f"{score:.1%}",
                    f"{parent_score_at_rung:.1%}",
                    f"{eps:.2%}",
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
            if using_rung_specific:
                logger.debug(
                    "   ❌ ASHA: Pruned (below variance tol: %s < %s - %s at rung %s)",
                    f"{score:.1%}",
                    f"{parent_score_at_rung:.1%}",
                    f"{eps:.2%}",
                    idx,
                )
            else:
                logger.debug(
                    "   ❌ ASHA: Pruned (insufficient improvement: %s < %s + %s at rung %s)",
                    f"{score:.1%}",
                    f"{parent_score_at_rung:.1%}",
                    f"{eps:.2%}",
                    idx,
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
