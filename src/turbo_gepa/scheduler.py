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
    """Variance-aware scheduler config with rung-specific promotion thresholds."""
    shards: Sequence[float]

    # Variance-aware promotion: rung-specific tolerance values
    # Higher tolerance at smaller rungs accounts for score noise with fewer examples
    variance_tolerance: dict[float, float]

    # Shrinkage coefficients for estimating parent@rung_i from parent@final when unavailable
    # Higher alpha = more weight to parent's final score (less shrinkage toward baseline)
    shrinkage_alpha: dict[float, float]

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

        # Per-rung score tracking: (candidate_key, rung_idx) -> score
        self._rung_scores: dict[tuple[str, int], float] = {}

        # Variance-aware tolerance and shrinkage (required parameters)
        self.variance_tolerance = dict(config.variance_tolerance)
        self.shrinkage_alpha = dict(config.shrinkage_alpha)

        # Convergence tracking per rung: generation + evaluation counters
        # rung_idx -> {
        #   "stagnant_generations": int,
        #   "improvement_this_gen": bool,
        #   "stagnant_evals": int,
        # }
        self._rung_generations: dict[int, dict[str, int | bool]] = {
            i: {
                "stagnant_generations": 0,
                "improvement_this_gen": False,
                "stagnant_evals": 0,
            }
            for i in range(len(self.shards))
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

    def _get_tolerance_for_rung(self, rung_fraction: float) -> float:
        """Get variance tolerance for a rung, interpolating if exact match not found."""
        # Try exact match first
        if rung_fraction in self.variance_tolerance:
            return self.variance_tolerance[rung_fraction]

        # Find closest rung fraction with interpolation
        sorted_rungs = sorted(self.variance_tolerance.keys())
        if not sorted_rungs:
            # Fallback: use 2% tolerance (shouldn't happen if config is valid)
            return 0.02

        # If smaller than smallest rung, use smallest rung's tolerance
        if rung_fraction <= sorted_rungs[0]:
            return self.variance_tolerance[sorted_rungs[0]]

        # If larger than largest rung, use largest rung's tolerance
        if rung_fraction >= sorted_rungs[-1]:
            return self.variance_tolerance[sorted_rungs[-1]]

        # Interpolate between two nearest rungs
        for i in range(len(sorted_rungs) - 1):
            lower_rung = sorted_rungs[i]
            upper_rung = sorted_rungs[i + 1]
            if lower_rung <= rung_fraction <= upper_rung:
                # Linear interpolation
                lower_tol = self.variance_tolerance[lower_rung]
                upper_tol = self.variance_tolerance[upper_rung]
                weight = (rung_fraction - lower_rung) / (upper_rung - lower_rung)
                return lower_tol + weight * (upper_tol - lower_tol)

        # Fallback (shouldn't reach here)
        return 0.02

    def current_shard_index(self, candidate: Candidate) -> int:
        return self._candidate_levels.get(self._sched_key(candidate), 0)

    def mark_generation_start(self, rung_idx: int) -> None:
        """
        Call this when orchestrator starts a new generation of mutations for a rung.
        This completes the previous generation and checks for convergence.
        """
        if rung_idx not in self._rung_generations:
            return

        info = self._rung_generations[rung_idx]
        final_rung_index = len(self.shards) - 1
        at_final_rung = rung_idx >= final_rung_index
        stagnant_gens = int(info["stagnant_generations"])
        improved_this_gen = bool(info["improvement_this_gen"])
        stagnant_evals = int(info.get("stagnant_evals", 0))

        if improved_this_gen:
            # Had improvement → reset counter
            info["stagnant_generations"] = 0
            info["stagnant_evals"] = 0
            info["improvement_this_gen"] = False
        else:
            # No improvement → increment counter
            new_count = stagnant_gens + 1
            info["stagnant_generations"] = new_count
            info.setdefault("stagnant_evals", 0)
            info["improvement_this_gen"] = False

            # Check for convergence
            eval_threshold = 20 if rung_idx >= final_rung_index else 0
            if new_count >= self.config.patience_generations and stagnant_evals >= eval_threshold:
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
                        info["stagnant_generations"] = 0
                        info["stagnant_evals"] = 0
                        info["improvement_this_gen"] = False
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
        self._rung_generations = {
            i: {
                "stagnant_generations": 0,
                "improvement_this_gen": False,
                "stagnant_evals": 0,
            }
            for i in range(len(self.shards))
        }
        self._best_on_rung.clear()
        self.converged = False

    def reset_final_rung_convergence(self) -> None:
        """Clear convergence flag/count for the final rung so optimization can continue."""
        final_idx = len(self.shards) - 1
        if final_idx >= 0:
            self._rung_generations[final_idx] = {
                "stagnant_generations": 0,
                "improvement_this_gen": False,
                "stagnant_evals": 0,
            }
        self.converged = False

    def current_shard_fraction(self, candidate: Candidate) -> float:
        idx = self.current_shard_index(candidate)
        return self.shards[idx]

    def record(self, candidate: Candidate, result: EvalResult, objective_key: str) -> str:
        """
        Variance-aware promotion: Parent-child comparison at same rung with variance tolerance.

        Rules:
        1. Seeds (no parent): Always promote
        2. Mutations: Compare child@rung_i vs parent@rung_i (fair comparison)
           - Promote if: child_score >= parent_score - tolerance
           - This accounts for score variance at each rung (higher tolerance at smaller rungs)
           - Falls back to shrinkage estimate if parent@rung_i unavailable
        3. Orchestrator calls mark_generation_start() when starting new round
        """
        score = result.objective(objective_key, default=None)
        if score is None:
            return "pending"

        idx = self.current_shard_index(candidate)
        if not self.shards:
            return "completed"
        max_idx = len(self.shards) - 1
        if idx < 0:
            idx = 0
        elif idx > max_idx:
            idx = max_idx
        final_rung_index = len(self.shards) - 1
        sched_key = self._sched_key(candidate)
        rung_fraction = self.shards[idx]
        at_final_rung = idx >= final_rung_index

        # Track score at this rung for future comparisons
        self._rung_scores[(sched_key, idx)] = score
        self._parent_scores[sched_key] = score
        if idx not in self._best_on_rung or score > self._best_on_rung[idx][1]:
            self._best_on_rung[idx] = (candidate, score)
            info = self._rung_generations.get(idx)
            if info is not None:
                info["improvement_this_gen"] = True
                info["stagnant_evals"] = 0
        else:
            info = self._rung_generations.get(idx)
            if info is not None:
                info["stagnant_evals"] = int(info.get("stagnant_evals", 0)) + 1

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
                info = self._rung_generations[idx]
                info["improvement_this_gen"] = True
                info["stagnant_evals"] = 0
            return "promoted"

        # MUTATION: Variance-aware parent comparison
        # Try to get parent's score at this same rung
        parent_score_at_rung = None
        if parent_sched_key:
            parent_score_at_rung = self._rung_scores.get((parent_sched_key, idx))

        # Check if orchestrator supplied explicit rung scores via metadata
        if parent_score_at_rung is None:
            meta = candidate.meta if isinstance(candidate.meta, dict) else None
            if meta:
                parent_rung_scores = meta.get("parent_rung_scores")
                if isinstance(parent_rung_scores, dict):
                    rung_val = parent_rung_scores.get(rung_fraction)
                    if isinstance(rung_val, (int, float)):
                        parent_score_at_rung = float(rung_val)

        # Fallback: Estimate parent score at this rung using shrinkage
        if parent_score_at_rung is None:
            parent_final_score = parent_objectives.get(objective_key)
            if parent_final_score is not None:
                alpha = self.shrinkage_alpha.get(rung_fraction, 0.5)
                global_baseline = 0.5  # Conservative baseline
                parent_score_at_rung = (1 - alpha) * global_baseline + alpha * parent_final_score
                logger.debug(
                    "   📉 Using shrinkage: parent@final=%.1f%% → parent@rung_%d≈%.1f%% (α=%.2f)",
                    parent_final_score * 100,
                    idx,
                    parent_score_at_rung * 100,
                    alpha,
                )
            else:
                logger.debug("   🌱 ASHA: No parent score found, treating as seed")
                self._candidate_levels[sched_key] = idx + 1
                self._pending_promotions.append(candidate)
                return "promoted"

        # Get rung-specific variance tolerance
        tolerance = self._get_tolerance_for_rung(rung_fraction)

        # FINAL RUNG: always mark completed; scheduler does not gate on
        # statistical confidence here. Any CI-style checks are handled
        # upstream by the evaluator/coverage logic.
        if at_final_rung:
            meta = candidate.meta if isinstance(candidate.meta, dict) else None
            if meta is not None:
                meta["_coverage_fraction"] = float(result.coverage_fraction or 0.0)
                meta["_samples_seen"] = int(result.n_examples or 0)
            logger.debug(
                "   ✅ ASHA: Final rung coverage=%.1f%%, samples=%s, score=%.1f%% (parent_thresh=%.1f%%)",
                (result.coverage_fraction or 0.0) * 100,
                result.n_examples,
                (parent_score_at_rung - tolerance) * 100,
            )
            return "completed"

        # Variance-aware comparison: child >= parent - tolerance
        # This tolerates being slightly worse due to score noise at smaller rungs
        improved = score >= parent_score_at_rung - tolerance

        # Special case: if parent hit ceiling (100%), promote equal scores
        at_ceiling = parent_score_at_rung >= 0.999
        child_at_ceiling = score >= 0.999
        if at_ceiling and child_at_ceiling:
            # Both at ceiling → promote (can't improve beyond 100%)
            improved = True
        elif at_ceiling and score >= parent_score_at_rung - 0.001:
            # Near-ceiling: allow tiny tolerance
            improved = True

        if improved:
            # Within variance tolerance → promote
            logger.debug(
                "   ⬆️  ASHA: PROMOTED! (score: %s >= %s - %s [variance tol], rung %s -> %s)",
                f"{score:.1%}",
                f"{parent_score_at_rung:.1%}",
                f"{tolerance:.2%}",
                idx,
                idx + 1,
            )
            self._candidate_levels[sched_key] = idx + 1
            self._pending_promotions.append(candidate)
            # Mark improvement on this rung
            if idx in self._rung_generations:
                info = self._rung_generations[idx]
                info["improvement_this_gen"] = True
                info["stagnant_evals"] = 0
            return "promoted"
        else:
            # Below variance tolerance → prune
            logger.debug(
                "   ❌ ASHA: Pruned (below variance tol: %s < %s - %s at rung %s)",
                f"{score:.1%}",
                f"{parent_score_at_rung:.1%}",
                f"{tolerance:.2%}",
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
