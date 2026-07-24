# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Learnability-weighted batch sampler.

Prioritizes training examples where candidate performance varies most —
these are the examples most likely to benefit from prompt improvements.
Examples where all candidates score similarly (all high or all low) provide
less signal for the reflection LM.
"""

import random
from typing import Any

from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler


class LearnabilityBatchSampler(BatchSampler[DataId, Any]):
    """Sample training examples weighted by how much candidates disagree on them.

    For each training example that appears in the validation scores, compute
    the **variance** of scores across all candidates that have been evaluated
    on it.  Examples with higher variance are sampled with higher probability,
    because they are the ones where better prompts can make the most difference.

    Falls back to uniform sampling (``EpochShuffledBatchSampler``) for the
    first few iterations when fewer than ``min_candidates`` have been evaluated.

    Parameters
    ----------
    minibatch_size:
        Number of examples per minibatch.
    min_candidates:
        Minimum number of candidates required before learnability weighting
        kicks in.  Before this threshold, falls back to epoch-shuffled.
    temperature:
        Controls how aggressively to favor high-variance examples.

        - ``1.0`` (default) — sample proportional to variance.
        - ``> 1.0`` — flatten the distribution (closer to uniform).
        - ``< 1.0`` — sharpen (focus heavily on highest-variance examples).
        - ``0.0`` — always pick the top-K highest-variance examples.

    rng:
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        minibatch_size: int,
        min_candidates: int = 3,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ):
        self.minibatch_size = minibatch_size
        self.min_candidates = min_candidates
        self.temperature = temperature
        self.rng = rng or random.Random(0)
        self._fallback = EpochShuffledBatchSampler(minibatch_size=minibatch_size, rng=self.rng)

    def _compute_learnability(self, state: GEPAState) -> dict[DataId, float]:
        """Compute per-example score variance across candidates.

        Returns a dict mapping example_id -> variance.  Only includes
        examples scored by at least ``min_candidates`` candidates.
        """
        # Collect all scores per example across candidates
        scores_by_example: dict[DataId, list[float]] = {}
        for candidate_scores in state.prog_candidate_val_subscores:
            for example_id, score in candidate_scores.items():
                scores_by_example.setdefault(example_id, []).append(score)

        # Compute variance for examples with enough data
        learnability: dict[DataId, float] = {}
        for example_id, scores in scores_by_example.items():
            if len(scores) < self.min_candidates:
                continue
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            learnability[example_id] = variance

        return learnability

    def next_minibatch_ids(self, loader: DataLoader[DataId, Any], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        learnability = self._compute_learnability(state)

        # Filter to training IDs that have learnability scores
        scored_ids = [eid for eid in all_ids if eid in learnability]

        # Fall back if not enough data yet
        if len(scored_ids) < self.minibatch_size:
            return self._fallback.next_minibatch_ids(loader, state)

        # Compute sampling weights
        if self.temperature == 0.0:
            # Deterministic: pick top-K by variance
            scored_ids.sort(key=lambda eid: learnability[eid], reverse=True)
            return scored_ids[: self.minibatch_size]

        # Weighted sampling
        variances = [learnability[eid] for eid in scored_ids]
        max_var = max(variances) if variances else 1.0

        if max_var == 0.0:
            # All variances are zero — fall back to uniform
            return self._fallback.next_minibatch_ids(loader, state)

        # Normalize and apply temperature
        weights = []
        for v in variances:
            normalized = v / max_var  # in [0, 1]
            # Add small epsilon so zero-variance examples still have nonzero weight
            w = (normalized + 0.01) ** (1.0 / max(self.temperature, 1e-6))
            weights.append(w)

        # Sample without replacement
        selected: list[DataId] = []
        available_indices = list(range(len(scored_ids)))
        available_weights = list(weights)

        for _ in range(min(self.minibatch_size, len(scored_ids))):
            total = sum(available_weights)
            if total <= 0:
                break
            r = self.rng.random() * total
            cumulative = 0.0
            chosen_pos = 0
            for pos, w in enumerate(available_weights):
                cumulative += w
                if cumulative >= r:
                    chosen_pos = pos
                    break
            selected.append(scored_ids[available_indices[chosen_pos]])
            available_indices.pop(chosen_pos)
            available_weights.pop(chosen_pos)

        # If we couldn't fill the minibatch (unlikely), pad from remaining
        if len(selected) < self.minibatch_size:
            remaining = [eid for eid in all_ids if eid not in set(selected)]
            self.rng.shuffle(remaining)
            selected.extend(remaining[: self.minibatch_size - len(selected)])

        return selected
