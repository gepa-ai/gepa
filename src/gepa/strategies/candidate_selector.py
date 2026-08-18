# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ParetoCandidateSelector(CandidateSelector):
    def __init__(self, rng: random.Random | None):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return select_program_candidate_from_pareto_front(
            state.get_pareto_front_mapping(),
            state.per_program_tracked_scores,
            self.rng,
        )


class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return idxmax(state.program_full_scores_val_set)


class EpsilonGreedyCandidateSelector(CandidateSelector):
    def __init__(self, epsilon: float, rng: random.Random | None):
        assert 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, len(state.program_candidates) - 1)
        else:
            return idxmax(state.program_full_scores_val_set)


class BatchLexicaseCandidateSelector(CandidateSelector):
    """Batch lexicase selection over per-instance validation scores.

    Shuffles the validation instances, groups them into batches of `batch_size`, and
    filters the candidate pool batch by batch, keeping only candidates with the maximal
    batch score sum, until a single candidate survives (remaining ties are broken
    uniformly). Only instances scored for every candidate participate.

    Every candidate can win under some batch ordering, so no candidate is ever assigned
    zero selection probability. This differs from Pareto-frontier sampling, whose dominance
    pruning can exclude the highest-aggregate candidates entirely when per-instance tie sets
    are large (e.g. binary metrics). `batch_size` tunes selection pressure: 1 recovers plain
    lexicase selection (Helmuth, Spector & Matheson, 2015); a batch spanning the whole
    valset recovers argmax by aggregate score. Batching follows Aenugu & Spector (2019).
    """

    def __init__(self, batch_size: int = 8, rng: random.Random | None = None):
        assert batch_size > 0
        self.batch_size = batch_size
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        subscores = state.prog_candidate_val_subscores
        assert len(subscores) == len(state.program_candidates)
        num_candidates = len(subscores)
        if num_candidates == 1:
            return 0

        common_ids = set(subscores[0].keys())
        for candidate_scores in subscores[1:]:
            common_ids &= candidate_scores.keys()
        if not common_ids:
            return self.rng.randrange(num_candidates)

        # Sort before shuffling so selection depends only on the rng seed, not set iteration order.
        instance_ids = sorted(common_ids, key=repr)
        self.rng.shuffle(instance_ids)

        survivors = list(range(num_candidates))
        for start in range(0, len(instance_ids), self.batch_size):
            batch = instance_ids[start : start + self.batch_size]
            batch_scores = [sum(subscores[idx][i] for i in batch) for idx in survivors]
            best = max(batch_scores)
            survivors = [idx for idx, score in zip(survivors, batch_scores, strict=False) if score >= best - 1e-9]
            if len(survivors) == 1:
                return survivors[0]
        return self.rng.choice(survivors)


class UnprunedParetoCandidateSelector(CandidateSelector):
    """Samples proportional to per-instance Pareto-front membership, skipping the
    dominance-pruning step of `ParetoCandidateSelector`. Serves as an ablation control
    for isolating the effect of dominance pruning on parent selection.
    """

    def __init__(self, rng: random.Random | None = None):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        sampling_list = [prog_idx for front in state.get_pareto_front_mapping().values() for prog_idx in front]
        assert len(sampling_list) > 0
        return self.rng.choice(sampling_list)


class TopKParetoCandidateSelector(CandidateSelector):
    """Pareto selection restricted to the top K programs by aggregate score."""

    def __init__(self, k: int, rng: random.Random | None):
        assert k > 0
        self.k = k
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        # Get top K program indices by aggregate score
        scores = state.per_program_tracked_scores
        top_k_indices = set(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k])

        # Filter pareto front mapping to only include top K programs
        pareto_mapping = state.get_pareto_front_mapping()
        filtered_mapping = {}
        for key, prog_set in pareto_mapping.items():
            filtered = prog_set & top_k_indices
            if filtered:
                filtered_mapping[key] = filtered

        if not filtered_mapping:
            # Fallback: pick the best program overall
            return idxmax(scores)

        return select_program_candidate_from_pareto_front(filtered_mapping, scores, self.rng)
