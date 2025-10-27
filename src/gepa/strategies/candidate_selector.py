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
            state.program_at_pareto_front_valset,
            state.program_full_scores_val_set,
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
