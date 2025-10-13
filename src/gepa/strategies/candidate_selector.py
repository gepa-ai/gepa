# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ParetoCandidateSelector(CandidateSelector):
    def __init__(self, rng: random.Random | None, frontier_type: str = "instance"):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng
        self.frontier_type = frontier_type

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return select_program_candidate_from_pareto_front(
            state.get_pareto_front_mapping(self.frontier_type),
            state.per_program_tracked_scores,
            self.rng,
        )


class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return idxmax(state.per_program_tracked_scores)
