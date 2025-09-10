# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from gepa.core.adapter import Trajectory
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector


class RoundRobinReflectionComponentSelector(ReflectionComponentSelector):
    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        pid = state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx]
        state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (pid + 1) % len(
            state.list_of_named_predictors
        )
        name = state.list_of_named_predictors[pid]
        return [name]


class AllReflectionComponentSelector(ReflectionComponentSelector):
    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        return list(candidate.keys())
