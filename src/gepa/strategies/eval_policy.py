"""Validation evaluation policy protocols and helpers."""

from __future__ import annotations
from abc import abstractmethod
from typing import Protocol

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState, ProgramIdx

from .batch_sampler import BatchSampler


class EvaluationPolicy(BatchSampler[DataId, DataInst]):
    """Strategy for choosing validation ids to evaluate and identifying best programs for validation instances."""

    @abstractmethod
    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Return "best" program given all validation results so far across candidates"""
        ...

    @abstractmethod
    def is_evaluation_sparse(self) -> bool:
        """ Returns true if policy will not return full validation  """

class FullEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that evaluates all validation instances every time."""

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None) -> list[DataId]:
        return list(loader.all_ids())

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        best_idx, best_score = -1, float("-inf")
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            avg = sum(scores.values()) / len(scores) if scores else float("-inf")
            if avg > best_score:
                best_score = avg
                best_idx = program_idx
        return best_idx
    
    def is_evaluation_sparse(self) -> bool:
        return False


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
]
