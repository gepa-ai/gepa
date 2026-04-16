# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol

from gepa.core.data_loader import DataId
from gepa.core.state import GEPAState


@dataclass
class SubsampleEvaluation:
    """Result of evaluating a candidate on the subsample minibatch.

    Captures all outputs from ``adapter.evaluate``:

    - ``scores``: per-example numeric scores.
    - ``outputs``: per-example raw outputs (e.g. the model's response text).
    - ``objective_scores``: optional per-example multi-objective score dicts.
    - ``trajectories``: optional per-example execution traces (captured for both
      before and after evaluations).
    """

    scores: list[float]
    outputs: list[Any] = field(default_factory=list)
    objective_scores: list[dict[str, float]] | None = None
    trajectories: list[Any] | None = None


@dataclass
class CandidateProposal(Generic[DataId]):
    candidate: dict[str, str]
    parent_program_ids: list[int]
    # Optional mini-batch / subsample info
    subsample_indices: list[DataId] | None = None
    subsample_scores_before: list[float] | None = None
    subsample_scores_after: list[float] | None = None
    # Rich evaluation data (superset of scores — includes outputs, objective_scores, trajectories)
    eval_before: SubsampleEvaluation | None = None
    eval_after: SubsampleEvaluation | None = None
    # Free-form metadata for logging/trace
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ProposeNewCandidate(Protocol[DataId]):
    """Strategy that proposes candidates from the current optimizer state.

    Returns a list of accepted proposals (may be empty). The engine runs
    full valset evaluation and adds each accepted proposal to the candidate
    pool.
    """

    def propose(self, state: GEPAState[Any, DataId]) -> list[CandidateProposal]: ...
