# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, runtime_checkable

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
    """
    Strategy that receives the current optimizer state and proposes a new candidate or returns None.
    It may compute subsample evaluations, set trace fields in state, etc.
    The engine will handle acceptance and full eval unless the strategy already did those and encoded in metadata.
    """

    def propose(self, state: GEPAState[Any, DataId]) -> CandidateProposal | None: ...


@runtime_checkable
class BatchProposer(Protocol[DataId]):
    """Universal proposer interface consumed by the engine.

    The engine calls ``propose_batch(state, n)`` to request up to ``n``
    proposals for the current iteration. ``None`` in a slot means the
    proposer declined to produce a proposal for that slot. The engine
    processes each non-None proposal sequentially for acceptance.

    Stateless proposers (e.g. :class:`ReflectiveMutationProposer`) batch
    their internal LM calls across the ``n`` slots; stateful ones (e.g. a
    coding-agent proposer) may ignore ``n > 1`` and return a list of length 1.

    Implementations signal post-acceptance events via optional notification
    methods (``on_proposal_accepted``, ``on_candidate_found``) — the engine
    calls these via ``getattr`` with a no-op default so the protocol stays
    minimal.
    """

    def propose_batch(
        self, state: GEPAState[Any, DataId], n: int
    ) -> list[CandidateProposal[DataId] | None]: ...
