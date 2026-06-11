# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol

from gepa.core.data_loader import DataId
from gepa.core.state import GEPAState
from gepa.strategies.acceptance import AcceptanceCriterion, StrictImprovementAcceptance
from gepa.strategies.proposal_sampling import ProposalTask, SamplingStrategy, SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements, SelectionStrategy


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


class Proposer(ABC, Generic[DataId]):
    """Abstract base for all GEPA proposers.

    Subclasses MUST implement ``_mutate(tasks, state) -> list[CandidateProposal]``.
    That is the only required method, it receives a list of pre-sampled tasks
    (each with a parent candidate and minibatch) and returns proposals.

    Everything else, sampling parents/minibatches, filtering proposals by the
    acceptance criterion — is handled here and does not need to be re-implemented, unless desired.
    The engine calls ``propose(state)`` and gets back a list of accepted proposals.

    Pipeline (runs automatically on every ``propose(state)`` call):
      1. ``sampling_strategy.sample_tasks()`` — select parents + minibatches
      2. ``_mutate(tasks, state)``            — **MUST implement** (evaluation + mutation)
      3. ``_select(proposals, state)``        — **optionally override** (default: filter by acceptance criterion)

    All three steps have sensible defaults that match original GEPA behavior
    (1 parent, 1 minibatch, strict improvement acceptance), so passing no
    strategy arguments produces the same result as the old single-proposal flow.
    Pass custom strategies only when you need to change sampling or selection.
    """

    def __init__(
        self,
        trainset: Any,
        candidate_selector: Any,
        batch_sampler: Any,
        acceptance_criterion: AcceptanceCriterion | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        selection_strategy: SelectionStrategy | None = None,
    ):
        self.trainset = trainset
        self.candidate_selector = candidate_selector
        self.batch_sampler = batch_sampler
        self.acceptance_criterion: AcceptanceCriterion = acceptance_criterion or StrictImprovementAcceptance()
        self.sampling_strategy: SamplingStrategy = sampling_strategy or SingleMutationSampling()
        self.selection_strategy: SelectionStrategy = selection_strategy or AllImprovements()

    def propose(self, state: GEPAState[Any, DataId]) -> list[CandidateProposal[DataId]]:
        tasks = self.sampling_strategy.sample_tasks(
            state, self.candidate_selector, self.batch_sampler, self.trainset
        )
        proposals = self._mutate(tasks, state)
        return self._select(proposals, state)

    @abstractmethod
    def _mutate(
        self, tasks: list[ProposalTask], state: GEPAState[Any, DataId]
    ) -> list[CandidateProposal[DataId]]:
        """Run evaluation + mutation for each task. MUST be implemented by subclasses.

        Args:
            tasks: Pre-sampled (parent, minibatch) pairs from the sampling strategy.
                   Each task has: parent_idx, parent_candidate, minibatch_ids, minibatch.
            state: Current optimizer state (read-only during parallel execution).

        Returns:
            List of proposals (no Nones). Failed/skipped tasks should simply be
            excluded from the list rather than returned as None.
        """
        ...

    def _select(
        self, proposals: list[CandidateProposal[DataId]], state: GEPAState[Any, DataId]
    ) -> list[CandidateProposal[DataId]]:
        """Filter proposals down to those worth running full eval on.

        Default: delegates to the injected ``selection_strategy`` (e.g. AllImprovements).
        Override this method for custom selection logic (agentic selection, LLM-based
        ranking, etc.) without needing to pass a SelectionStrategy object.
        """
        return self.selection_strategy.select(proposals, state, self.acceptance_criterion)
