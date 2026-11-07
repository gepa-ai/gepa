# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Sampling strategies for selecting (parent, minibatch) pairs each iteration."""

from dataclasses import dataclass
from typing import Generic, Protocol

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.task_curator import TaskCuratorSampler


@dataclass
class ProposalTask(Generic[DataId, DataInst]):
    """One (parent, minibatch) pair to propose from."""

    parent_idx: int
    parent_candidate: dict[str, str]
    minibatch_ids: list[DataId]
    minibatch: list[DataInst]
    proposal_id: str | None = None


class SamplingStrategy(Protocol[DataId, DataInst]):
    """Generates one or more (parent, minibatch) tasks per iteration.

    The default (``SingleMutationSampling``) produces exactly one task,
    matching GEPA's original sequential behavior.  Swap in a different
    strategy to propose multiple candidates per iteration.

    Note on trainset size: multi-task strategies draw ``n`` minibatches from
    the epoch-shuffled sampler per iteration. When
    ``n * minibatch_size > len(trainset)`` the sampler wraps around within the
    iteration, so some tasks will receive repeated minibatches. This is safe
    but reduces the diversity benefit — prefer ``n <= len(trainset) //
    minibatch_size``.
    """

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        trainset: DataLoader[DataId, DataInst],
    ) -> list[ProposalTask[DataId, DataInst]]: ...


def _sample_task(
    state: GEPAState, batch_sampler: BatchSampler, trainset: DataLoader, parent_idx: int, slot: int
) -> ProposalTask:
    proposal_id = None
    if isinstance(batch_sampler, TaskCuratorSampler):
        # Allocate identity before curation; filtering/reordering later must not
        # change which sampling decisions, history, or anchor belong to this job.
        proposal_id = f"{state.i + 1}-{slot}"
        ids = batch_sampler.next_minibatch_ids_for_candidate(
            trainset, state, candidate_idx=parent_idx, proposal_id=proposal_id
        )
    else:
        ids = batch_sampler.next_minibatch_ids(trainset, state)
    return ProposalTask(parent_idx, state.program_candidates[parent_idx], ids, trainset.fetch(ids), proposal_id)


class SingleMutationSampling(SamplingStrategy):
    """1 parent, 1 mutation — default GEPA behavior."""

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        parent_idx = candidate_selector.select_candidate_idx(state)
        return [_sample_task(state, batch_sampler, trainset, parent_idx, 0)]


class SameParentSampling(SamplingStrategy):
    """Best-of-N on the same parent with different minibatches."""

    def __init__(self, n: int):
        self.n = n

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        parent_idx = candidate_selector.select_candidate_idx(state)
        tasks = []
        for _ in range(self.n):
            tasks.append(_sample_task(state, batch_sampler, trainset, parent_idx, len(tasks)))
        return tasks


class IndependentSampling(SamplingStrategy):
    """N different parents, each with their own minibatch."""

    def __init__(self, n: int):
        self.n = n

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        tasks = []
        for _ in range(self.n):
            parent_idx = candidate_selector.select_candidate_idx(state)
            tasks.append(_sample_task(state, batch_sampler, trainset, parent_idx, len(tasks)))
        return tasks


class PxNSampling(SamplingStrategy):
    """P parents x N mutations each = P*N total tasks."""

    def __init__(self, p: int, n: int):
        self.p = p
        self.n = n

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        tasks = []
        for _ in range(self.p):
            parent_idx = candidate_selector.select_candidate_idx(state)
            for _ in range(self.n):
                tasks.append(_sample_task(state, batch_sampler, trainset, parent_idx, len(tasks)))
        return tasks
