# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Sampling strategies for selecting (parent, minibatch) pairs each iteration."""

from dataclasses import dataclass
from typing import Protocol

from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.batch_sampler import BatchSampler


@dataclass
class ProposalTask:
    """One (parent, minibatch) pair to propose from."""

    parent_idx: int
    parent_candidate: dict[str, str]
    minibatch_ids: list
    minibatch: list


class SamplingStrategy(Protocol):
    """Generates one or more (parent, minibatch) tasks per iteration.

    The default (``SingleMutationSampling``) produces exactly one task,
    matching GEPA's original sequential behavior.  Swap in a different
    strategy to propose multiple candidates per iteration.
    """

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]: ...


class SingleMutationSampling:
    """1 parent, 1 mutation — default GEPA behavior."""

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        parent_idx = candidate_selector.select_candidate_idx(state)
        mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
        return [ProposalTask(parent_idx, state.program_candidates[parent_idx], mb_ids, trainset.fetch(mb_ids))]


class SameParentSampling:
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
        parent = state.program_candidates[parent_idx]
        tasks = []
        for _ in range(self.n):
            mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
            tasks.append(ProposalTask(parent_idx, parent, mb_ids, trainset.fetch(mb_ids)))
        return tasks


class IndependentSampling:
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
            mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
            tasks.append(ProposalTask(parent_idx, state.program_candidates[parent_idx], mb_ids, trainset.fetch(mb_ids)))
        return tasks


class PxNSampling:
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
            parent = state.program_candidates[parent_idx]
            for _ in range(self.n):
                mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
                tasks.append(ProposalTask(parent_idx, parent, mb_ids, trainset.fetch(mb_ids)))
        return tasks
