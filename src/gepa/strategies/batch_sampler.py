# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gepa.core.adapter import DataInst, EvaluationBatch
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal


class BatchSampler(Protocol[DataId, DataInst]):
    """Yields the minibatch of trainset ids to propose from.

    Multi-proposal sampling strategies call ``next_minibatch_ids`` once per
    task within a single iteration (``state.i`` unchanged between calls).
    Implementations should return a *different* minibatch on each repeated
    call within an iteration, so parallel proposal tasks don't all share one
    minibatch.
    """

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class BatchSamplerError(ValueError):
    """Fatal sampling/evidence contract violation; continuing could corrupt history."""


@runtime_checkable
class CandidateAwareBatchSampler(Protocol[DataId, DataInst]):
    """Optional sampling interface for state scoped to the explicitly selected parent.

    Ordinary samplers continue to receive ``next_minibatch_ids`` unchanged.
    """

    def next_minibatch_ids_for_candidate(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, *, candidate_idx: int
    ) -> list[DataId]: ...


@runtime_checkable
class EvaluationAwareBatchSampler(Protocol):
    """Optional hook receiving each sampled parent's evaluation, even without a proposal.

    Implementations may set ``requires_sequential_sampling = True`` when each
    sampling decision must see all previous evaluation/proposal outcomes.
    """

    def observe_evaluation(
        self,
        *,
        iteration: int,
        task_ids: Sequence[DataId],
        evaluation: EvaluationBatch,
        state: GEPAState,
    ) -> None: ...


@runtime_checkable
class ProposalAwareBatchSampler(Protocol):
    """Optional hook receiving evaluated proposals and their final selection outcome.

    ``accepted`` means the proposal was selected and added to the candidate pool,
    not merely that its minibatch score passed the acceptance criterion.
    ``candidate_idx`` identifies the newly registered child, or is None on rejection.
    """

    def observe_proposal(
        self,
        *,
        iteration: int,
        proposal: CandidateProposal,
        accepted: bool,
        state: GEPAState,
        candidate_idx: int | None = None,
    ) -> None: ...


ReflectiveDatasetBuilder = Callable[
    [dict[str, str], EvaluationBatch, list[str]], Mapping[str, Sequence[Mapping[str, Any]]]
]


@runtime_checkable
class ReflectionAwareBatchSampler(Protocol):
    """Optional reflection assembly before the user's reflective-dataset enricher.

    The sampler receives the original adapter builder and the aligned evaluation
    and task IDs. It owns any source tracking and enrichment; adapters need no
    additional metadata or interface changes. Ordinary samplers keep the
    original whole-minibatch builder call. Components without records are not
    proposed; if all selected components are empty, the proposal is skipped.
    """

    def build_reflective_dataset(
        self,
        *,
        candidate: dict[str, str],
        task_ids: Sequence[DataId],
        evaluation: EvaluationBatch,
        components_to_update: list[str],
        make_reflective_dataset: ReflectiveDatasetBuilder,
        state: GEPAState,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None):
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.last_trainset_size = 0
        self._current_iteration: int | None = None
        self._calls_in_iteration = 0
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        # Repeated calls within one iteration (multi-proposal sampling
        # strategies request one minibatch per task) advance one chunk each,
        # so tasks in the same iteration get distinct minibatches. The first
        # call of an iteration is unchanged from the classic behavior. Chunks
        # wrap around, so distinctness holds only while the iteration's call
        # count stays within len(shuffled_ids) / minibatch_size.
        if state.i == self._current_iteration:
            self._calls_in_iteration += 1
        else:
            self._current_iteration = state.i
            self._calls_in_iteration = 0

        base_idx = state.i * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)

        needs_refresh = not self.shuffled_ids or trainset_size != self.last_trainset_size or curr_epoch > self.epoch
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        # The epoch bookkeeping above uses the un-offset base_idx (constant
        # within an iteration), so repeat calls never trigger a reshuffle and
        # the shuffle sequence stays identical to the single-call path.
        base_idx = (base_idx + self._calls_in_iteration * self.minibatch_size) % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]
