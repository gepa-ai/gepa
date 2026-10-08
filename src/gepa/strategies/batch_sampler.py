# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState


class BatchSampler(Protocol[DataId, DataInst]):
    """Yields the minibatch of trainset ids to propose from.

    Multi-proposal sampling strategies call ``next_minibatch_ids`` once per
    task within a single iteration (``state.i`` unchanged between calls).
    Implementations should return a *different* minibatch on each repeated
    call within an iteration, so parallel proposal tasks don't all share one
    minibatch.

    A sampler may also implement ``observe_evaluation(ids, scores)``. Reflective
    mutation calls it with deduplicated parent training evaluations, including
    batches that are subsequently skipped for perfect scores. Validation scores
    are never supplied to this optional method.
    """

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None):
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


class DifficultyAwareBatchSampler(BatchSampler[DataId, DataInst]):
    """Favor examples with low recent parent scores while retaining epoch coverage.

    At least one slot per batch comes from a shuffled traversal of the entire
    training set. The other slots are sampled without replacement, weighted by
    the gap between an example's latest observed score and the highest observed
    score. Unseen examples receive the largest current gap. Equal scores fall
    back to uniform sampling. This assumes higher metric scores are better.

    Observations are local to this sampler and training loader. Replacing the
    loader or starting a new optimization resets them; growing a loader retains
    observations for surviving ids. No extra evaluations are requested.
    """

    def __init__(
        self,
        minibatch_size: int,
        *,
        exploration_fraction: float = 0.25,
        rng: random.Random | None = None,
    ):
        if not isinstance(minibatch_size, int) or isinstance(minibatch_size, bool) or minibatch_size < 1:
            raise ValueError("minibatch_size must be a positive integer.")
        if (
            not isinstance(exploration_fraction, (int, float))
            or isinstance(exploration_fraction, bool)
            or not math.isfinite(exploration_fraction)
            or not 0 < exploration_fraction <= 1
        ):
            raise ValueError("exploration_fraction must be finite and in (0, 1].")
        self.minibatch_size = minibatch_size
        self.exploration_fraction = exploration_fraction
        self.rng = rng if rng is not None else random.Random(0)
        self._loader: DataLoader[DataId, DataInst] | None = None
        self._state: GEPAState | None = None
        self._ids: list[DataId] = []
        self._scores: dict[DataId, float] = {}
        self._exploration_ids: list[DataId] = []
        self._cursor = 0
        self._iteration: int | None = None

    @property
    def observed_scores(self) -> dict[DataId, float]:
        """Snapshot of the latest finite training scores seen for each id."""
        return dict(self._scores)

    def observe_evaluation(self, ids: Sequence[DataId], scores: Sequence[float]) -> None:
        if len(ids) != len(scores):
            raise ValueError("Training ids and scores must have the same length.")
        allowed = set(self._ids)
        for data_id, score in zip(ids, scores, strict=True):
            if data_id in allowed and math.isfinite(score):
                self._scores[data_id] = score

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise ValueError("Cannot sample a minibatch from an empty loader.")
        allowed_ids = set(all_ids)
        if len(allowed_ids) != len(all_ids):
            raise ValueError("Training loader ids must be unique.")
        new_run = self._iteration is not None and state.i < self._iteration
        if loader is not self._loader or state is not self._state or new_run:
            self._scores.clear()
            self._ids = []
            self._loader = loader
            self._state = state
        if all_ids != self._ids:
            self._ids = all_ids
            self._scores = {data_id: score for data_id, score in self._scores.items() if data_id in allowed_ids}
            self._exploration_ids = []
            self._cursor = 0
        self._iteration = state.i

        size = min(self.minibatch_size, len(all_ids))
        exploration_size = max(1, math.ceil(size * self.exploration_fraction))
        selected: list[DataId] = []
        used: set[DataId] = set()
        while len(selected) < exploration_size:
            if self._cursor >= len(self._exploration_ids):
                self._exploration_ids = list(all_ids)
                self.rng.shuffle(self._exploration_ids)
                self._cursor = 0
            data_id = self._exploration_ids[self._cursor]
            self._cursor += 1
            if data_id not in used:
                selected.append(data_id)
                used.add(data_id)

        best_score = max(self._scores.values(), default=0.0)
        scale = max((abs(score) for score in self._scores.values()), default=1.0) or 1.0
        gaps = {data_id: max(0.0, best_score / scale - score / scale) for data_id, score in self._scores.items()}
        unseen_gap = max(gaps.values(), default=1.0) or 1.0
        remaining = [data_id for data_id in all_ids if data_id not in used]
        while len(selected) < size:
            weights = [gaps.get(data_id, unseen_gap) for data_id in remaining]
            chosen = (
                self.rng.choices(remaining, weights=weights, k=1)[0] if any(weights) else self.rng.choice(remaining)
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return selected
