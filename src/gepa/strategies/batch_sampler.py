# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter
from typing import Any, Protocol

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
    """

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    - Refresh when available ids change; order-only changes retain the current shuffle
    - Shuffle position is saved with the run state so a resumed run finishes the current epoch
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

    def get_sampler_state(self) -> dict[str, Any]:
        """Return the shuffle position to persist in a checkpoint.

        The RNG is not included; it is shared with other strategies and is
        recreated from ``seed`` on resume.
        """
        return {
            "minibatch_size": self.minibatch_size,
            "shuffled_ids": list(self.shuffled_ids),
            "epoch": self.epoch,
            "id_freqs": Counter(self.id_freqs),
            "last_trainset_size": self.last_trainset_size,
        }

    def set_sampler_state(self, sampler_state: dict[str, Any]) -> None:
        """Restore a shuffle position saved by :meth:`get_sampler_state`.

        State saved with a different ``minibatch_size`` is ignored and the
        sampler starts a fresh shuffle. If the loader's IDs changed since the
        save, the next call refreshes the shuffle as usual.
        """
        if sampler_state.get("minibatch_size") != self.minibatch_size:
            return
        self.shuffled_ids = list(sampler_state["shuffled_ids"])
        self.epoch = sampler_state["epoch"]
        self.id_freqs = Counter(sampler_state["id_freqs"])
        self.last_trainset_size = sampler_state["last_trainset_size"]
        self._current_iteration = None
        self._calls_in_iteration = 0

    def _update_shuffled(self, all_ids: list[DataId]):
        trainset_size = len(all_ids)
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
        all_ids = list(loader.all_ids())
        trainset_size = len(all_ids)
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

        needs_refresh = (
            not self.shuffled_ids
            or trainset_size != self.last_trainset_size
            or curr_epoch > self.epoch
            or set(all_ids) != set(self.shuffled_ids)
        )
        if needs_refresh:
            self._update_shuffled(all_ids)
            # Derive the epoch from the new shuffle. A sampler created partway
            # through a run (e.g. resuming an older checkpoint without saved
            # sampler state) would otherwise record epoch 0 and reshuffle on
            # every following iteration of the current epoch.
            self.epoch = base_idx // len(self.shuffled_ids)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        # The epoch bookkeeping above uses the un-offset base_idx (constant
        # within an iteration), so repeat calls never trigger a reshuffle and
        # the shuffle sequence stays identical to the single-call path.
        base_idx = (base_idx + self._calls_in_iteration * self.minibatch_size) % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]
