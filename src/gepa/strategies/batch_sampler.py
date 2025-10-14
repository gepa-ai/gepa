# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.proposer.reflective_mutation.base import BatchSampler


class EpochShuffledBatchSampler(BatchSampler):
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
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        for i in self.shuffled_ids:
            self.id_freqs[i] += 1

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], iteration: int) -> list[int]:
        base_idx = iteration * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)
        if curr_epoch > self.epoch:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]
