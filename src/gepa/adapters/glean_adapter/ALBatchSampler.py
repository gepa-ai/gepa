# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.adapters.glean_adapter.al_adapter import ALDataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.strategies.batch_sampler import BatchSampler


class ALBatchSampler(BatchSampler[DataId, ALDataInst]):
    """
    Active Learning Batch Sampler that randomly selects an eval set.

    Each ALDataInst represents a complete eval set that runs in totality.
    This sampler randomly selects one eval set from the loader.

    Args:
        rng: Optional random number generator for reproducibility
    """

    def __init__(self, rng: random.Random | None = None):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def next_minibatch_ids(self, loader: DataLoader[DataId, ALDataInst], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())

        if not all_ids:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        # Randomly select one eval set
        selected_id = self.rng.choice(all_ids)

        return [selected_id]
