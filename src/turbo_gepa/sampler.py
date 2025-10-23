"""
Instance sampling utilities for shard selection.

The sampler balances coverage between a rotating coreset and a queue of hard
examples identified via disagreement among top candidates.
"""

from __future__ import annotations

import collections
import random
from typing import Deque, Iterable, List, Sequence


class InstanceSampler:
    """Coreset plus hardness-aware sampler."""

    def __init__(self, example_ids: Sequence[str], seed: int | None = None) -> None:
        self.example_ids = list(example_ids)
        if not self.example_ids:
            raise ValueError("InstanceSampler requires at least one example id")
        self._order = list(self.example_ids)
        self._pointer = 0
        self.hardness: Deque[str] = collections.deque(maxlen=128)
        self.random = random.Random(seed)

    def sample_shard(self, round_id: int, k: int) -> List[str]:
        """
        Return ``k`` example identifiers for the current round.

        Uses random sampling to ensure shards are representative of the full
        dataset, avoiding bias from sequential selection. This is critical for
        ASHA promotion decisions - we want partial shards to be unbiased estimates
        of full-dataset performance.
        """
        # For small k relative to dataset size, use random sampling without replacement
        # This ensures each shard is a random representative subset
        k = min(k, len(self.example_ids))
        shard = self.random.sample(self.example_ids, k)
        return shard

    def register_hard_examples(self, example_ids: Iterable[str]) -> None:
        """Record examples that triggered failures for increased sampling."""
        for example_id in example_ids:
            if example_id in self.example_ids and example_id not in self.hardness:
                self.hardness.append(example_id)

    def hardness_size(self) -> int:
        """Number of hardness-prioritized examples queued."""
        return len(self.hardness)
