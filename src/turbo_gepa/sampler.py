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

        Hardness-aware sampling: reserves up to 25% of the shard for hard examples
        (those that caused failures) to help focus reflection on challenging cases.
        """
        k = min(k, len(self.example_ids))

        # Reserve up to 25% of shard for hard examples (min 1 if hardness deque non-empty)
        hardness_count = 0
        if self.hardness:
            hardness_count = min(len(self.hardness), max(1, k // 4))

        # Sample from hardness deque (take from the end for most recent hard examples)
        hard_ids = []
        if hardness_count > 0:
            # Sample without replacement from hardness deque
            # Convert to list first to enable random sampling
            hardness_list = list(self.hardness)
            hard_ids = self.random.sample(hardness_list, min(hardness_count, len(hardness_list)))

        # Fill remaining slots with random sampling from non-hardness examples
        remaining = k - len(hard_ids)
        if remaining > 0:
            # Sample from all examples, excluding the entire hardness deque to maintain the 25% cap
            remaining_pool = [ex_id for ex_id in self.example_ids if ex_id not in self.hardness]
            if remaining_pool:
                random_ids = self.random.sample(remaining_pool, min(remaining, len(remaining_pool)))
            else:
                random_ids = []
        else:
            random_ids = []

        shard = hard_ids + random_ids
        return shard

    def register_hard_examples(self, example_ids: Iterable[str]) -> None:
        """Record examples that triggered failures for increased sampling."""
        for example_id in example_ids:
            if example_id in self.example_ids and example_id not in self.hardness:
                self.hardness.append(example_id)

    def hardness_size(self) -> int:
        """Number of hardness-prioritized examples queued."""
        return len(self.hardness)
