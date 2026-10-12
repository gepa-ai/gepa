# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from collections import Counter
from collections.abc import Sequence
from time import monotonic
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


class DynamicBatchSampler(EpochShuffledBatchSampler[DataId, DataInst]):
    """Profile ordinary optimizer iterations and select a bounded batch-size plateau.

    Trial iteration delays are converted to epoch estimates using the number of
    examples sampled in that iteration. A log-linear fit of T(bs) = A * bs**-alpha
    selects the point whose slope is slope_threshold of the fitted peak slope
    (at the smallest measured size). The default threshold is ComBEE's 1.6%.

    Profiling consumes normal optimization steps, without extra evaluations.
    Repeated calls at the same state.i share a trial size and are timed together.
    The shuffle cursor survives size changes so unfinished epochs retain coverage.
    A changed loader or dataset size restarts profiling. Flat/increasing curves or
    fewer than two usable timings retain the smallest trial size.
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        min_batch_size: int = 1,
        profile_batch_sizes: Sequence[int] | None = None,
        slope_threshold: float = 0.016,
        rng: random.Random | None = None,
    ):
        for size in (min_batch_size, max_batch_size):
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError("Batch size bounds must be positive integers.")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be at least min_batch_size.")
        if (
            not isinstance(slope_threshold, (int, float))
            or isinstance(slope_threshold, bool)
            or not math.isfinite(slope_threshold)
            or not 0 < slope_threshold <= 1
        ):
            raise ValueError("slope_threshold must be finite and in (0, 1].")
        if profile_batch_sizes is None:
            sizes = [min_batch_size]
            while sizes[-1] < max_batch_size:
                sizes.append(min(max_batch_size, sizes[-1] * 2))
        else:
            sizes = list(profile_batch_sizes)
            if not sizes or any(
                not isinstance(size, int) or isinstance(size, bool) or not min_batch_size <= size <= max_batch_size
                for size in sizes
            ):
                raise ValueError("profile_batch_sizes must be nonempty integer sizes within the batch bounds.")
            sizes = sorted(set(sizes))
        super().__init__(minibatch_size=sizes[0], rng=rng)
        self.max_batch_size = max_batch_size
        self.slope_threshold = slope_threshold
        self._requested_profile_sizes = sizes
        self._profile_sizes = sizes
        self._profile_index = 0
        self._epoch_times: dict[int, float] = {}
        self._loader: DataLoader[DataId, DataInst] | None = None
        self._cursor = 0
        self._iteration: int | None = None
        self._iteration_started_at = 0.0
        self._sample_count = 0
        self._profiling = True

    @property
    def profiled_epoch_times(self) -> dict[int, float]:
        """A snapshot of usable trial epoch-time estimates, in seconds."""
        return dict(self._epoch_times)

    def _reset_profile(self, loader: DataLoader[DataId, DataInst]) -> None:
        bound = min(self.max_batch_size, len(loader))
        self._profile_sizes = sorted({min(size, bound) for size in self._requested_profile_sizes})
        self._profile_index = 0
        self.minibatch_size = self._profile_sizes[0]
        self._epoch_times.clear()
        self._profiling = len(self._profile_sizes) > 1
        self._loader = loader
        self.last_trainset_size = len(loader)
        self._iteration = None
        self._sample_count = 0
        self._cursor = 0
        self.shuffled_ids = []
        self.epoch = -1

    def _plateau_size(self) -> int:
        fallback = self._profile_sizes[0]
        if len(self._epoch_times) < 2:
            return fallback
        xs = [math.log(size) for size in self._epoch_times]
        ys = [math.log(delay) for delay in self._epoch_times.values()]
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        variance = sum((x - mean_x) ** 2 for x in xs)
        alpha = -sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / variance
        if not math.isfinite(alpha) or alpha <= 0:
            return fallback
        log_a = mean_y + alpha * mean_x
        # tau is a fraction of the model's peak slope, not an absolute delay
        # in seconds. Work in log space to avoid overflowing A or the plateau.
        peak_size = min(self._epoch_times)
        log_peak_slope = math.log(alpha) + log_a - (alpha + 1) * math.log(peak_size)
        log_tau = math.log(self.slope_threshold) + log_peak_slope
        log_plateau = (math.log(alpha) + log_a - log_tau) / (alpha + 1)
        bound = min(self.max_batch_size, self.last_trainset_size)
        if log_plateau >= math.log(bound):
            return bound
        return max(fallback, min(bound, math.ceil(math.exp(log_plateau))))

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        if len(loader) == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")
        if loader is not self._loader or len(loader) != self.last_trainset_size:
            self._reset_profile(loader)
        now = monotonic()
        if state.i != self._iteration:
            if self._iteration is not None and self._profiling:
                delay = now - self._iteration_started_at
                if math.isfinite(delay) and delay > 0:
                    epoch_time = delay * (len(loader) / self._sample_count)
                    if math.isfinite(epoch_time) and epoch_time > 0:
                        self._epoch_times[self.minibatch_size] = epoch_time
                self._profile_index += 1
                if self._profile_index < len(self._profile_sizes):
                    self.minibatch_size = self._profile_sizes[self._profile_index]
                else:
                    self.minibatch_size = self._plateau_size()
                    self._profiling = False
            self._iteration = state.i
            self._iteration_started_at = now
            self._sample_count = 0

        if self._cursor >= len(self.shuffled_ids):
            self.epoch += 1
            self._update_shuffled(loader)
            # Padding is decided at each tail, since the size can change while
            # traversing this epoch. Keep only the real shuffled inventory.
            del self.shuffled_ids[len(loader) :]
            self.id_freqs = Counter(self.shuffled_ids)
            self._cursor = 0
        end = min(self._cursor + self.minibatch_size, len(self.shuffled_ids))
        batch = self.shuffled_ids[self._cursor : end]
        self._cursor = end
        used = set(batch)
        while len(batch) < self.minibatch_size:
            choices = (data_id for data_id in self.shuffled_ids if data_id not in used)
            selected = min(choices, key=lambda data_id: self.id_freqs[data_id])
            batch.append(selected)
            used.add(selected)
            self.id_freqs[selected] += 1
        self._sample_count += len(batch)
        return batch
