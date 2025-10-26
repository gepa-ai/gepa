"""Data loader protocols and concrete helpers."""

from __future__ import annotations

from typing import Hashable, Protocol, Sequence, TypeVar, runtime_checkable

from gepa.core.adapter import DataInst

DataId = TypeVar("DataId", bound=Hashable)
""" Generic for the identifier for data examples """


@runtime_checkable
class DataLoader(Protocol[DataId, DataInst]):
    """Minimal interface for retrieving validation examples keyed by opaque ids."""

    def all_ids(self) -> Sequence[DataId]:
        """Return the ordered universe of ids currently available. This may change over time."""
        ...

    def fetch(self, ids: Sequence[DataId]) -> list[DataInst]:
        """Materialise the payloads corresponding to `ids`, preserving order."""
        ...

    def __len__(self) -> int:
        """Return current number of items in the loader."""
        ...


class MutableDataLoader(DataLoader[DataId, DataInst], Protocol):
    """A data loader that can be mutated."""

    def add_items(self, items: list[DataInst]) -> None:
        """Add items to the loader."""


class ListDataLoader(MutableDataLoader[int, DataInst]):
    """In-memory reference implementation backed by a list."""

    def __init__(self, items: Sequence[DataInst]):
        self.items = list(items)

    def all_ids(self) -> Sequence[int]:
        return list(range(len(self.items)))

    def fetch(self, ids: Sequence[int]) -> list[DataInst]:
        return [self.items[data_id] for data_id in ids]

    def __len__(self) -> int:
        return len(self.items)

    def add_items(self, items: Sequence[DataInst]) -> None:
        self.items.extend(items)


class StagedDataLoader(ListDataLoader):
    """ListDataLoader that gradually unlocks staged examples after serving a number of batches."""

    def __init__(
        self,
        initial_items: Sequence[DataInst],
        staged_items: Sequence[tuple[int, Sequence[DataInst]]],
    ):
        """
        Args:
            initial_items: Items available from the beginning.
            staged_items: Sequence of (batches_served_threshold, items). Each stage becomes available after the loader
                has served at least the given number of batches via `fetch`.
        """
        super().__init__(initial_items)
        self._stages = sorted(
            [(max(0, threshold), list(items)) for threshold, items in staged_items],
            key=lambda pair: pair[0],
        )
        self._next_stage_idx = 0
        self._batches_served = 0
        self.num_unlocked_stages = 1  # the initial batch is always unlocked
        self._unlock_if_due()

    @property
    def batches_served(self) -> int:
        return self._batches_served

    def fetch(self, ids: Sequence[int]) -> list[DataInst]:
        batch = super().fetch(ids)
        self._batches_served += 1
        self._unlock_if_due()
        return batch

    def unlock_next_stage(self) -> bool:
        """Manually unlock the next stage, returning True if one existed."""
        if self._next_stage_idx >= len(self._stages):
            return False
        _, items = self._stages[self._next_stage_idx]
        self.add_items(items)
        self._next_stage_idx += 1
        self.num_unlocked_stages += 1
        return True

    def _unlock_if_due(self) -> None:
        while self._next_stage_idx < len(self._stages):
            threshold, _ = self._stages[self._next_stage_idx]
            if self._batches_served < threshold:
                break
            self.unlock_next_stage()


def ensure_loader(data_or_loader: Sequence[DataInst] | DataLoader[DataId, DataInst]) -> DataLoader[DataId, DataInst]:
    if isinstance(data_or_loader, DataLoader):
        return data_or_loader
    if isinstance(data_or_loader, Sequence):
        return ListDataLoader(data_or_loader)
    raise TypeError(f"Unable to cast to a DataLoader type: {type(data_or_loader)}")
