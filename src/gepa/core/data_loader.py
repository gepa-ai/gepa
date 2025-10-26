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


def ensure_loader(data_or_loader: Sequence[DataInst] | DataLoader[DataId, DataInst]) -> DataLoader[DataId, DataInst]:
    if isinstance(data_or_loader, DataLoader):
        return data_or_loader
    if isinstance(data_or_loader, Sequence):
        return ListDataLoader(data_or_loader)
    raise TypeError(f"Unable to cast to a DataLoader type: {type(data_or_loader)}")
