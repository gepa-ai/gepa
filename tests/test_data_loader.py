from typing import Sequence

from gepa.core.adapter import DataInst
from gepa.core.data_loader import ListDataLoader


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


def test_list_data_loader_basic():
    loader = ListDataLoader(["a", "b"])
    assert loader.all_ids() == [0, 1]
    assert loader.fetch([1, 0]) == ["b", "a"]

    loader.add_items(["c"])
    assert loader.all_ids() == [0, 1, 2]
    assert loader.fetch([2]) == ["c"]


def test_staged_data_loader_unlocks_after_batches():
    initial = ["base0", "base1"]
    staged = [
        (1, ["stage1_item"]),
        (3, ["stage2_item"]),
    ]
    loader = StagedDataLoader(initial, staged)

    assert loader.all_ids() == [0, 1]
    assert loader.num_unlocked_stages == 1
    assert loader.batches_served == 0

    loader.fetch([0])
    assert loader.batches_served == 1
    assert loader.num_unlocked_stages == 2
    assert loader.all_ids() == [0, 1, 2]

    loader.fetch([1])
    assert loader.batches_served == 2
    assert loader.num_unlocked_stages == 2

    loader.fetch([2])
    assert loader.batches_served == 3
    assert loader.num_unlocked_stages == 3
    assert loader.all_ids() == [0, 1, 2, 3]


def test_staged_data_loader_manual_unlock():
    loader = StagedDataLoader(["base"], [(5, ["late"])])
    assert loader.all_ids() == [0]
    assert loader.num_unlocked_stages == 1

    unlocked = loader.unlock_next_stage()
    assert unlocked is True
    assert loader.num_unlocked_stages == 2
    assert loader.all_ids() == [0, 1]

    assert loader.unlock_next_stage() is False
