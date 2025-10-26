from gepa.core.data_loader import ListDataLoader, StagedDataLoader


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
