import random
from types import SimpleNamespace

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


def test_epoch_sampler_refreshes_when_loader_expands():
    loader = ListDataLoader(["a", "b", "c", "d"])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    first_batch = sampler.next_minibatch_ids(loader, state)
    assert len(first_batch) == 2
    assert len(sampler.shuffled_ids) == 4
    assert sampler.last_trainset_size == 4

    state.i += 1
    loader.add_items(["e", "f"])

    second_batch = sampler.next_minibatch_ids(loader, state)
    assert len(second_batch) == 2
    assert sampler.last_trainset_size == 6
    assert len(sampler.shuffled_ids) == 6
    assert {4, 5}.issubset(set(sampler.shuffled_ids))


def test_epoch_sampler_errors_when_loader_empty():
    loader = ListDataLoader([])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    with pytest.raises(ValueError):
        sampler.next_minibatch_ids(loader, state)


def test_repeated_calls_within_iteration_return_distinct_minibatches():
    # Multi-proposal sampling strategies call the sampler once per task within
    # a single iteration; each call must yield a different minibatch.
    loader = ListDataLoader(["a", "b", "c", "d", "e", "f"])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    batches = [tuple(sampler.next_minibatch_ids(loader, state)) for _ in range(3)]
    assert len(set(batches)) == 3
    assert {i for batch in batches for i in batch} == {0, 1, 2, 3, 4, 5}

    # A fourth call exceeds the number of chunks and wraps around.
    assert tuple(sampler.next_minibatch_ids(loader, state)) == batches[0]


def test_extra_calls_do_not_perturb_later_iterations():
    # The first call of each iteration must be byte-identical whether or not
    # earlier iterations made extra (multi-proposal) calls.
    items = list("abcdefgh")
    single = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(0))
    multi = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(0))

    for i in range(10):
        loader_single, loader_multi = ListDataLoader(items), ListDataLoader(items)
        state = SimpleNamespace(i=i)
        expected = single.next_minibatch_ids(loader_single, state)
        assert multi.next_minibatch_ids(loader_multi, state) == expected
        for _ in range(4):
            multi.next_minibatch_ids(loader_multi, state)
