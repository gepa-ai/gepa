import random
from types import SimpleNamespace

import pytest

import gepa
from gepa.core.data_loader import ListDataLoader
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


@pytest.mark.parametrize("minibatch_size", [0, -1, -3])
def test_epoch_sampler_rejects_nonpositive_minibatch_size(minibatch_size):
    with pytest.raises(ValueError, match="minibatch_size must be positive"):
        EpochShuffledBatchSampler(minibatch_size=minibatch_size)


@pytest.mark.parametrize("minibatch_size", [0, -1, -3])
@pytest.mark.parametrize("entrypoint", ["optimize", "optimize_anything"])
def test_optimize_rejects_nonpositive_minibatch_size_before_evaluation(minibatch_size, entrypoint):
    def task_lm(_messages):
        pytest.fail("Invalid minibatch size must be rejected before evaluating the seed")

    with pytest.raises(ValueError, match="minibatch_size must be positive"):
        if entrypoint == "optimize_anything":
            optimize_anything(
                seed_candidate="Answer the question.",
                evaluator=lambda _candidate, _example: task_lm([]),
                dataset=["question"],
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=1, display_progress_bar=False),
                    reflection=ReflectionConfig(
                        reflection_lm=lambda _prompt: "unused",
                        reflection_minibatch_size=minibatch_size,
                    ),
                ),
            )
            return
        gepa.optimize(
            seed_candidate={"system_prompt": "Answer the question."},
            trainset=[{"input": "question", "answer": "answer"}],
            task_lm=task_lm,
            reflection_lm=lambda _prompt: "unused",
            reflection_minibatch_size=minibatch_size,
            max_metric_calls=1,
            display_progress_bar=False,
        )


@pytest.mark.parametrize("minibatch_size", [1, 2, 7])
def test_epoch_sampler_accepts_positive_sizes_and_pads_large_batches(minibatch_size):
    loader = ListDataLoader(["a", "b", "c"])
    sampler = EpochShuffledBatchSampler(minibatch_size=minibatch_size)

    for i in range(5):
        batch = sampler.next_minibatch_ids(loader, SimpleNamespace(i=i))
        assert len(batch) == minibatch_size
        assert set(batch).issubset(loader.all_ids())


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
