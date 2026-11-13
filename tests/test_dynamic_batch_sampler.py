"""Adaptive sampling uses measured throughput while preserving training coverage."""

import math
import random
from types import SimpleNamespace

import pytest

import gepa
import gepa.strategies.batch_sampler as batch_sampler
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.proposer.reflective_mutation.combee import ComBEEReflectionLM
from gepa.strategies.batch_sampler import DynamicBatchSampler, EpochShuffledBatchSampler


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.mark.parametrize("maximum,selected,exponent", [(6, 6, 0), (16, 8, 0), (16, 16, 0.5)])
def test_profile_selects_the_plateau_without_skipping_epoch_ids(maximum, selected, exponent, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(batch_sampler, "monotonic", clock)
    loader = ListDataLoader(list(range(17)))
    sampler = DynamicBatchSampler(max_batch_size=maximum, profile_batch_sizes=[1, 2, 4], rng=random.Random(0))
    batches = []
    for i in range(8):
        batches.append(sampler.next_minibatch_ids(loader, SimpleNamespace(i=i)))
        clock.advance(10 * len(batches[-1]) ** exponent)
    assert [len(batch) for batch in batches[:4]] == [1, 2, 4, selected]
    assert sampler.profiled_epoch_times == pytest.approx({size: 170 * size ** (exponent - 1) for size in (1, 2, 4)})
    # Size changes continue the current shuffle rather than dropping its tail.
    first_epoch_calls = 3 + (17 - 7 + selected - 1) // selected
    assert set(item for batch in batches[:first_epoch_calls] for item in batch) == set(range(17))
    assert all(len(set(batch)) == len(batch) for batch in batches)


@pytest.mark.parametrize(
    "delay",
    [
        lambda size: 10 * size,
        lambda size: 10 * size**2,
        lambda size: 0,
        lambda size: float("nan"),
        lambda size: float("inf"),
    ],
)
def test_flat_increasing_and_unusable_timings_keep_the_smallest_batch(delay, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(batch_sampler, "monotonic", clock)
    loader = ListDataLoader(list(range(20)))
    sampler = DynamicBatchSampler(max_batch_size=16, profile_batch_sizes=[1, 2, 4])
    for i, size in enumerate((1, 2, 4)):
        assert len(sampler.next_minibatch_ids(loader, SimpleNamespace(i=i))) == size
        clock.advance(delay(size))
    assert len(sampler.next_minibatch_ids(loader, SimpleNamespace(i=3))) == 1


def test_parallel_sampling_profiles_only_after_the_iteration_finishes(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(batch_sampler, "monotonic", clock)
    loader = ListDataLoader(list(range(24)))
    sampler = DynamicBatchSampler(max_batch_size=16, profile_batch_sizes=[1, 2, 4], rng=random.Random(7))
    for i, size in enumerate((1, 2, 4)):
        state = SimpleNamespace(i=i)
        first = sampler.next_minibatch_ids(loader, state)
        second = sampler.next_minibatch_ids(loader, state)
        assert len(first) == len(second) == size
        assert set(first).isdisjoint(second)
        assert size not in sampler.profiled_epoch_times
        clock.advance(10)
    assert len(sampler.next_minibatch_ids(loader, SimpleNamespace(i=3))) == 8
    assert sampler.profiled_epoch_times == {1: 120.0, 2: 60.0, 4: 30.0}


def test_loader_changes_restart_profiling_and_enforce_dataset_bounds(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(batch_sampler, "monotonic", clock)
    loader = ListDataLoader(list(range(4)))
    sampler = DynamicBatchSampler(max_batch_size=16, profile_batch_sizes=[1, 2, 4])
    sampler.next_minibatch_ids(loader, SimpleNamespace(i=0))
    clock.advance(1)
    sampler.next_minibatch_ids(loader, SimpleNamespace(i=1))
    assert sampler.profiled_epoch_times
    loader.add_items([4, 5])
    assert len(sampler.next_minibatch_ids(loader, SimpleNamespace(i=2))) == 1
    assert sampler.profiled_epoch_times == {}
    seen = set()
    for i in range(3, 9):
        seen.update(sampler.next_minibatch_ids(loader, SimpleNamespace(i=i)))
        clock.advance(1)
    assert {4, 5} <= seen
    # A different, smaller loader must replace the previous shuffle, even if
    # the optimizer is still in the same iteration.
    assert sampler.next_minibatch_ids(ListDataLoader(["only"]), SimpleNamespace(i=8)) == [0]
    assert sampler.minibatch_size == 1
    with pytest.raises(ValueError, match="empty"):
        sampler.next_minibatch_ids(ListDataLoader([]), SimpleNamespace(i=9))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch_size": 0},
        {"min_batch_size": 3, "max_batch_size": 2},
        {"max_batch_size": 2.5},
        {"max_batch_size": True},
        {"max_batch_size": 8, "profile_batch_sizes": []},
        {"max_batch_size": 8, "profile_batch_sizes": [0, 2]},
        {"max_batch_size": 8, "profile_batch_sizes": [1, 9]},
        {"max_batch_size": 8, "slope_threshold": 0},
        {"max_batch_size": 8, "slope_threshold": float("nan")},
    ],
)
def test_invalid_controller_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        DynamicBatchSampler(**kwargs)


@pytest.mark.parametrize("mode", ["adaptive", "fixed", "combee"])
def test_optimizer_uses_the_sampler_without_extra_profiling_evaluations(mode, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(batch_sampler, "monotonic", clock)

    class Adapter:
        def __init__(self):
            self.evaluation_sizes = []
            self.proposals = 0

        def evaluate(self, batch, candidate, capture_traces=False):
            self.evaluation_sizes.append(len(batch))
            clock.advance(5)
            return EvaluationBatch(
                outputs=[""] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=list(batch) if capture_traces else None,
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {
                component: [
                    {"Inputs": str(item), "Feedback": "Try another instruction"} for item in eval_batch.trajectories
                ]
                for component in components_to_update
            }

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            self.proposals += 1
            return dict.fromkeys(components_to_update, f"instruction {self.proposals}")

    class Callback:
        def __init__(self):
            self.sizes = []

        def on_minibatch_sampled(self, event):
            self.sizes.append(len(event["minibatch_ids"]))

    sampler = (
        DynamicBatchSampler(max_batch_size=8, profile_batch_sizes=[1, 2, 4])
        if mode != "fixed"
        else EpochShuffledBatchSampler(minibatch_size=3)
    )
    adapter, callback = Adapter(), Callback()
    reflection_kwargs = {}
    reflection_calls = []
    if mode == "combee":
        adapter.propose_new_texts = None

        def scripted_lm(prompt):
            reflection_calls.append(prompt)
            clock.advance(1)
            return f"```\nupdated instruction {len(reflection_calls)}\n```"

        reflection_kwargs["reflection_strategy"] = ComBEEReflectionLM(scripted_lm, rng=random.Random(0))
    result = gepa.optimize(
        seed_candidate={"instruction": "seed"},
        trainset=list(range(20)),
        valset=[100],
        adapter=adapter,
        batch_sampler=sampler,
        max_metric_calls=60,
        callbacks=[callback],
        display_progress_bar=False,
        **reflection_kwargs,
    )
    assert callback.sizes[:4] == ([1, 2, 4, 8] if mode != "fixed" else [3, 3, 3, 3])
    if mode == "combee":
        assert len(reflection_calls) == sum(1 if size < 4 else math.isqrt(size) + 1 for size in callback.sizes)
    else:
        assert adapter.proposals == len(callback.sizes)
    assert adapter.evaluation_sizes == [1] + [size for n in callback.sizes for size in (n, n)]
    assert result.total_metric_calls == sum(adapter.evaluation_sizes)
