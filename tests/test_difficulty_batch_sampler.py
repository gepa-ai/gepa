import random
from collections import Counter
from types import SimpleNamespace

import pytest

from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import DifficultyAwareBatchSampler, EpochShuffledBatchSampler
from gepa.strategies.proposal_sampling import SameParentSampling


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="minibatch_size"):
        DifficultyAwareBatchSampler(size)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_exploration_fractions_are_rejected(fraction):
    with pytest.raises(ValueError, match="exploration_fraction"):
        DifficultyAwareBatchSampler(2, exploration_fraction=fraction)


def test_hard_examples_gain_frequency_and_easy_examples_retain_coverage():
    loader = ListDataLoader(list(range(10)))
    sampler = DifficultyAwareBatchSampler(2, rng=random.Random(3))
    state = SimpleNamespace(i=0)
    sampler.next_minibatch_ids(loader, state)
    sampler.observe_evaluation(list(range(10)), [0.0] + [1.0] * 9)
    counts = Counter()
    for i in range(100):
        state.i = i
        batch = sampler.next_minibatch_ids(loader, state)
        assert len(batch) == len(set(batch)) == 2
        counts.update(batch)
    assert counts[0] > max(counts[i] for i in range(1, 10))
    assert set(counts) == set(loader.all_ids())
    assert min(counts.values()) >= 10


def test_changed_scores_loader_growth_and_replacement_are_observed():
    loader = ListDataLoader(["first", "second"])
    sampler = DifficultyAwareBatchSampler(2)
    state = SimpleNamespace(i=0)
    sampler.next_minibatch_ids(loader, state)
    sampler.observe_evaluation([0, 1], [0.0, 1.0])
    sampler.observe_evaluation([0], [1.0])
    assert sampler.observed_scores == {0: 1.0, 1: 1.0}
    loader.add_items(["third"])
    state.i += 1
    sampler.next_minibatch_ids(loader, state)
    assert sampler.observed_scores == {0: 1.0, 1: 1.0}
    replacement = ListDataLoader(["unrelated", "examples", "here"])
    sampler.next_minibatch_ids(replacement, state)
    assert sampler.observed_scores == {}
    sampler.observe_evaluation([0], [0.0])
    sampler.next_minibatch_ids(replacement, SimpleNamespace(i=0))
    assert sampler.observed_scores == {}


def test_small_loader_and_extreme_scores_remain_valid():
    loader = ListDataLoader([0, 1])
    sampler = DifficultyAwareBatchSampler(10)
    state = SimpleNamespace(i=0)
    assert set(sampler.next_minibatch_ids(loader, state)) == {0, 1}
    sampler.observe_evaluation([0, 1], [-1e308, 1e308])
    assert set(sampler.next_minibatch_ids(loader, state)) == {0, 1}
    with pytest.raises(ValueError, match="empty"):
        sampler.next_minibatch_ids(ListDataLoader([]), state)


def test_seeded_sampling_is_reproducible_and_advances_within_an_iteration():
    loader = ListDataLoader(list(range(12)))
    first = DifficultyAwareBatchSampler(3, rng=random.Random(11))
    second = DifficultyAwareBatchSampler(3, rng=random.Random(11))
    state = SimpleNamespace(i=0)
    batches = []
    for _ in range(4):
        batch = first.next_minibatch_ids(loader, state)
        assert batch == second.next_minibatch_ids(loader, state)
        batches.append(tuple(batch))
    assert len(set(batches)) == 4


class _Adapter:
    propose_new_texts = None

    def __init__(self):
        self.calls = 0
        self.reflections = 0
        self.parent_examples = Counter()

    def evaluate(self, batch, candidate, capture_traces=False):
        self.calls += len(batch)
        if capture_traces:
            self.parent_examples.update(batch)
        scores = [-1.0 if item == "validation" else 0.0 if item == "hard" else 1.0 for item in batch]
        return EvaluationBatch(
            outputs=list(batch),
            scores=scores,
            trajectories=[{"input": item} for item in batch] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        self.reflections += 1
        return {component: [{"Feedback": "hard example failed"}] for component in components_to_update}


def _run(sampler):
    adapter = _Adapter()
    result = optimize(
        seed_candidate={"system": "seed"},
        trainset=["easy"] * 9 + ["hard"],
        valset=["validation"],
        adapter=adapter,
        batch_sampler=sampler,
        max_metric_calls=160,
        custom_candidate_proposer=lambda candidate, dataset, components: {"system": candidate["system"] + "!"},
        display_progress_bar=False,
    )
    assert result.total_metric_calls == adapter.calls
    return adapter


def test_optimizer_learns_from_parent_evaluations_and_skips_fewer_perfect_batches():
    sampler = DifficultyAwareBatchSampler(2, rng=random.Random(0))
    focused = _run(sampler)
    uniform = _run(EpochShuffledBatchSampler(2, rng=random.Random(0)))
    assert set(sampler.observed_scores) == set(range(10))
    assert sampler.observed_scores[9] == 0.0
    assert sampler.observed_scores[0] == 1.0  # validation has the same positional id but no observation
    assert focused.reflections > uniform.reflections * 2
    assert focused.parent_examples["hard"] > uniform.parent_examples["hard"] * 2
    assert focused.parent_examples["easy"] > 0


@pytest.mark.parametrize("cache", [False, True])
def test_observer_receives_each_deduplicated_parent_evaluation_once(cache):
    class RecordingSampler:
        def __init__(self):
            self.observations = []

        def next_minibatch_ids(self, loader, state):
            return [0, 1]

        def observe_evaluation(self, ids, scores):
            self.observations.append((ids, scores))

    class RecordingAdapter(_Adapter):
        def __init__(self):
            super().__init__()
            self.parents = []

        def evaluate(self, batch, candidate, capture_traces=False):
            evaluation = super().evaluate(batch, candidate, capture_traces)
            if capture_traces:
                self.parents.append(evaluation.scores)
            return evaluation

    sampler, adapter = RecordingSampler(), RecordingAdapter()
    result = optimize(
        seed_candidate={"system": "seed"},
        trainset=["easy", "easy"],
        valset=["validation"],
        adapter=adapter,
        batch_sampler=sampler,
        sampling_strategy=SameParentSampling(n=3),
        max_metric_calls=15,
        cache_evaluation=cache,
        custom_candidate_proposer=lambda *args: pytest.fail("Perfect parents should skip reflection"),
        display_progress_bar=False,
    )
    assert result.total_metric_calls == adapter.calls
    assert len(sampler.observations) == len(adapter.parents) > 0
    assert all(ids == [0, 1] and scores == [1.0, 1.0] for ids, scores in sampler.observations)
