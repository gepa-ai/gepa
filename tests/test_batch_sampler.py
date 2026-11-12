import random
from types import SimpleNamespace

import pytest

import gepa
from gepa.core.data_loader import ListDataLoader
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


class ChangingIdLoader:
    def __init__(self, ids):
        self.ids = ids

    def all_ids(self):
        return self.ids

    def __len__(self):
        return len(self.ids)

    def fetch(self, ids):
        items = {i: f"example {i}" for i in self.ids}
        return [items[i] for i in ids]


@pytest.mark.parametrize("next_iteration", [0, 1])
@pytest.mark.parametrize("new_ids", [[10, 11, 12, 13], [0, 1, 2, 10]])
def test_epoch_sampler_refreshes_replaced_ids_without_size_change(next_iteration, new_ids):
    loader = ChangingIdLoader([0, 1, 2, 3])
    sampler = EpochShuffledBatchSampler(minibatch_size=2)
    state = SimpleNamespace(i=0)
    sampler.next_minibatch_ids(loader, state)

    loader.ids = new_ids
    state.i = next_iteration
    batch = sampler.next_minibatch_ids(loader, state)

    assert len(batch) == 2
    assert set(batch).issubset(loader.all_ids())
    assert len(loader.fetch(batch)) == 2


def test_epoch_sampler_keeps_shuffle_when_only_loader_order_changes():
    loader = ChangingIdLoader([0, 1, 2, 3])
    control_loader = ChangingIdLoader([0, 1, 2, 3])
    sampler = EpochShuffledBatchSampler(minibatch_size=2)
    control = EpochShuffledBatchSampler(minibatch_size=2)
    for i in range(5):
        state = SimpleNamespace(i=i)
        assert sampler.next_minibatch_ids(loader, state) == control.next_minibatch_ids(control_loader, state)
        loader.ids = list(reversed(loader.ids))


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


class _ConstantAdapter:
    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        return EvaluationBatch(
            outputs=[candidate["prompt"]] * len(batch),
            scores=[0.25] * len(batch),
            trajectories=[{"score": 0.25}] * len(batch) if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {name: [{"score": 0.25}] for name in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        return {name: candidate[name] for name in components_to_update}


def _record_minibatches(run_dir, stop_at):
    seen = []

    class Recorder:
        def on_minibatch_sampled(self, event):
            seen.append(list(event["minibatch_ids"]))

    gepa.optimize(
        seed_candidate={"prompt": "seed"},
        trainset=list(range(8)),
        valset=[99],
        adapter=_ConstantAdapter(),
        seed=9,
        reflection_minibatch_size=2,
        skip_perfect_score=False,
        stop_callbacks=[lambda state: state.i + 1 >= stop_at],
        callbacks=[Recorder()],
        display_progress_bar=False,
        run_dir=str(run_dir),
    )
    return seen


@pytest.mark.parametrize("split_at", [1, 2, 5, 6, 7])
def test_resuming_mid_epoch_keeps_epoch_coverage(tmp_path, split_at):
    continuous = _record_minibatches(tmp_path / "continuous", 12)
    resumed = _record_minibatches(tmp_path / "split", split_at) + _record_minibatches(tmp_path / "split", 12)

    # The interrupted epoch continues exactly where it left off, and every
    # epoch (including the one split by the resume) covers each example.
    epoch_end = -(-split_at // 4) * 4
    assert len(resumed) == len(continuous) == 12
    assert resumed[:epoch_end] == continuous[:epoch_end]
    for epoch_start in (0, 4, 8):
        epoch_ids = {i for batch in resumed[epoch_start : epoch_start + 4] for i in batch}
        assert epoch_ids == set(range(8))


def test_sampler_without_saved_state_uses_one_shuffle_for_rest_of_epoch():
    # Checkpoints written before the sampler state was saved resume with a
    # fresh sampler partway through an epoch. It should draw the remaining
    # batches of that epoch from a single shuffle instead of reshuffling on
    # every iteration.
    loader = ListDataLoader(list("abcdefgh"))
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))

    sampler.next_minibatch_ids(loader, SimpleNamespace(i=5))
    shuffle = list(sampler.shuffled_ids)
    for i in (6, 7):
        assert sampler.next_minibatch_ids(loader, SimpleNamespace(i=i)) == shuffle[2 * (i % 4) : 2 * (i % 4) + 2]
    sampler.next_minibatch_ids(loader, SimpleNamespace(i=8))
    assert sampler.epoch == 2


def test_sampler_ignores_saved_state_from_different_minibatch_size():
    loader = ListDataLoader(list("abcdef"))
    saved = EpochShuffledBatchSampler(minibatch_size=4, rng=random.Random(0))
    saved.next_minibatch_ids(loader, SimpleNamespace(i=0))

    sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(0))
    sampler.set_sampler_state(saved.get_sampler_state())

    assert sampler.shuffled_ids == []
    assert len(sampler.next_minibatch_ids(loader, SimpleNamespace(i=1))) == 3
