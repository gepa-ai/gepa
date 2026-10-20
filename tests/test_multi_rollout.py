# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Regression for #139: repeated scores and original failed-rollout reflection."""

from dataclasses import dataclass

import pytest

from gepa.adapters.multi_rollout import MultiRolloutAdapter
from gepa.core.adapter import EvaluationBatch, invoke_batch_evaluate


@dataclass
class Trace:
    example: int
    repetition: int


class Adapter:
    def __init__(self):
        self.evaluations = []
        self.reflections = []
        self.state = {"counter": 3}
        self.propose_new_texts = lambda *args, **kwargs: {"p": "improved"}

    def evaluate(self, batch, candidate, capture_traces=False):
        repetition = len(self.evaluations) % 3
        self.evaluations.append((batch, candidate, capture_traces))
        scores = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.8]][repetition]
        return EvaluationBatch(
            outputs=[object() for _ in batch],
            scores=scores[: len(batch)],
            trajectories=[Trace(example, repetition) for example in batch] if capture_traces else None,
            objective_scores=[{"quality": score, "cost": 0.2 * repetition} for score in scores[: len(batch)]],
            num_metric_calls=2 * len(batch),
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        self.reflections.append(eval_batch)
        return {
            component: [{"trace": trace.example, "repetition": trace.repetition} for trace in eval_batch.trajectories]
            for component in components_to_update
        }

    def get_adapter_state(self):
        return dict(self.state)

    def set_adapter_state(self, state):
        self.state = state


@pytest.mark.parametrize("capture_traces", [False, True])
def test_means_outputs_objectives_and_actual_calls(capture_traces):
    original = Adapter()
    adapter = MultiRolloutAdapter(original)
    result = adapter.evaluate([11, 12], {"p": "seed"}, capture_traces=capture_traces)
    assert result.scores == pytest.approx([0.5, 0.6])
    assert result.objective_scores[0] == pytest.approx({"quality": 0.5, "cost": 0.2})
    assert result.objective_scores[1] == pytest.approx({"quality": 0.6, "cost": 0.2})
    assert result.num_metric_calls == 12
    assert all(len(group) == 3 for group in result.outputs)
    assert len(original.evaluations) == 3
    assert all(call[2] is capture_traces for call in original.evaluations)
    if capture_traces:
        for i, group in enumerate(result.trajectories):
            assert [rollout.trajectory.repetition for rollout in group] == [0, 1, 2]
            assert [rollout.trajectory.example for rollout in group] == [11 + i] * 3
            assert all(rollout.output is result.outputs[i][j] for j, rollout in enumerate(group))
    else:
        assert result.trajectories is None


def test_reflection_filters_original_scores_instead_of_group_means():
    original = Adapter()
    adapter = MultiRolloutAdapter(original, reflection_score_threshold=1.0)
    batch = adapter.evaluate([11, 12], {"p": "seed"}, capture_traces=True)
    records = adapter.make_reflective_dataset({"p": "seed"}, batch, ["p", "q"])
    assert [evaluation.scores for evaluation in original.reflections] == [[0.0], [0.0], [0.5, 0.8]]
    assert records["p"] == [
        {"trace": 11, "repetition": 0},
        {"trace": 12, "repetition": 1},
        {"trace": 11, "repetition": 2},
        {"trace": 12, "repetition": 2},
    ]
    assert records["q"] == records["p"]
    # Delegate opaque traces and outputs as the same objects, preserving adapter-specific contracts.
    first = original.reflections[0]
    assert first.trajectories[0] is batch.trajectories[0][0].trajectory
    assert first.outputs[0] is batch.outputs[0][0]
    assert first.objective_scores == [{"quality": 0.0, "cost": 0.0}]


def test_default_reflection_keeps_every_rollout():
    original = Adapter()
    adapter = MultiRolloutAdapter(original)
    batch = adapter.evaluate([11, 12], {"p": "seed"}, capture_traces=True)
    assert len(adapter.make_reflective_dataset({"p": "seed"}, batch, ["p"])["p"]) == 6
    assert all(len(evaluation.scores) == 2 for evaluation in original.reflections)


@pytest.mark.parametrize("native_batch", [False, True])
def test_optional_batch_evaluator_is_used_for_each_repetition(native_batch):
    original = Adapter()
    batches = []
    if native_batch:

        def batch_evaluate(items, *, capture_traces):
            batches.append(capture_traces)
            return [original.evaluate(batch, candidate, capture_traces) for candidate, batch in items]

        original.batch_evaluate = batch_evaluate
    adapter = MultiRolloutAdapter(original)
    results = invoke_batch_evaluate(adapter, [({"p": "a"}, [1, 2]), ({"p": "b"}, [3])], capture_traces=False)
    assert [result.num_metric_calls for result in results] == [12, 6]
    assert len(original.evaluations) == 6
    assert batches == ([False] * 3 if native_batch else [])
    assert adapter.batch_evaluate([], capture_traces=True) == []


def test_optional_proposer_and_checkpoint_capabilities_delegate():
    original = Adapter()
    adapter = MultiRolloutAdapter(original)
    assert adapter.propose_new_texts is original.propose_new_texts
    assert adapter.get_adapter_state() == original.get_adapter_state()
    adapter.set_adapter_state({"counter": 8})
    assert original.state == {"counter": 8}
    del original.propose_new_texts
    assert not hasattr(adapter, "propose_new_texts")


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_rollout_counts_are_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        MultiRolloutAdapter(Adapter(), value)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("scores", [float("nan"), 0.0], "finite"),
        ("outputs", [], "one output"),
        ("trajectories", None, "one trajectory"),
        ("objective_scores", [], "one objective"),
        ("num_metric_calls", -1, "nonnegative integer"),
    ],
)
def test_invalid_evaluation_batches_fail_clearly(field, value, message):
    original = Adapter()
    evaluate = original.evaluate

    def invalid(*args, **kwargs):
        batch = evaluate(*args, **kwargs)
        setattr(batch, field, value)
        return batch

    original.evaluate = invalid
    with pytest.raises(ValueError, match=message):
        MultiRolloutAdapter(original).evaluate([1, 2], {"p": "seed"}, capture_traces=True)


def test_missing_or_inconsistent_objectives_do_not_silently_disappear():
    original = Adapter()
    evaluate = original.evaluate

    def inconsistent(*args, **kwargs):
        batch = evaluate(*args, **kwargs)
        if len(original.evaluations) == 2:
            batch.objective_scores[0] = {"other": 1.0}
        return batch

    original.evaluate = inconsistent
    with pytest.raises(ValueError, match="objective names must agree"):
        MultiRolloutAdapter(original).evaluate([1, 2], {"p": "seed"})


def test_wrapper_can_be_copied_without_recursive_optional_method_lookup():
    import copy

    original = MultiRolloutAdapter(Adapter())
    copied = copy.deepcopy(original)
    copied.set_adapter_state({"counter": 9})
    assert copied.get_adapter_state() == {"counter": 9}
    assert original.get_adapter_state() == {"counter": 3}
