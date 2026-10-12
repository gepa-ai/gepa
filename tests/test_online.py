from collections import Counter
from functools import wraps

import pytest

import gepa.online as online
from gepa import EvaluationBatch, optimize_online
from gepa.core.data_loader import ListDataLoader


class RuleAdapter:
    """A deterministic classification task exercising the real optimizer."""

    def __init__(self):
        self.events = []
        self.phase = "test"
        self.fail_on_target = None

    def evaluate(self, batch, candidate, capture_traces=False):
        self.events.append((self.phase, [item["id"] for item in batch], dict(candidate)))
        if self.phase == "learn" and capture_traces and any(item["target"] == self.fail_on_target for item in batch):
            self.fail_on_target = None
            raise RuntimeError("interrupted update")
        predictions = [int(candidate["rule"])] * len(batch)
        return EvaluationBatch(
            outputs=predictions,
            scores=[float(prediction == item["target"]) for prediction, item in zip(predictions, batch, strict=True)],
            trajectories=batch if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {"rule": eval_batch.trajectories}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update, *, metadata=None):
        target = Counter(item["target"] for item in reflective_dataset["rule"]).most_common(1)[0][0]
        return {"rule": str(target)}


@pytest.fixture
def adapter(monkeypatch):
    adapter = RuleAdapter()
    offline_optimize = online.optimize

    @wraps(offline_optimize)
    def traced_optimize(*args, **kwargs):
        adapter.phase = "learn"
        try:
            return offline_optimize(*args, **kwargs)
        finally:
            adapter.phase = "test"

    monkeypatch.setattr(online, "optimize", traced_optimize)
    return adapter


def examples(targets):
    return [{"id": i, "target": target} for i, target in enumerate(targets)]


def run(data, adapter, **kwargs):
    return optimize_online(
        {"rule": "0"},
        data,
        adapter,
        window_size=2,
        max_metric_calls_per_window=12,
        optimize_kwargs={"reflection_minibatch_size": 2},
        **kwargs,
    )


def test_evaluates_before_learning_and_scores_deployed_candidates(adapter):
    result = run(examples([1, 1, 1, 1, 0]), adapter)
    assert [window.candidate for window in result.windows] == [{"rule": "0"}, {"rule": "1"}, {"rule": "1"}]
    assert [window.scores for window in result.windows] == [[0, 0], [1, 1], [0]]
    assert result.online_score == 2 / 5  # Weight examples, not equally sized and partial windows.
    assert result.num_examples == 5
    # The last update keeps two recent examples (one positive, one negative).
    assert result.candidate == {"rule": "1"}
    assert result.total_metric_calls > result.num_examples

    seen = set()
    for phase, ids, _ in adapter.events:
        if phase == "test":
            assert seen.isdisjoint(ids)
            seen.update(ids)
        else:
            assert set(ids) <= seen
    assert seen == set(range(5))


def test_future_labels_cannot_affect_past_candidates_or_scores(adapter):
    first = run(examples([1, 1, 1, 1, 0, 0]), adapter)
    changed_future = run(examples([1, 1, 1, 1, 1, 1]), adapter)
    assert first.windows[:2] == changed_future.windows[:2]
    assert first.windows[2].candidate == changed_future.windows[2].candidate
    assert first.windows[2].scores != changed_future.windows[2].scores


def test_training_history_is_bounded(adapter):
    run(examples([1] * 10), adapter, history_size=4)
    last_test_end = 0
    for phase, ids, _ in adapter.events:
        if phase == "test":
            last_test_end = max(ids) + 1
        else:
            assert set(ids) <= set(range(max(0, last_test_end - 4), last_test_end))


def test_resume_after_update_failure_reuses_pre_update_evaluation(adapter, tmp_path):
    data = examples([1, 1, 0, 0])
    adapter.fail_on_target = 0
    with pytest.raises(RuntimeError, match="interrupted update"):
        run(data, adapter, run_dir=str(tmp_path))
    assert [ids for phase, ids, _ in adapter.events if phase == "test"] == [[0, 1], [2, 3]]
    adapter.events.clear()
    result = run(data, adapter, run_dir=str(tmp_path))
    assert not any(phase == "test" for phase, _, _ in adapter.events)
    assert result.online_score == 0
    assert result.candidate == {"rule": "0"}
    assert len(result.windows) == 2


def test_resume_completed_windows_and_appended_data(adapter, tmp_path):
    loader = ListDataLoader(examples([1, 1, 1, 1]))
    first = run(loader, adapter, run_dir=str(tmp_path), max_windows=1)
    assert len(first.windows) == 1
    second = run(loader, adapter, run_dir=str(tmp_path))
    assert len(second.windows) == 2
    adapter.events.clear()
    unchanged = run(loader, adapter, run_dir=str(tmp_path))
    assert unchanged == second
    assert adapter.events == []
    loader.add_items([{"id": 4, "target": 0}])
    third = run(loader, adapter, run_dir=str(tmp_path))
    assert third.windows[:2] == second.windows
    assert third.windows[-1].example_ids == [4]
    assert third.online_score == 2 / 5


def test_resume_rejects_changed_id_prefix(adapter, tmp_path):
    class KeyedLoader:
        def __init__(self, ids):
            self.ids = ids

        def all_ids(self):
            return self.ids

        def fetch(self, ids):
            return [{"id": i, "target": 1} for i in ids]

        def __len__(self):
            return len(self.ids)

    run(KeyedLoader(["a", "b", "c"]), adapter, run_dir=str(tmp_path), max_windows=1)
    adapter.events.clear()
    with pytest.raises(ValueError, match="processed prefix"):
        run(KeyedLoader(["b", "a", "c"]), adapter, run_dir=str(tmp_path))
    assert adapter.events == []


def test_failed_evaluation_is_not_committed(adapter, tmp_path, monkeypatch):
    evaluate = adapter.evaluate

    def fail(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(adapter, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run(examples([1, 1]), adapter, run_dir=str(tmp_path))
    assert not (tmp_path / "online_state.bin").exists()
    monkeypatch.setattr(adapter, "evaluate", evaluate)
    assert run(examples([1, 1]), adapter, run_dir=str(tmp_path)).num_examples == 2


def test_empty_and_zero_window_limit_do_not_call_adapter(adapter):
    empty = run([], adapter)
    assert empty.online_score is None
    assert empty.total_metric_calls == 0
    assert empty.candidate == {"rule": "0"}
    assert run(examples([1, 1]), adapter, max_windows=0) == empty
    assert adapter.events == []


@pytest.mark.parametrize("option", ["trainset", "valset", "val_evaluation_policy", "run_dir", "max_metric_calls"])
def test_cannot_override_online_data_or_lifecycle(adapter, option):
    with pytest.raises(ValueError, match="cannot override"):
        optimize_online(
            {"rule": "0"}, examples([1, 1]), adapter, max_metric_calls_per_window=12, optimize_kwargs={option: None}
        )
    assert adapter.events == []


@pytest.mark.parametrize(
    "change", [{"window_size": 0}, {"history_size": 1}, {"max_windows": -1}, {"max_metric_calls_per_window": 0}]
)
def test_invalid_options_fail_before_evaluation(adapter, change):
    options = {"window_size": 2, "max_metric_calls_per_window": 12, **change}
    with pytest.raises(ValueError):
        optimize_online({"rule": "0"}, examples([1, 1]), adapter, **options)
    assert adapter.events == []


def test_resume_rejects_changed_window_configuration(adapter, tmp_path):
    run(examples([1, 1]), adapter, run_dir=str(tmp_path))
    with pytest.raises(ValueError, match="must match"):
        run(examples([1, 1]), adapter, run_dir=str(tmp_path), history_size=4)


def test_interrupted_checkpoint_commit_preserves_test_result(adapter, tmp_path, monkeypatch):
    replace = online.os.replace
    writes = 0

    def fail_second_online_write(source, target):
        nonlocal writes
        if str(target).endswith("online_state.bin"):
            writes += 1
            if writes == 2:
                raise OSError("checkpoint commit failed")
        return replace(source, target)

    monkeypatch.setattr(online.os, "replace", fail_second_online_write)
    with pytest.raises(OSError, match="checkpoint commit failed"):
        run(examples([1, 1]), adapter, run_dir=str(tmp_path))
    adapter.events.clear()
    result = run(examples([1, 1]), adapter, run_dir=str(tmp_path))
    assert result.num_examples == 2
    assert result.online_score == 0
    assert result.candidate == {"rule": "1"}
    assert not any(phase == "test" for phase, _, _ in adapter.events)


def test_metric_call_accounting_uses_adapter_report(adapter, monkeypatch):
    evaluate = adapter.evaluate

    def with_cost(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if adapter.phase == "test":
            result.num_metric_calls = 7
        return result

    monkeypatch.setattr(adapter, "evaluate", with_cost)
    result = run(examples([1, 1]), adapter)
    assert result.windows[0].evaluation_metric_calls == 7
    assert result.total_metric_calls == 7 + result.windows[0].optimization_metric_calls


def test_incomplete_scores_do_not_start_training(adapter, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "evaluate", lambda *args, **kwargs: EvaluationBatch(outputs=[], scores=[]))
    with pytest.raises(ValueError, match="one score per example"):
        run(examples([1, 1]), adapter, run_dir=str(tmp_path))
    assert not (tmp_path / "online_state.bin").exists()
