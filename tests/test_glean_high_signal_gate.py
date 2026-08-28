import threading

from glean_gepa.al_adapter import GleanAdapterBase
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evolutionary_proposer import _select_screened_children


def _batch(scores: list[float]) -> GleanEvaluationBatch:
    trajectories = [
        {
            "data": {
                "eval_set_name": "set",
                "eval_set_version": "v1",
                "deployment_ids": ["prod"],
                "status": "active",
            },
            "output": {"entry_id": f"entry-{index}"},
            "score": score,
            "objective_scores": {},
        }
        for index, score in enumerate(scores)
    ]
    return GleanEvaluationBatch(
        outputs=[],
        scores=scores,
        trajectories=trajectories,
        objective_scores=[{} for _ in scores],
        summary={"objective": sum(scores) / len(scores) if scores else 0.0},
    )


def test_high_signal_batch_contains_only_parent_failures():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)

    focused = adapter.high_signal_batch(_batch([0.0, 0.5, 1.0]))

    assert len(focused) == 1
    assert focused[0]["eval_entry_ids"] == ["entry-0", "entry-1"]


def test_high_signal_fix_rate_requires_error_free_entries():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    parent = _batch([0.0, 0.0, 0.0, 1.0])
    # A focused child eval contains only the three parent failures.
    child = _batch([1.0, 1.0, 0.0])

    assert adapter.high_signal_fix_rate(parent, child) == 2 / 3


def test_high_signal_fix_rate_is_zero_without_parent_failures():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)

    assert adapter.high_signal_fix_rate(_batch([1.0]), _batch([1.0])) == 0.0


def test_high_signal_screen_keeps_all_children_over_half():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    parent = _batch([0.0, 0.0, 0.0, 0.0])
    children = [object(), object(), object()]
    evaluations = [
        _batch([1.0, 1.0, 1.0, 0.0]),
        _batch([1.0, 1.0, 0.0, 0.0]),
        _batch([1.0, 1.0, 1.0, 1.0]),
    ]

    selected = _select_screened_children(
        adapter,
        parent,
        children,  # type: ignore[arg-type]
        evaluations,
        use_high_signal_gate=True,
    )

    assert [(child, score) for child, _evaluation, score in selected] == [
        (children[0], 0.75),
        (children[2], 1.0),
    ]


def test_high_signal_screen_returns_empty_when_every_child_is_rejected():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    parent = _batch([0.0, 0.0, 0.0, 0.0])

    selected = _select_screened_children(
        adapter,
        parent,
        [object(), object()],  # type: ignore[list-item]
        [_batch([1.0, 1.0, 0.0, 0.0]), _batch([1.0, 0.0, 0.0, 0.0])],
        use_high_signal_gate=True,
    )

    assert selected == []


def test_high_signal_batch_evaluation_dispatches_children_concurrently():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    barrier = threading.Barrier(2)

    def evaluate_fn(_batch_data, _candidate, _capture_traces):
        barrier.wait(timeout=1)
        return _batch([1.0])

    adapter._evaluate_fn = evaluate_fn
    items = [
        (
            {"WRITING_CODE": f"child-{index}"},
            [
                {
                    "eval_set_name": "set",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_entry_ids": ["entry"],
                }
            ],
        )
        for index in range(2)
    ]

    results = adapter.batch_evaluate(items)

    assert len(results) == 2
