# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for OptimizeAnythingAdapter.batch_evaluate's batch-aware parallel fallback.

When the engine hands multiple (candidate, batch) items to ``batch_evaluate``
(parallel proposals / multi-accept iterations), the no-refiner path should fan
*all* (candidate, example) pairs out across a single thread pool — saturating
``max_workers`` across candidates — rather than running one underfilled pool per
candidate.
"""

import threading

import pytest

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter


def _make_evaluator(barrier: threading.Barrier | None = None):
    """Evaluator returning the (score, output, side_info) 3-tuple the adapter expects.

    score = example + candidate["bias"]; an optional barrier lets a test assert
    that N pairs run concurrently.
    """

    def _eval(candidate, example=None, opt_state=None):
        if barrier is not None:
            barrier.wait(timeout=5)
        score = float(example) + float(candidate["bias"])
        side_info = {"example": example, "bias": candidate["bias"]}
        output = (score, candidate, side_info)
        return score, output, side_info

    return _eval


def test_batch_evaluate_fans_all_pairs_concurrently():
    # 2 candidates x 2 examples = 4 pairs. A Barrier(4) only releases when all
    # four evaluator calls are in flight at once. The old per-candidate path
    # (2-wide waves) could never assemble 4 at the barrier -> BrokenBarrierError.
    barrier = threading.Barrier(4)
    adapter = OptimizeAnythingAdapter(
        evaluator=_make_evaluator(barrier=barrier),
        parallel=True,
        max_workers=8,
        cache_mode="off",
    )
    items = [
        ({"bias": "0"}, [1, 2]),
        ({"bias": "10"}, [3, 4]),
    ]

    results = adapter.batch_evaluate(items)

    assert [r.scores for r in results] == [[1.0, 2.0], [13.0, 14.0]]


def test_batch_evaluate_regroups_results_per_item():
    # Uneven batch sizes: results must be regrouped back to the right item and
    # keep per-item example order.
    adapter = OptimizeAnythingAdapter(
        evaluator=_make_evaluator(),
        parallel=True,
        max_workers=4,
        cache_mode="off",
    )
    items = [
        ({"bias": "0"}, [1, 2, 3]),
        ({"bias": "100"}, [5]),
    ]

    results = adapter.batch_evaluate(items)

    assert len(results) == 2
    assert results[0].scores == [1.0, 2.0, 3.0]
    assert results[1].scores == [105.0]
    # outputs carry the candidate that produced them
    assert all(out[1]["bias"] == "0" for out in results[0].outputs)
    assert results[1].outputs[0][1]["bias"] == "100"
    # trajectories (side_infos) line up with examples, in order
    assert [t["example"] for t in results[0].trajectories] == [1, 2, 3]
    assert [t["example"] for t in results[1].trajectories] == [5]


def test_batch_evaluate_matches_per_item_evaluate():
    # The batched fallback must produce results identical to calling evaluate()
    # once per item (the behavior it replaces).
    adapter = OptimizeAnythingAdapter(
        evaluator=_make_evaluator(),
        parallel=True,
        max_workers=4,
        cache_mode="off",
    )
    items = [
        ({"bias": "0"}, [1, 2]),
        ({"bias": "5"}, [3]),
    ]

    batched = adapter.batch_evaluate(items)
    per_item = [adapter.evaluate(batch, candidate, capture_traces=True) for candidate, batch in items]

    for b, p in zip(batched, per_item, strict=True):
        assert b.scores == p.scores
        assert b.objective_scores == p.objective_scores
        assert b.num_metric_calls == p.num_metric_calls


@pytest.mark.parametrize("parallel", [True, False])
def test_batch_evaluate_single_item_unchanged(parallel):
    # A single item is not routed through the cross-candidate fan-out; it still
    # produces correct results on both parallel and sequential settings.
    adapter = OptimizeAnythingAdapter(
        evaluator=_make_evaluator(),
        parallel=parallel,
        max_workers=4,
        cache_mode="off",
    )
    items = [({"bias": "1"}, [1, 2, 3])]

    results = adapter.batch_evaluate(items)

    assert len(results) == 1
    assert results[0].scores == [2.0, 3.0, 4.0]
