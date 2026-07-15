# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the user-facing ``batch_evaluator`` path (BatchEvaluatorWrapper +
OptimizeAnythingAdapter._batch_evaluate_via_user_fn).

Contract under test: the batch path preserves evaluator-path parity for
everything that survives a single external batch call — str-candidate
unwrapping, per-pair ``opt_states`` injection, exception policy, best-evals
(warm-start) history, and output packaging — while the user keeps exactly one
call for all pairs.
"""

import pytest

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import (
    BatchEvaluatorWrapper,
    OptimizeAnythingAdapter,
)

# ---------------------------------------------------------------------------
# BatchEvaluatorWrapper unit tests
# ---------------------------------------------------------------------------


PAIRS = [({"text": "a"}, 1), ({"text": "b"}, 2)]


def test_wrapper_normalizes_all_result_shapes():
    def fn(pairs):
        return [0.5, (0.7, {"note": "two"}), (0.9, "ignored-output", {"note": "three"})][: len(pairs)]

    wrapper = BatchEvaluatorWrapper(fn)
    out = wrapper([({"t": "x"}, i) for i in range(3)])
    assert out == [(0.5, {}), (0.7, {"note": "two"}), (0.9, {"note": "three"})]


def test_wrapper_two_tuple_side_info_is_not_dropped():
    """(score, side_info) — the natural mirror of the Evaluator protocol."""
    wrapper = BatchEvaluatorWrapper(lambda pairs: [(1.0, {"bias": "0"}) for _ in pairs])
    out = wrapper(PAIRS)
    assert all(si == {"bias": "0"} for _, si in out)


def test_wrapper_length_mismatch_raises():
    wrapper = BatchEvaluatorWrapper(lambda pairs: [1.0])
    with pytest.raises(ValueError, match="returned 1 results but expected 2"):
        wrapper(PAIRS)


def test_wrapper_exception_policy_raise():
    def boom(pairs):
        raise RuntimeError("cluster down")

    with pytest.raises(RuntimeError):
        BatchEvaluatorWrapper(boom, raise_on_exception=True)(PAIRS)


def test_wrapper_exception_policy_convert():
    def boom(pairs):
        raise RuntimeError("cluster down")

    out = BatchEvaluatorWrapper(boom, raise_on_exception=False)(PAIRS)
    assert out == [(0.0, {"error": "cluster down"})] * 2


def test_wrapper_str_candidate_unwrapping():
    seen = []

    def fn(pairs):
        seen.extend(c for c, _ in pairs)
        return [0.0 for _ in pairs]

    BatchEvaluatorWrapper(fn, str_candidate_key="text")(PAIRS)
    assert seen == ["a", "b"]


def test_wrapper_injects_opt_states_when_signature_accepts():
    received = {}

    def fn(pairs, opt_states):
        received["opt_states"] = opt_states
        return [0.0 for _ in pairs]

    states = ["s1", "s2"]
    BatchEvaluatorWrapper(fn)(PAIRS, opt_states=states)
    assert received["opt_states"] is states


def test_wrapper_omits_opt_states_when_signature_does_not_accept():
    def fn(pairs):
        return [0.0 for _ in pairs]

    # Must not raise TypeError for the unexpected kwarg.
    out = BatchEvaluatorWrapper(fn)(PAIRS, opt_states=["s1", "s2"])
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Adapter-level parity tests
# ---------------------------------------------------------------------------


def _sequential_evaluator(candidate, example=None, **kwargs):
    score = float(example) + float(candidate["bias"])
    return score, None, {"example": example, "bias": candidate["bias"]}


def _make_adapter(batch_evaluator=None):
    return OptimizeAnythingAdapter(
        evaluator=_sequential_evaluator,
        parallel=False,
        cache_mode="off",
        batch_evaluator=batch_evaluator,
    )


ITEMS = [
    ({"bias": "0"}, [1, 2, 3]),
    ({"bias": "100"}, [5]),
]


def _user_batch_fn(pairs):
    return [(float(ex) + float(c["bias"]), {"example": ex, "bias": c["bias"]}) for c, ex in pairs]


def test_batch_path_packaging_matches_sequential_path():
    """outputs must be (score, candidate, side_info) on BOTH paths (H4/B1)."""
    sequential = _make_adapter().batch_evaluate(ITEMS)
    batched = _make_adapter(batch_evaluator=_user_batch_fn).batch_evaluate(ITEMS)

    for seq_batch, usr_batch in zip(sequential, batched, strict=True):
        assert usr_batch.scores == seq_batch.scores
        assert usr_batch.num_metric_calls == seq_batch.num_metric_calls
        assert usr_batch.objective_scores == seq_batch.objective_scores
        for seq_out, usr_out in zip(seq_batch.outputs, usr_batch.outputs, strict=True):
            # (score, candidate, side_info) triple parity
            assert usr_out[0] == seq_out[0]
            assert usr_out[1] == seq_out[1]
            assert usr_out[2]["bias"] == seq_out[2]["bias"]
    # And specifically: slot 1 is the candidate, never None (H4 regression).
    assert all(out[1] == {"bias": "0"} for out in batched[0].outputs)
    assert batched[1].outputs[0][1] == {"bias": "100"}


def test_batch_path_updates_best_evals_for_warm_start():
    """B1: the top-K best-eval buffer must be populated on the batch path."""
    adapter = _make_adapter(batch_evaluator=_user_batch_fn)
    adapter.batch_evaluate(ITEMS)
    opt_state = adapter._build_opt_state(1)
    assert opt_state.best_example_evals, "best_example_evals not populated by batch path"


def test_batch_path_receives_per_pair_opt_states():
    received = []

    def fn(pairs, opt_states):
        received.append(list(opt_states))
        return [(0.0, {}) for _ in pairs]

    adapter = _make_adapter(batch_evaluator=fn)
    adapter.batch_evaluate(ITEMS)
    assert len(received) == 1
    assert len(received[0]) == 4  # one OptimizationState per flattened pair
