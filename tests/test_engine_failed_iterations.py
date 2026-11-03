# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""With ``raise_on_exception=False`` a failed iteration is logged and the run continues (as in v0.1.4)."""

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.engine import MAX_FAILED_ITERATIONS_WITHOUT_PROGRESS

DATA = [{"q": i} for i in range(4)]


class FlakyAdapter:
    """Scores "improved" prompts 1.0; raises on the adapter calls listed in ``fail_on``."""

    propose_new_texts = None

    def __init__(self, fail_on=(), fail_always_after=None):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.fail_always_after = fail_always_after

    def evaluate(self, batch, candidate, capture_traces=False):
        self.calls += 1
        if self.calls in self.fail_on or (self.fail_always_after is not None and self.calls > self.fail_always_after):
            raise RuntimeError(f"transient failure on call {self.calls}")
        scores = [1.0 if "improved" in candidate["p"] else 0.0 for _ in batch]
        return EvaluationBatch(
            outputs=[None] * len(batch),
            scores=scores,
            trajectories=[{}] * len(batch) if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: [{"Feedback": "say improved"}] for c in components_to_update}


def _optimize(adapter, raise_on_exception=False, max_metric_calls=20):
    return gepa.optimize(
        seed_candidate={"p": "seed"},
        trainset=DATA,
        valset=DATA,
        adapter=adapter,
        reflection_lm=lambda prompt: "```\nimproved\n```",
        max_metric_calls=max_metric_calls,
        raise_on_exception=raise_on_exception,
    )


def test_one_failure_before_any_budget_is_spent_does_not_end_the_run():
    # Call 1 is the seed's valset evaluation; call 2 is the first minibatch evaluation.
    result = _optimize(FlakyAdapter(fail_on={2}))
    assert len(result.candidates) == 2
    assert result.val_aggregate_scores[result.best_idx] == 1.0


def test_raise_on_exception_true_still_raises_on_the_first_failure():
    with pytest.raises(RuntimeError, match="transient failure on call 2"):
        _optimize(FlakyAdapter(fail_on={2}), raise_on_exception=True)


def test_failures_without_progress_raise_after_the_cap():
    adapter = FlakyAdapter(fail_always_after=1)
    with pytest.raises(RuntimeError, match="transient failure"):
        _optimize(adapter, max_metric_calls=1000)
    assert adapter.calls == 1 + MAX_FAILED_ITERATIONS_WITHOUT_PROGRESS


def test_failures_that_spent_budget_do_not_count_toward_the_cap():
    # Fail on every other adapter call after the seed, so most failing iterations
    # have already spent part of their budget. No streak of failures without
    # progress reaches the cap, and the run ends on its budget instead of raising.
    fail_on = set(range(2, 400, 2))
    result = _optimize(FlakyAdapter(fail_on=fail_on), max_metric_calls=60)
    assert result.total_metric_calls >= 60
