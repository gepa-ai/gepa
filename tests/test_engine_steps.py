# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Characterization tests for ``GEPAEngine.run()`` (issue #116).

These lock in the observable behavior of the monolithic ``run()`` loop before it
is split into ``initialize`` / ``step`` / ``is_done`` / ``finalize`` so that the
refactor can be shown to be behavior-preserving: same candidate tree, same
budget accounting, same callback sequence, same resume boundary.

Everything here is LLM-free: the adapter proposes deterministically and scores
by prompt length, so two runs with the same seed must be byte-identical.
"""

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch


class LengthScoredAdapter:
    """Each proposal lengthens the prompt; score grows with length, so every
    proposal beats its parent. Mirrors ``ImprovingAdapter`` in
    ``test_parallel_valset_eval.py`` but kept local so this module has no
    cross-test imports."""

    def __init__(self):
        self.propose_new_texts = self._propose_new_texts
        self._suffix = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        score = min(1.0, len(candidate["system_prompt"]) / 80.0)
        trajectories = [{"score": score} for _ in batch] if capture_traces else None
        return EvaluationBatch(outputs=[score] * len(batch), scores=[score] * len(batch), trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: [{"score": s} for s in eval_batch.scores] for c in components_to_update}

    def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for c in components_to_update:
            self._suffix += 1
            out[c] = candidate.get(c, "") + f" w{self._suffix}"
        return out


class EventRecorder:
    """Records the lifecycle callbacks ``run()`` emits, in order."""

    def __init__(self):
        self.events: list[tuple] = []

    def on_optimization_start(self, event):
        self.events.append(("optimization_start", None))

    def on_state_saved(self, event):
        self.events.append(("state_saved", event["iteration"]))

    def on_iteration_start(self, event):
        self.events.append(("iteration_start", event["iteration"]))

    def on_candidate_accepted(self, event):
        self.events.append(("candidate_accepted", event["iteration"], event["new_candidate_idx"]))

    def on_iteration_end(self, event):
        self.events.append(("iteration_end", event["iteration"], event["proposal_accepted"]))

    def on_error(self, event):
        self.events.append(("error", event["iteration"], event["will_continue"]))

    def on_optimization_end(self, event):
        self.events.append(("optimization_end", event["total_iterations"], event["total_metric_calls"]))

    def named(self, name):
        return [e for e in self.events if e[0] == name]


def _optimize(recorder, *, max_metric_calls=60, run_dir=None, **overrides):
    kwargs = {
        "seed_candidate": {"system_prompt": "s"},
        "trainset": [{"id": i, "split": "train"} for i in range(4)],
        "valset": [{"id": i, "split": "val"} for i in range(4)],
        "adapter": LengthScoredAdapter(),
        "reflection_lm": None,
        "max_metric_calls": max_metric_calls,
        "callbacks": [recorder],
        "run_dir": run_dir,
        "seed": 0,
    }
    kwargs.update(overrides)
    return gepa.optimize(**kwargs)


def _fingerprint(result):
    """Everything a behavior-preserving refactor must keep identical."""
    return {
        "candidates": result.candidates,
        "parents": result.parents,
        "val_aggregate_scores": result.val_aggregate_scores,
        "val_subscores": result.val_subscores,
        "discovery_eval_counts": result.discovery_eval_counts,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
    }


def test_run_is_deterministic_across_identical_runs():
    rec_a, rec_b = EventRecorder(), EventRecorder()
    result_a = _optimize(rec_a)
    result_b = _optimize(rec_b)

    assert len(result_a.candidates) > 1, "budget must allow at least one accepted proposal"
    assert _fingerprint(result_a) == _fingerprint(result_b)
    assert rec_a.events == rec_b.events


def test_lifecycle_callback_sequence():
    rec = EventRecorder()
    result = _optimize(rec)

    assert rec.events[0] == ("optimization_start", None)
    assert rec.events[-1][0] == "optimization_end"

    starts = rec.named("iteration_start")
    ends = rec.named("iteration_end")
    saves = rec.named("state_saved")

    # Every iteration that starts also ends, with the same iteration number, in order.
    assert [s[1] for s in starts] == [e[1] for e in ends]
    assert [s[1] for s in starts] == list(range(starts[0][1], starts[0][1] + len(starts)))

    # One save per loop pass. The save fires *before* ``state.i`` advances while
    # iteration_start fires *after*, so the save reports one less than the
    # iteration it precedes. This is current behavior and must survive the refactor.
    assert len(saves) == len(starts)
    assert all(save[1] + 1 == start[1] for save, start in zip(saves, starts, strict=True))

    # optimization_end reports ``state.i`` and the final budget. ``state.i`` starts
    # at -1 (state.py), so it lags the number of loop passes by one. Current
    # behavior; must survive the refactor unchanged.
    _, total_iterations, total_metric_calls = rec.events[-1]
    assert total_iterations == len(starts) - 1
    assert total_metric_calls == result.total_metric_calls

    # Every accepted candidate is reported inside the iteration that accepted it,
    # and that iteration's end event says so.
    accepted_iters = {e[1] for e in rec.named("candidate_accepted")}
    assert accepted_iters <= {s[1] for s in starts}
    assert all(e[2] for e in ends if e[1] in accepted_iters)

    # Candidate indices are handed out sequentially starting after the seed.
    assert [e[2] for e in rec.named("candidate_accepted")] == list(range(1, len(result.candidates)))


def test_budget_is_respected_and_reported():
    rec = EventRecorder()
    result = _optimize(rec, max_metric_calls=40)
    # The stop check runs once per loop pass, so the last pass may overshoot by
    # at most one iteration's worth of evals; never by more.
    assert result.total_metric_calls >= 40
    assert result.total_metric_calls == rec.events[-1][2]


def test_resume_from_run_dir_preserves_state(tmp_path):
    run_dir = str(tmp_path / "run")
    first = _optimize(EventRecorder(), max_metric_calls=40, run_dir=run_dir)
    assert (tmp_path / "run" / "gepa_state.bin").exists()

    # Resuming with a zero budget must return the persisted state untouched.
    rec = EventRecorder()
    second = _optimize(rec, max_metric_calls=0, run_dir=run_dir)

    assert _fingerprint(second) == _fingerprint(first)
    assert rec.named("iteration_start") == []
    assert rec.events[-1] == ("optimization_end", rec.events[-1][1], first.total_metric_calls)


def test_resume_continues_from_checkpoint_not_from_scratch(tmp_path):
    run_dir = str(tmp_path / "run")
    first = _optimize(EventRecorder(), max_metric_calls=40, run_dir=run_dir)

    rec = EventRecorder()
    second = _optimize(rec, max_metric_calls=first.total_metric_calls + 20, run_dir=run_dir)

    # The resumed run extends the first run's candidate tree rather than rebuilding it.
    assert second.candidates[: len(first.candidates)] == first.candidates
    assert second.parents[: len(first.parents)] == first.parents
    assert len(second.candidates) > len(first.candidates)
    assert second.total_metric_calls > first.total_metric_calls
    # Iteration numbering continues from the checkpoint.
    assert rec.named("iteration_start")[0][1] > 1


class ExplodingOnValsetAdapter(LengthScoredAdapter):
    """Proposals succeed on the train minibatch, then the engine-owned full
    valset evaluation of the accepted child raises. Exceptions raised inside the
    proposer are retried and swallowed there, so this is the path that actually
    reaches ``run()``'s ``except``."""

    def evaluate(self, batch, candidate, capture_traces=False):
        if batch[0]["split"] == "val" and candidate["system_prompt"] != "s":
            raise RuntimeError("boom")
        return super().evaluate(batch, candidate, capture_traces)


def test_exception_with_raise_on_exception_true_propagates():
    rec = EventRecorder()
    with pytest.raises(RuntimeError, match="boom"):
        _optimize(rec, adapter=ExplodingOnValsetAdapter(), raise_on_exception=True)

    errors = rec.named("error")
    assert len(errors) == 1
    assert errors[0][2] is False  # will_continue
    # iteration_end still fires for the failed iteration (finally-block semantics).
    assert rec.named("iteration_end")[-1][1] == errors[0][1]
    assert rec.named("optimization_end") == []


def test_exception_with_raise_on_exception_false_continues_when_progress_was_made():
    rec = EventRecorder()
    result = _optimize(rec, adapter=ExplodingOnValsetAdapter(), raise_on_exception=False)

    errors = rec.named("error")
    assert len(errors) >= 2, "every iteration fails at the valset eval, run ends on budget"
    # Minibatch evals were consumed before the failure, so the engine treats
    # each failed iteration as progress and keeps going.
    assert all(e[2] is True for e in errors)
    assert result.candidates == [{"system_prompt": "s"}]
    assert rec.events[-1][0] == "optimization_end"
    # Every failed iteration still closes with iteration_end(proposal_accepted=False).
    assert {e[1] for e in errors} <= {e[1] for e in rec.named("iteration_end") if e[2] is False}
