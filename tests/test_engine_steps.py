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

import contextlib

import pytest

import gepa
from gepa import StepOutcome, StepResult
from gepa.core.adapter import EvaluationBatch
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.proposer.base import CandidateProposal


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


# ---------------------------------------------------------------------------
# Step API: initialize / is_done / step / finalize
# ---------------------------------------------------------------------------


class _CapturedError(Exception):
    pass


def _capture_engine(monkeypatch, recorder=None, **overrides):
    """Build an engine exactly as ``gepa.optimize`` wires it, without running it."""
    box = {}

    def fake_run(self):
        box["engine"] = self
        raise _CapturedError

    monkeypatch.setattr(GEPAEngine, "run", fake_run)
    with contextlib.suppress(_CapturedError):
        _optimize(recorder or EventRecorder(), **overrides)
    return box["engine"]


def _drive(engine):
    engine.initialize()
    results = []
    while not engine.is_done():
        results.append(engine.step())
    return engine.finalize(), results


def test_manual_drive_matches_run(monkeypatch):
    rec_run = EventRecorder()
    via_run = _optimize(rec_run)

    rec_step = EventRecorder()
    engine = _capture_engine(monkeypatch, rec_step)
    seed_evals = engine.initialize().total_num_evals  # seed valset eval belongs to no step
    state, results = _drive(engine)
    via_steps = GEPAResult.from_state(state, run_dir=None, seed=0)

    assert _fingerprint(via_steps) == _fingerprint(via_run)
    assert rec_step.events == rec_run.events
    assert len(results) == len(rec_step.named("iteration_start"))
    assert all(isinstance(r, StepResult) for r in results)
    assert [r.iteration for r in results] == [s[1] for s in rec_step.named("iteration_start")]
    assert seed_evals == 4  # one eval per valset example
    assert sum(r.evals_consumed for r in results) == via_steps.total_metric_calls - seed_evals
    assert [i for r in results for i in r.new_candidate_indices] == list(range(1, len(via_steps.candidates)))


def test_step_api_requires_initialize(monkeypatch):
    engine = _capture_engine(monkeypatch)
    for method in (engine.step, engine.is_done, engine.finalize):
        with pytest.raises(RuntimeError, match="initialize"):
            method()


def test_initialize_is_idempotent(monkeypatch):
    engine = _capture_engine(monkeypatch)
    first = engine.initialize()
    evals_after_first = first.total_num_evals
    second = engine.initialize()
    assert second is first
    assert first.total_num_evals == evals_after_first, "seed must not be re-evaluated"


def test_step_reports_reflective_and_no_proposal_outcomes(monkeypatch):
    engine = _capture_engine(monkeypatch)
    engine.initialize()

    accepted = engine.step()
    assert accepted.outcome is StepOutcome.REFLECTIVE_ACCEPTED
    assert accepted.proposal_accepted is True
    assert accepted.new_candidate_indices == (1,)
    assert accepted.iteration == 1
    assert accepted.evals_consumed > 0

    monkeypatch.setattr(engine.reflective_proposer, "propose", lambda state: [])
    empty = engine.step()
    assert empty.outcome is StepOutcome.NO_PROPOSAL
    assert empty.proposal_accepted is False
    assert empty.new_candidate_indices == ()
    assert empty.iteration == 2


def test_step_reports_reflective_rejected(monkeypatch):
    class WorseningAdapter(LengthScoredAdapter):
        """Longer prompts score lower, so every proposal loses to its parent."""

        def evaluate(self, batch, candidate, capture_traces=False):
            score = max(0.0, 1.0 - len(candidate["system_prompt"]) / 80.0)
            trajectories = [{"score": score} for _ in batch] if capture_traces else None
            return EvaluationBatch(outputs=[score] * len(batch), scores=[score] * len(batch), trajectories=trajectories)

    engine = _capture_engine(monkeypatch, adapter=WorseningAdapter())
    engine.initialize()
    result = engine.step()
    assert result.outcome is StepOutcome.REFLECTIVE_REJECTED
    assert result.proposal_accepted is False
    assert result.new_candidate_indices == ()
    assert result.evals_consumed > 0


class _StubMergeProposer:
    """Bypasses merge discovery so the engine's merge branch can be exercised directly."""

    def __init__(self, after_score):
        self.use_merge = True
        self.merges_due = 1
        self.last_iter_found_new_program = True
        self.total_merges_tested = 0
        self.max_merge_invocations = 5
        self._after_score = after_score

    def propose(self, state):
        return CandidateProposal(
            candidate={"system_prompt": "merged " + "x" * 40},
            parent_program_ids=[0, 1],
            subsample_scores_before=[0.2, 0.2],
            subsample_scores_after=[self._after_score],
            tag="merge",
        )


def test_step_reports_merge_outcomes(monkeypatch):
    engine = _capture_engine(monkeypatch, use_merge=True)
    engine.initialize()
    assert engine.step().outcome is StepOutcome.REFLECTIVE_ACCEPTED  # candidate 1 exists now

    engine.merge_proposer = _StubMergeProposer(after_score=0.9)
    accepted = engine.step()
    assert accepted.outcome is StepOutcome.MERGE_ACCEPTED
    assert accepted.proposal_accepted is True
    assert accepted.new_candidate_indices == (2,)
    assert engine.merge_proposer.merges_due == 0
    assert engine.merge_proposer.total_merges_tested == 1
    assert engine._state.parent_program_for_candidate[2] == [0, 1]

    engine.merge_proposer = _StubMergeProposer(after_score=0.0)
    rejected = engine.step()
    assert rejected.outcome is StepOutcome.MERGE_REJECTED
    assert rejected.proposal_accepted is False
    assert rejected.new_candidate_indices == ()
    # A rejected merge does not consume the scheduled attempt.
    assert engine.merge_proposer.merges_due == 1
    assert engine.merge_proposer.total_merges_tested == 0


def test_interleaved_engines_do_not_interfere(monkeypatch):
    """Island-model smoke test (#61): two engines advanced alternately must each
    produce exactly what they would have produced running alone."""
    solo_a = _optimize(EventRecorder(), max_metric_calls=50)
    solo_b = _optimize(EventRecorder(), max_metric_calls=70)

    a = _capture_engine(monkeypatch, max_metric_calls=50)
    b = _capture_engine(monkeypatch, max_metric_calls=70)
    a.initialize()
    b.initialize()
    while not (a.is_done() and b.is_done()):
        if not a.is_done():
            a.step()
        if not b.is_done():
            b.step()
    island_a = GEPAResult.from_state(a.finalize(), run_dir=None, seed=0)
    island_b = GEPAResult.from_state(b.finalize(), run_dir=None, seed=0)

    assert _fingerprint(island_a) == _fingerprint(solo_a)
    assert _fingerprint(island_b) == _fingerprint(solo_b)
    assert _fingerprint(island_a) != _fingerprint(island_b)


def test_request_stop_is_visible_through_is_done(monkeypatch):
    engine = _capture_engine(monkeypatch)
    engine.initialize()
    assert not engine.is_done()
    engine.step()
    engine.request_stop()
    assert engine.is_done()
