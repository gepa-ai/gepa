# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end invariants for the multi-proposal machinery (#369) on a fully
deterministic synthetic task — no network, no recorded cache, so it can drive
the NEW paths (PxN/SameParent/Independent sampling, Best/TopK selection, the
adapter batch_evaluate seam, evaluation caching) that the replay-based tests
cannot reach.

Invariants:
  I1 budget ledger == independently instrumented evaluator truth == result field
  I2 candidate count == 1 + accepted events; lineage indices sane
  I3 selection is margin-correct: accepted == min(K, improving); no dropped
     margin above the selection cutoff; rejection reasons truthful per class
  I4 evaluation event pairs balanced; one child-eval pair per proposal
  I5 trace records every task with scores; save/load round-trips; state consistent
  I6 batched adapters receive grouped calls
  I7 determinism: identical config -> identical outcome
"""

import re
from collections import defaultdict
from itertools import pairwise

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState
from gepa.strategies.proposal_sampling import (
    IndependentSampling,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
)
from gepa.strategies.proposal_selection import AllImprovements, BestImprovement, TopKImprovements

TRAIN = [{"k": i} for i in range(10)]


def _level_of(candidate):
    return int(re.search(r"LEVEL=(\d+)", candidate["system_prompt"]).group(1))


class SyntheticAdapter:
    """score(candidate, example k) = 1.0 iff LEVEL >= k. Deterministic."""

    propose_new_texts = None

    def __init__(self):
        self.metric_calls = 0
        self.grouped_calls: list[int] = []

    def evaluate(self, batch, candidate, capture_traces=False):
        level = _level_of(candidate)
        self.metric_calls += len(batch)
        return EvaluationBatch(
            outputs=[f"out-l{level}-k{ex['k']}" for ex in batch],
            scores=[1.0 if level >= ex["k"] else 0.0 for ex in batch],
            trajectories=(
                [{"k": ex["k"], "feedback": f"level={level} needed k={ex['k']}"} for ex in batch]
                if capture_traces
                else None
            ),
            objective_scores=None,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(self, candidate, eval_batch, components):
        return {c: [{"Feedback": t["feedback"]} for t in (eval_batch.trajectories or [])] for c in components}


class BatchedSyntheticAdapter(SyntheticAdapter):
    def batch_evaluate(self, items):
        self.grouped_calls.append(len(items))
        return [self.evaluate(batch, candidate, capture_traces=True) for candidate, batch in items]


class SyntheticReflectionLM:
    """child level = parent + (calls % 4): delta 0 children never improve, so
    acceptance and selection both do real work."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        m = re.findall(r"LEVEL=(\d+)", prompt)
        parent_level = int(m[-1]) if m else 0
        delta = self.calls % 4
        self.calls += 1
        return f"```\nLEVEL={parent_level + delta}\n```"


class _EventLog:
    def __init__(self):
        self.by = defaultdict(list)

    def __getattr__(self, name):
        if name.startswith("on_"):

            def rec(e, n=name):
                self.by[n].append(dict(e))

            return rec
        raise AttributeError(name)


def _run(sampling, selection, adapter_cls, cache, run_dir):
    adapter = adapter_cls()
    log = _EventLog()
    result = gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=adapter,
        reflection_lm=SyntheticReflectionLM(),
        max_metric_calls=400,
        sampling_strategy=sampling,
        selection_strategy=selection,
        cache_evaluation=cache,
        callbacks=[log],
        run_dir=str(run_dir),
        seed=0,
        display_progress_bar=False,
    )
    return result, adapter, log.by


def state_trace_entries(run_dir):
    state = GEPAState.load(str(run_dir))
    assert state.is_consistent()
    return [t for t in state.full_program_trace if "n_tasks" in t]


def _assert_invariants(result, adapter, by, run_dir, expect_tasks, topk):
    # I1 — budget truth
    budget = by["on_budget_updated"]
    assert budget, "no budget events"
    assert budget[-1]["metric_calls_used"] == adapter.metric_calls
    assert result.total_metric_calls == adapter.metric_calls
    for prev, cur in pairwise(budget):
        assert cur["metric_calls_used"] == prev["metric_calls_used"] + cur["metric_calls_delta"]
    assert budget[0]["metric_calls_used"] - budget[0]["metric_calls_delta"] == len(TRAIN)

    # I2 — candidates and lineage
    accepted = by["on_candidate_accepted"]
    assert len(result.candidates) == 1 + len(accepted)
    for e in accepted:
        assert all(0 <= pid < e["new_candidate_idx"] for pid in e["parent_ids"])

    # I2b — every accepted program index appears in the trace (no overwrite)
    traced = [idx for t in state_trace_entries(run_dir) for idx in t.get("new_program_indices", [])]
    assert sorted(traced) == list(range(1, len(result.candidates)))

    # I4 — event pairing
    n_prop = len(by["on_proposal_end"])
    assert len(by["on_evaluation_start"]) == len(by["on_evaluation_end"])
    assert len([e for e in by["on_evaluation_end"] if e.get("candidate_idx") is None]) == n_prop

    # I5 — trace + save/load
    entries = state_trace_entries(run_dir)
    assert entries, "no trace entries with task records"
    assert all(t["n_tasks"] == expect_tasks and len(t["tasks"]) == expect_tasks for t in entries)
    assert sum(1 for t in entries for task in t["tasks"] if "new_subsample_scores" in task) == n_prop
    assert all(t["selected_program_candidate"] == t["tasks"][0]["parent_idx"] for t in entries if t["tasks"])

    # I3 — margin-correct selection + truthful rejection classes
    rejected = by["on_candidate_rejected"]
    sel_dropped = [e for e in rejected if "not selected by the selection strategy" in e["reason"]]
    duplicates = [e for e in rejected if "Duplicate of another candidate" in e["reason"]]
    crit_rejected = [e for e in rejected if e not in sel_dropped and e not in duplicates]
    for e in sel_dropped:
        assert e["new_score"] > e["old_score"], "selection-dropped reason on a non-improving proposal"
    for e in crit_rejected:
        assert not e["new_score"] > e["old_score"], "criterion-rejection reason on an improving proposal"
    if topk is not None:
        acc_by_iter = defaultdict(int)
        for e in accepted:
            acc_by_iter[e["iteration"]] += 1
        margins_by_iter = {}
        it = 0
        for t in entries:
            it += 1
            margins_by_iter[it] = sorted(
                (
                    sum(task["new_subsample_scores"]) - sum(task["subsample_scores"])
                    for task in t["tasks"]
                    if "new_subsample_scores" in task
                ),
                reverse=True,
            )
        dup_by_iter = defaultdict(int)
        for e in duplicates:
            dup_by_iter[e["iteration"]] += 1
        for it, margins in margins_by_iter.items():
            improving = [m for m in margins if m > 0]
            expected = min(topk, len(improving))
            got = acc_by_iter.get(it, 0)
            # Identical selected children are deduplicated, so accepted may
            # fall below the naive expectation by exactly the duplicates removed.
            assert expected - dup_by_iter.get(it, 0) <= got <= expected
            drop_margins = [e["new_score"] - e["old_score"] for e in sel_dropped if e["iteration"] == it]
            if drop_margins and len(improving) >= topk:
                assert max(drop_margins) <= improving[topk - 1]

    # I6 — batched adapters get grouped calls in multi-task iterations
    if isinstance(adapter, BatchedSyntheticAdapter) and expect_tasks > 1:
        assert any(n > 1 for n in adapter.grouped_calls)


CONFIGS = [
    ("single-all-plain", SingleMutationSampling, AllImprovements, SyntheticAdapter, False, 1, None),
    ("pxn4x4-top2-plain", lambda: PxNSampling(p=4, n=4), lambda: TopKImprovements(k=2), SyntheticAdapter, False, 16, 2),
    ("pxn4x4-best-batched-cache", lambda: PxNSampling(p=4, n=4), BestImprovement, BatchedSyntheticAdapter, True, 16, 1),
    ("same4-all-batched", lambda: SameParentSampling(n=4), AllImprovements, BatchedSyntheticAdapter, False, 4, None),
    ("indep4-top2-cache", lambda: IndependentSampling(n=4), lambda: TopKImprovements(k=2), SyntheticAdapter, True, 4, 2),
]


@pytest.mark.parametrize("name,sampling,selection,adapter_cls,cache,expect_tasks,topk", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_multi_proposal_invariants(tmp_path, name, sampling, selection, adapter_cls, cache, expect_tasks, topk):
    result, adapter, by = _run(sampling(), selection(), adapter_cls, cache, tmp_path / name)
    _assert_invariants(result, adapter, by, tmp_path / name, expect_tasks, topk)


def test_multi_proposal_determinism(tmp_path):
    r1, _a1, _ = _run(PxNSampling(p=4, n=4), TopKImprovements(k=2), SyntheticAdapter, False, tmp_path / "d1")
    r2, _a2, _ = _run(PxNSampling(p=4, n=4), TopKImprovements(k=2), SyntheticAdapter, False, tmp_path / "d2")
    assert r1.best_candidate == r2.best_candidate
    assert r1.total_metric_calls == r2.total_metric_calls
    assert len(r1.candidates) == len(r2.candidates)


class ConstantReflectionLM:
    """Always proposes the same improved text — every task in an iteration
    yields an identical child."""

    def __call__(self, prompt):
        return "```\nLEVEL=9\n```"


def test_identical_children_are_deduplicated(tmp_path):
    """Two tasks proposing byte-identical children must yield ONE pool entry,
    one valset eval, and a truthful duplicate-rejection for the other."""
    adapter = SyntheticAdapter()
    log = _EventLog()
    result = gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=adapter,
        reflection_lm=ConstantReflectionLM(),
        max_metric_calls=80,
        sampling_strategy=SameParentSampling(n=2),
        callbacks=[log],
        run_dir=str(tmp_path / "dedup"),
        seed=0,
        display_progress_bar=False,
    )
    candidates = [c["system_prompt"] for c in result.candidates]
    assert len(candidates) == len(set(candidates)), f"duplicate pool entries: {candidates}"
    dup_rejections = [
        e for e in log.by["on_candidate_rejected"] if "Duplicate of another candidate" in e["reason"]
    ]
    assert dup_rejections, "expected duplicate-rejection events for identical children"


class EveryOtherCallCriterion:
    """Stateful acceptance: alternates verdicts BY CALL. If the engine judges
    any proposal more than once, verdicts flip and selection/classification
    disagree — the pre-#329 contract is one judgement per proposal."""

    def __init__(self):
        self.calls = 0

    def should_accept(self, proposal, state):
        self.calls += 1
        return self.calls % 2 == 1

    def reject_reason(self, proposal, state):
        return "rejected by every-other-call criterion"


def test_stateful_acceptance_criterion_judged_once_per_proposal(tmp_path):
    criterion = EveryOtherCallCriterion()
    log = _EventLog()
    _result = gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=SyntheticAdapter(),
        reflection_lm=SyntheticReflectionLM(),
        max_metric_calls=200,
        sampling_strategy=SameParentSampling(n=2),
        acceptance_criterion=criterion,
        callbacks=[log],
        run_dir=str(tmp_path / "stateful"),
        seed=0,
        display_progress_bar=False,
    )
    n_proposals = len(log.by["on_proposal_end"])
    assert criterion.calls == n_proposals, (
        f"criterion judged {criterion.calls} times for {n_proposals} proposals — must be exactly once each"
    )
    # Every-other acceptance with proposals present must accept roughly half.
    assert len(log.by["on_candidate_accepted"]) > 0
    # The custom reject_reason hook must be honored for criterion rejections.
    crit_rejects = [
        e
        for e in log.by["on_candidate_rejected"]
        if "not selected" not in e["reason"] and "Duplicate" not in e["reason"]
    ]
    assert all("every-other-call criterion" in e["reason"] for e in crit_rejects)
