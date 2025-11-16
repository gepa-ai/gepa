import pytest

from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.interfaces import Candidate, EvalResult


def make_scheduler():
    shards = (0.2, 0.5, 1.0)
    variance_tolerance = {0.2: 0.08, 0.5: 0.05, 1.0: 0.02}
    shrinkage_alpha = {0.2: 0.7, 0.5: 0.85, 1.0: 1.0}
    cfg = SchedulerConfig(
        shards=shards,
        variance_tolerance=variance_tolerance,
        shrinkage_alpha=shrinkage_alpha,
    )
    return BudgetedScheduler(cfg)


def test_promote_within_tolerance_parent_same_rung():
    sched = make_scheduler()

    # Parent (seed) evaluated at rung 0 (20%) with 0.60 -> promotes to next rung
    parent = Candidate(text="parent", meta={})
    res_parent_r0 = EvalResult(objectives={"quality": 0.60}, traces=[], n_examples=1, shard_fraction=0.2)
    d1 = sched.record(parent, res_parent_r0, "quality")
    assert d1 == "promoted"

    # Child at rung 0 with score slightly below parent but within tolerance (0.08)
    child = Candidate(
        text="child",
        meta={
            "parent_objectives": {"quality": 0.60},
            "parent_sched_key": parent.fingerprint,
        },
    )
    res_child_r0 = EvalResult(objectives={"quality": 0.56}, traces=[], n_examples=1, shard_fraction=0.2)
    d2 = sched.record(child, res_child_r0, "quality")
    assert d2 == "promoted", "Child within variance tolerance should promote"


def test_prune_when_below_tolerance():
    sched = make_scheduler()

    # Parent rung 1 (50%) score 0.70
    parent = Candidate(text="parent", meta={})
    # First evaluate parent on rung 1: we simulate by first recording rung 0 (seed promote), then rung 1
    res_parent_r0 = EvalResult(objectives={"quality": 0.60}, traces=[], n_examples=1, shard_fraction=0.2)
    _ = sched.record(parent, res_parent_r0, "quality")  # seed promote to rung 1
    # Now artificially set the scheduler's current level to 1 for parent by calling record at rung 1
    # Since current_shard_index(parent) is 1, the rung_fraction resolves to 0.5
    res_parent_r1 = EvalResult(objectives={"quality": 0.70}, traces=[], n_examples=1, shard_fraction=0.5)
    _ = sched.record(parent, res_parent_r1, "quality")

    # Child at rung 0.5 with score below parent - tolerance (0.05)
    child = Candidate(
        text="child",
        meta={
            "parent_objectives": {"quality": 0.70},
            "parent_sched_key": parent.fingerprint,
        },
    )
    # Place child at rung index 1 directly for the test (internal state adjustment)
    sched._candidate_levels[sched._sched_key(child)] = 1
    # Record the child at rung 1 (50%)
    res_child_r1 = EvalResult(objectives={"quality": 0.64}, traces=[], n_examples=1, shard_fraction=0.5)
    decision = sched.record(child, res_child_r1, "quality")
    assert decision == "pruned", "Child below parent minus tolerance should be pruned"


def test_shrinkage_fallback_promotes():
    # No parent rung score present -> use shrinkage from parent final
    sched = make_scheduler()

    child = Candidate(
        text="child",
        meta={
            # Parent final = 0.60 → shrinkage at 0.2 rung: (1-0.7)*0.5 + 0.7*0.6 = 0.57
            # Tolerance at 0.2 = 0.08 → threshold = 0.57 - 0.08 = 0.49
            # Child 0.50 >= 0.49 → promote
            "parent_objectives": {"quality": 0.60},
            # Intentionally omit parent_sched_key to force shrinkage path
        },
    )
    res_child_r0 = EvalResult(objectives={"quality": 0.50}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = sched.record(child, res_child_r0, "quality")
    assert decision == "promoted"


def test_final_rung_confidence_flag():
    sched = make_scheduler()
    parent = Candidate(text="parent", meta={})
    res_parent_r0 = EvalResult(objectives={"quality": 0.70}, traces=[], n_examples=1, shard_fraction=0.2)
    _ = sched.record(parent, res_parent_r0, "quality")
    res_parent_r1 = EvalResult(objectives={"quality": 0.75}, traces=[], n_examples=1, shard_fraction=0.5)
    _ = sched.record(parent, res_parent_r1, "quality")

    child = Candidate(
        text="child",
        meta={
            "parent_objectives": {"quality": 0.75},
            "parent_sched_key": parent.fingerprint,
        },
    )
    sched._candidate_levels[sched._sched_key(child)] = 2
    res_child_final = EvalResult(
        objectives={"quality": 0.90},
        traces=[],
        n_examples=80,
        shard_fraction=1.0,
        coverage_fraction=0.8,
    )
    decision = sched.record(child, res_child_final, "quality")
    assert decision == "completed"
    assert child.meta.get("_confidence_met") is True


def test_final_rung_confidence_requires_samples():
    sched = make_scheduler()
    parent = Candidate(text="parent", meta={})
    res_parent_r0 = EvalResult(objectives={"quality": 0.70}, traces=[], n_examples=1, shard_fraction=0.2)
    _ = sched.record(parent, res_parent_r0, "quality")
    res_parent_r1 = EvalResult(objectives={"quality": 0.75}, traces=[], n_examples=1, shard_fraction=0.5)
    _ = sched.record(parent, res_parent_r1, "quality")

    child = Candidate(
        text="child",
        meta={
            "parent_objectives": {"quality": 0.75},
            "parent_sched_key": parent.fingerprint,
        },
    )
    sched._candidate_levels[sched._sched_key(child)] = 2
    res_child_final = EvalResult(
        objectives={"quality": 0.90},
        traces=[],
        n_examples=5,
        shard_fraction=1.0,
        coverage_fraction=0.2,
    )
    decision = sched.record(child, res_child_final, "quality")
    assert decision == "completed"
    assert child.meta.get("_confidence_met") is not True
