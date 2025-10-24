from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig


def test_scheduler_promotes_after_good_score():
    scheduler = BudgetedScheduler(
        SchedulerConfig(shards=(0.1, 1.0), eps_improve=0.0, quantile=0.5)
    )
    # Need at least 2 samples to establish threshold
    baseline = Candidate(text="baseline")
    baseline_result = EvalResult(objectives={"quality": 0.4}, traces=[], n_examples=5)
    scheduler.record(baseline, baseline_result, objective_key="quality")

    candidate = Candidate(text="prompt")
    result = EvalResult(objectives={"quality": 0.8}, traces=[], n_examples=5)
    decision = scheduler.record(candidate, result, objective_key="quality")
    assert decision == "promoted"
    promotions = scheduler.promote_ready()
    assert candidate in promotions


def test_scheduler_promotes_single_candidate():
    scheduler = BudgetedScheduler(
        SchedulerConfig(shards=(0.1, 1.0), eps_improve=0.0, quantile=0.6)
    )
    candidate = Candidate(text="solo")
    result = EvalResult(objectives={"quality": 0.3}, traces=[], n_examples=5)
    decision = scheduler.record(candidate, result, objective_key="quality")
    assert decision == "promoted"
    promotions = scheduler.promote_ready()
    assert candidate in promotions


def test_scheduler_respects_parent_improvement():
    scheduler = BudgetedScheduler(
        SchedulerConfig(shards=(0.1, 1.0), eps_improve=0.1, quantile=0.5)
    )
    parent_result = EvalResult(objectives={"quality": 0.5}, traces=[], n_examples=5)
    parent_candidate = Candidate(text="parent")
    scheduler.record(parent_candidate, parent_result, objective_key="quality")
    child = Candidate(text="child", meta={"parent_objectives": {"quality": 0.5}})
    poor_result = EvalResult(objectives={"quality": 0.55}, traces=[], n_examples=5)
    decision = scheduler.record(child, poor_result, objective_key="quality")
    assert decision == "pruned"
    promotions = scheduler.promote_ready()
    assert child not in promotions


def test_scheduler_cleans_up_rung_on_promotion():
    """Test that when a candidate promotes, its results are removed from the previous rung.

    This test simulates the streaming scenario where multiple candidates have already
    been evaluated on a rung (bypassing the sequential single-candidate promotion issue),
    then a new strong candidate comes in and promotes.
    """
    from collections import deque
    from turbo_gepa.cache import candidate_key

    scheduler = BudgetedScheduler(
        SchedulerConfig(shards=(0.3, 1.0), eps_improve=0.0, quantile=0.5)
    )

    # Manually populate the rung with multiple candidates to establish threshold
    # This simulates async evaluation where multiple candidates finish around the same time
    weak = Candidate(text="weak")
    baseline = Candidate(text="baseline")
    candidate = Candidate(text="good")

    rung_0 = scheduler.rungs[0]

    # Directly add results to rung to bypass sequential single-candidate promotions
    weak_hash = candidate_key(weak)
    baseline_hash = candidate_key(baseline)
    cand_hash = candidate_key(candidate)

    rung_0.results[weak_hash] = deque([0.3], maxlen=64)
    rung_0.results[baseline_hash] = deque([0.5], maxlen=64)
    rung_0.results[cand_hash] = deque([0.9], maxlen=64)

    # Now when we record a new result for the good candidate, it should promote and clean up
    result = EvalResult(objectives={"quality": 0.95}, traces=[], n_examples=5, shard_fraction=0.3)
    decision = scheduler.record(candidate, result, objective_key="quality")

    # The good candidate should promote
    assert decision == "promoted"

    # The promoted candidate should NOT be in the first rung anymore
    assert cand_hash not in rung_0.results

    # But baseline and weak should still be there
    assert baseline_hash in rung_0.results
    assert weak_hash in rung_0.results
