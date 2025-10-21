from ufast_gepa.interfaces import Candidate, EvalResult
from ufast_gepa.scheduler import BudgetedScheduler, SchedulerConfig


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
