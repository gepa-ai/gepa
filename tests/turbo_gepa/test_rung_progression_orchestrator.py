"""Test rung progression in the full orchestrator context."""

import pytest

from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig


def test_promoted_candidate_preserves_parent_objectives():
    """
    Test that when a candidate is promoted, it should be re-evaluated
    on the next shard with its parent_objectives metadata intact so the
    scheduler can make proper promotion decisions.

    This is the ROOT CAUSE: When scheduler.promote_ready() returns promoted
    candidates, those candidates need to have parent_objectives set in their
    metadata so that when they're evaluated on the next shard, the scheduler
    knows what score they achieved on the previous shard.
    """
    config = SchedulerConfig(
        shards=[0.3, 1.0],
        eps_improve=0.0,
        quantile=0.6,
        enable_convergence=False,
        lineage_patience=0,
        lineage_min_improve=0.01,
    )
    scheduler = BudgetedScheduler(config)

    # Seed candidate evaluated on shard 0
    # Simulate what the orchestrator does: add _sched_key before calling scheduler.record()
    seed = Candidate(text="seed prompt", meta={})
    seed_with_sched_key = Candidate(
        text=seed.text,
        meta={"_sched_key": seed.fingerprint}  # Orchestrator adds this in _ingest_result
    )

    result_shard0 = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=3,
    )

    decision = scheduler.record(seed_with_sched_key, result_shard0, "quality")
    assert decision == "promoted"

    # Get the promoted candidate
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    promoted_seed = promotions[0]

    # THE BUG: The promoted candidate does NOT have parent_objectives in its metadata!
    # When the orchestrator re-evaluates it on the next shard, the scheduler won't
    # know its previous score.
    print(f"Promoted seed metadata: {promoted_seed.meta}")
    print(f"Original seed fingerprint: {seed.fingerprint}")
    print(f"Promoted seed fingerprint: {promoted_seed.fingerprint}")
    print(f"Scheduler current_shard_index for promoted: {scheduler.current_shard_index(promoted_seed)}")

    # This is what SHOULD happen: The orchestrator should add parent_objectives
    # before enqueuing the promoted candidate for re-evaluation.
    # Let's simulate what the orchestrator SHOULD do:

    # The orchestrator should update the promoted candidate's metadata to include
    # parent_objectives from the previous evaluation
    promoted_with_parent = promoted_seed.with_meta(
        parent_objectives=result_shard0.objectives
    )

    print(f"Promoted with parent fingerprint: {promoted_with_parent.fingerprint}")
    print(f"Fingerprints match: {promoted_seed.fingerprint == promoted_with_parent.fingerprint}")

    # Now when this candidate is evaluated on the next shard, the scheduler
    # will see that it matches or exceeds the parent score and promote it
    result_shard1 = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=10,
    )

    # Check current shard index before recording
    current_idx_before = scheduler.current_shard_index(promoted_with_parent)
    print(f"Current shard index before second eval: {current_idx_before}")

    decision2 = scheduler.record(promoted_with_parent, result_shard1, "quality")
    print(f"Second evaluation decision (with parent_objectives): {decision2}")

    current_idx_after = scheduler.current_shard_index(promoted_with_parent)
    print(f"Current shard index after second eval: {current_idx_after}")
    print(f"Number of shards: {len(config.shards)}")

    assert decision2 == "completed", "Should complete at final rung"

    # Compare to what happens WITHOUT parent_objectives:
    result_shard1_no_parent = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=10,
    )

    # Re-create scheduler to reset state
    scheduler2 = BudgetedScheduler(config)
    seed2 = Candidate(text="seed prompt 2", meta={})
    result2_shard0 = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=3,
    )
    scheduler2.record(seed2, result2_shard0, "quality")
    promotions2 = scheduler2.promote_ready()
    promoted_seed2 = promotions2[0]

    # Evaluate WITHOUT adding parent_objectives
    decision3 = scheduler2.record(promoted_seed2, result_shard1_no_parent, "quality")
    print(f"Second evaluation decision (WITHOUT parent_objectives): {decision3}")
    # Without parent_objectives, the scheduler has to rely on the quantile check
    # which might work if it's the only candidate, but could fail in a real scenario


if __name__ == "__main__":
    test_promoted_candidate_preserves_parent_objectives()
    print("Test passed!")
