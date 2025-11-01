"""Test rung progression and promotion logic."""

import pytest

from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig


def test_seed_candidate_promotion_through_rungs():
    """Test that a seed candidate (100% quality) properly progresses through all rungs."""
    config = SchedulerConfig(
        shards=[0.3, 1.0],
        eps_improve=0.0,
        quantile=0.6,
        enable_convergence=False,
        lineage_patience=0,
        lineage_min_improve=0.01,
    )
    scheduler = BudgetedScheduler(config)

    # Create seed candidate with 100% quality
    seed = Candidate(text="seed prompt", meta={})

    # First evaluation on shard 0 (30%)
    result_shard0 = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=3,
    )

    # Record result on first shard
    decision = scheduler.record(seed, result_shard0, "quality")
    print(f"Shard 0 decision: {decision}")
    assert decision == "promoted", f"Expected seed to be promoted from shard 0, got {decision}"

    # Check current shard index after promotion
    current_idx = scheduler.current_shard_index(seed)
    print(f"Current shard index after first promotion: {current_idx}")
    assert current_idx == 1, f"Expected shard index 1, got {current_idx}"

    # Get promoted candidates
    promotions = scheduler.promote_ready()
    print(f"Promotions after shard 0: {len(promotions)}")
    assert len(promotions) == 1, f"Expected 1 promotion, got {len(promotions)}"
    promoted = promotions[0]

    # Now evaluate on second shard (100%)
    # IMPORTANT: The promoted candidate should carry parent_objectives in its metadata
    result_shard1 = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=10,
    )

    decision2 = scheduler.record(promoted, result_shard1, "quality")
    print(f"Shard 1 decision: {decision2}")
    assert decision2 == "completed", f"Expected 'completed' at final rung, got {decision2}"


def test_mutation_child_with_parent_objectives():
    """Test that a mutation child with parent_objectives progresses correctly."""
    config = SchedulerConfig(
        shards=[0.3, 1.0],
        eps_improve=0.0,
        quantile=0.6,
        enable_convergence=False,
        lineage_patience=0,
        lineage_min_improve=0.01,
    )
    scheduler = BudgetedScheduler(config)

    # First, evaluate a parent to establish baseline
    parent = Candidate(
        text="parent prompt",
        meta={"_sched_key": "parent_key"}
    )

    parent_result = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=3,
    )

    decision = scheduler.record(parent, parent_result, "quality")
    assert decision == "promoted"
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1

    # Now create a child mutation with parent_objectives metadata
    child = Candidate(
        text="child prompt",
        meta={
            "_sched_key": "child_key",
            "parent": "parent_key",
            "parent_objectives": {"quality": 1.0, "neg_cost": -0.1}
        }
    )

    # Child also gets 100% on first shard
    child_result = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=3,
    )

    # Record child result - should promote based on parent comparison
    decision = scheduler.record(child, child_result, "quality")
    print(f"Child shard 0 decision: {decision}")
    print(f"Child current shard index: {scheduler.current_shard_index(child)}")

    # With eps_improve=0.0 and same score as parent, should promote
    assert decision == "promoted", f"Expected child to be promoted, got {decision}"

    promotions = scheduler.promote_ready()
    print(f"Promotions after child eval: {len(promotions)}")
    assert len(promotions) == 1

    promoted_child = promotions[0]

    # Evaluate promoted child on final shard
    child_result_final = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=10,
    )

    decision2 = scheduler.record(promoted_child, child_result_final, "quality")
    print(f"Child shard 1 decision: {decision2}")
    assert decision2 == "completed", f"Expected 'completed' at final rung, got {decision2}"


def test_multiple_children_from_same_parent():
    """Test that multiple mutations from the same parent all progress correctly."""
    config = SchedulerConfig(
        shards=[0.3, 1.0],
        eps_improve=0.0,
        quantile=0.6,
        enable_convergence=False,
        lineage_patience=0,
        lineage_min_improve=0.01,
    )
    scheduler = BudgetedScheduler(config)

    # Evaluate parent
    parent = Candidate(
        text="parent prompt",
        meta={"_sched_key": "parent_key"}
    )

    parent_result = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=3,
    )

    scheduler.record(parent, parent_result, "quality")
    scheduler.promote_ready()  # Clear promotions

    # Create 20 children
    for i in range(20):
        child = Candidate(
            text=f"child prompt {i}",
            meta={
                "_sched_key": f"child_key_{i}",
                "parent": "parent_key",
                "parent_objectives": {"quality": 1.0, "neg_cost": -0.1}
            }
        )

        child_result = EvalResult(
            objectives={"quality": 1.0, "neg_cost": -0.1},
            traces=[],
            n_examples=3,
        )

        decision = scheduler.record(child, child_result, "quality")
        print(f"Child {i} decision: {decision}")
        assert decision == "promoted", f"Child {i} should be promoted"

    # All should be promoted
    promotions = scheduler.promote_ready()
    assert len(promotions) == 20, f"Expected 20 promotions, got {len(promotions)}"


if __name__ == "__main__":
    test_seed_candidate_promotion_through_rungs()
    test_mutation_child_with_parent_objectives()
    test_multiple_children_from_same_parent()
    print("All tests passed!")
