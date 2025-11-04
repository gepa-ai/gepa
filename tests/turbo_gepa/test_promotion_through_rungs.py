"""
Test that candidates actually get promoted through all rungs.

This is a focused test to debug why candidates seem stuck at rung 0.
"""

import pytest
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig


def test_seed_promotes_through_all_rungs():
    """Test that a seed candidate gets promoted through all rungs."""

    # Setup scheduler with 3 rungs
    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Create a seed candidate
    seed = Candidate(text="test seed", meta={"source": "seed"})

    # Rung 0: Evaluate seed at 20% shard
    result_0 = EvalResult(
        shard_fraction=0.2,
        objectives={"quality": 0.8, "neg_cost": -0.001},
        traces=[],
        n_examples=3,
    )

    decision_0 = scheduler.record(seed, result_0, "quality")
    print(f"Rung 0 decision: {decision_0}")
    assert decision_0 == "promoted", "Seed should be promoted from rung 0"

    # Check that scheduler has it ready for promotion
    promotions = scheduler.promote_ready()
    print(f"Promotions after rung 0: {len(promotions)}")
    assert len(promotions) == 1, "Should have 1 promotion ready"
    assert promotions[0].fingerprint == seed.fingerprint

    # Rung 1: Evaluate at 50% shard
    current_idx = scheduler.current_shard_index(seed)
    print(f"Current shard index after rung 0 promotion: {current_idx}")
    assert current_idx == 1, "Seed should be at rung 1 now"

    result_1 = EvalResult(
        shard_fraction=0.5,
        objectives={"quality": 0.8, "neg_cost": -0.002},
        traces=[],
        n_examples=7,
    )

    decision_1 = scheduler.record(seed, result_1, "quality")
    print(f"Rung 1 decision: {decision_1}")
    assert decision_1 == "promoted", "Seed should be promoted from rung 1"

    promotions = scheduler.promote_ready()
    print(f"Promotions after rung 1: {len(promotions)}")
    assert len(promotions) == 1, "Should have 1 promotion ready"

    # Rung 2: Evaluate at 100% shard (final rung)
    current_idx = scheduler.current_shard_index(seed)
    print(f"Current shard index after rung 1 promotion: {current_idx}")
    assert current_idx == 2, "Seed should be at rung 2 (final) now"

    result_2 = EvalResult(
        shard_fraction=1.0,
        objectives={"quality": 0.8, "neg_cost": -0.005},
        traces=[],
        n_examples=15,
    )

    decision_2 = scheduler.record(seed, result_2, "quality")
    print(f"Rung 2 decision: {decision_2}")
    assert decision_2 == "completed", "Seed should be completed at final rung"

    promotions = scheduler.promote_ready()
    print(f"Promotions after rung 2: {len(promotions)}")
    assert len(promotions) == 0, "No more promotions at final rung"


def test_mutation_promotes_if_better_than_parent():
    """Test that a mutation that improves gets promoted."""

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Create parent and child
    parent = Candidate(text="parent", meta={"source": "seed"})
    child = Candidate(
        text="child mutation",
        meta={
            "source": "mutation",
            "parent_score": 0.5,
            "parent_objectives": {"quality": 0.5, "neg_cost": -0.001},
        }
    )

    # Child scores better than parent
    result = EvalResult(
        shard_fraction=0.2,
        objectives={"quality": 0.6, "neg_cost": -0.001},  # 0.6 > 0.5 + 0.01
        traces=[],
        n_examples=3,
    )

    decision = scheduler.record(child, result, "quality")
    print(f"Child decision (0.6 vs parent 0.5): {decision}")
    assert decision == "promoted", "Child with better score should be promoted"

    promotions = scheduler.promote_ready()
    assert len(promotions) == 1


def test_mutation_pruned_if_not_better():
    """Test that a mutation that doesn't improve gets pruned."""

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    child = Candidate(
        text="child mutation",
        meta={
            "source": "mutation",
            "parent_score": 0.5,
            "parent_objectives": {"quality": 0.5, "neg_cost": -0.001},
        }
    )

    # Child scores same or worse than parent
    result = EvalResult(
        shard_fraction=0.2,
        objectives={"quality": 0.5, "neg_cost": -0.001},  # 0.5 < 0.5 + 0.01
        traces=[],
        n_examples=3,
    )

    decision = scheduler.record(child, result, "quality")
    print(f"Child decision (0.5 vs parent 0.5, eps=0.01): {decision}")
    assert decision == "pruned", "Child without improvement should be pruned"

    promotions = scheduler.promote_ready()
    assert len(promotions) == 0, "Pruned candidates should not be promoted"


if __name__ == "__main__":
    print("=" * 80)
    print("Testing seed promotion through all rungs...")
    print("=" * 80)
    test_seed_promotes_through_all_rungs()
    print("\n✅ Seed promotion test passed!\n")

    print("=" * 80)
    print("Testing mutation promotion...")
    print("=" * 80)
    test_mutation_promotes_if_better_than_parent()
    print("\n✅ Mutation promotion test passed!\n")

    print("=" * 80)
    print("Testing mutation pruning...")
    print("=" * 80)
    test_mutation_pruned_if_not_better()
    print("\n✅ Mutation pruning test passed!\n")

    print("=" * 80)
    print("🎉 All promotion tests passed!")
    print("=" * 80)
