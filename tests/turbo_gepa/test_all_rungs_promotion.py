"""
Test that verifies seeds go through ALL rungs (0.2 → 0.5 → 1.0) sequentially.
"""

import pytest
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.config import Config


@pytest.mark.asyncio
async def test_seed_visits_all_three_rungs():
    """Verify that a seed is promoted through all 3 rungs sequentially."""
    print("\n" + "=" * 80)
    print("TEST: Seed visits ALL rungs (0.2 → 0.5 → 1.0)")
    print("=" * 80)

    # Create scheduler with 3 rungs
    config = SchedulerConfig(
        shards=(0.2, 0.5, 1.0),
        eps_improve=0.01,
    )
    scheduler = BudgetedScheduler(config)

    # Create seed
    seed = Candidate(text="You are a helpful assistant.", meta={"source": "seed"})

    print(f"\n1. Initial state:")
    print(f"   Rung index: {scheduler.current_shard_index(seed)} (expected: 0)")
    print(f"   Shard fraction: {scheduler.current_shard_fraction(seed)} (expected: 0.2)")
    assert scheduler.current_shard_index(seed) == 0
    assert scheduler.current_shard_fraction(seed) == 0.2

    # ===== RUNG 0: Evaluate at 0.2 =====
    print(f"\n2. Evaluate at rung 0 (0.2):")
    result_0 = EvalResult(
        objectives={"quality": 0.8},
        traces=[],
        n_examples=3,
        shard_fraction=0.2,
    )
    decision_0 = scheduler.record(seed, result_0, "quality")
    print(f"   Decision: {decision_0} (expected: promoted)")
    print(f"   Rung after record: {scheduler.current_shard_index(seed)} (expected: 1)")
    assert decision_0 == "promoted", "Seed should be promoted from rung 0"
    assert scheduler.current_shard_index(seed) == 1, "Seed should be at rung 1 after promotion"

    promotions = scheduler.promote_ready()
    print(f"   Promotions: {len(promotions)} (expected: 1)")
    assert len(promotions) == 1
    promoted_0 = promotions[0]
    assert promoted_0.fingerprint == seed.fingerprint
    assert scheduler.current_shard_index(promoted_0) == 1

    # ===== RUNG 1: Evaluate at 0.5 =====
    print(f"\n3. Evaluate at rung 1 (0.5):")
    print(f"   Current rung: {scheduler.current_shard_index(seed)} (expected: 1)")
    print(f"   Current shard: {scheduler.current_shard_fraction(seed)} (expected: 0.5)")
    assert scheduler.current_shard_fraction(seed) == 0.5, f"Expected shard 0.5, got {scheduler.current_shard_fraction(seed)}"

    result_1 = EvalResult(
        objectives={"quality": 0.8},
        traces=[],
        n_examples=7,
        shard_fraction=0.5,
    )
    decision_1 = scheduler.record(seed, result_1, "quality")
    print(f"   Decision: {decision_1} (expected: promoted)")
    print(f"   Rung after record: {scheduler.current_shard_index(seed)} (expected: 2)")
    assert decision_1 == "promoted", "Seed should be promoted from rung 1"
    assert scheduler.current_shard_index(seed) == 2, "Seed should be at rung 2 after promotion"

    promotions = scheduler.promote_ready()
    print(f"   Promotions: {len(promotions)} (expected: 1)")
    assert len(promotions) == 1
    promoted_1 = promotions[0]
    assert promoted_1.fingerprint == seed.fingerprint
    assert scheduler.current_shard_index(promoted_1) == 2

    # ===== RUNG 2: Evaluate at 1.0 (final) =====
    print(f"\n4. Evaluate at rung 2 (1.0 - final):")
    print(f"   Current rung: {scheduler.current_shard_index(seed)} (expected: 2)")
    print(f"   Current shard: {scheduler.current_shard_fraction(seed)} (expected: 1.0)")
    assert scheduler.current_shard_fraction(seed) == 1.0, f"Expected shard 1.0, got {scheduler.current_shard_fraction(seed)}"

    result_2 = EvalResult(
        objectives={"quality": 0.8},
        traces=[],
        n_examples=15,
        shard_fraction=1.0,
    )
    decision_2 = scheduler.record(seed, result_2, "quality")
    print(f"   Decision: {decision_2} (expected: completed)")
    assert decision_2 == "completed", "Seed should be completed at final rung"

    promotions = scheduler.promote_ready()
    print(f"   Promotions: {len(promotions)} (expected: 0 - no more rungs)")
    assert len(promotions) == 0, "No promotions at final rung"

    print(f"\n✅ TEST PASSED!")
    print(f"   Seed was evaluated at:")
    print(f"   - Rung 0 (0.2) → promoted")
    print(f"   - Rung 1 (0.5) → promoted")
    print(f"   - Rung 2 (1.0) → completed")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_seed_visits_all_three_rungs())
