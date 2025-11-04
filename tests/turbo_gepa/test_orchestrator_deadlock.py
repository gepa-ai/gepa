#!/usr/bin/env python3
"""
Test for orchestrator deadlock issues.
Run with: python tests/turbo_gepa/test_orchestrator_deadlock.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.interfaces import Candidate


def test_seed_promotion_and_requeue():
    """Debug: Check if seed gets promoted AND re-queued properly."""
    print("\n" + "=" * 80)
    print("DEADLOCK DEBUG: Seed promotion + queue state")
    print("=" * 80)

    scheduler = BudgetedScheduler(config)

    # Create seed
    seed = Candidate(text="You are a helpful assistant.", meta={"source": "seed"})

    print(f"1. Initial state:")
    print(f"   Seed rung: {scheduler.current_shard_index(seed)}")
    print(f"   Pending promotions: {len(scheduler._pending_promotions)}")

    # Simulate evaluation on shard 0
    from turbo_gepa.interfaces import EvalResult
    result = EvalResult(objectives={"quality": 0.67}, traces=[], n_examples=1, shard_fraction=0.3)

    print(f"\n2. Recording result...")
    decision = scheduler.record(seed, result, "quality")
    print(f"   Decision: {decision}")
    print(f"   Seed rung after record: {scheduler.current_shard_index(seed)}")
    print(f"   Pending promotions: {len(scheduler._pending_promotions)}")

    # Get promotions
    print(f"\n3. Getting promotions...")
    promotions = scheduler.promote_ready()
    print(f"   Promotions returned: {len(promotions)}")
    if promotions:
        for p in promotions:
            print(f"     - {p.text[:30]}... at rung {scheduler.current_shard_index(p)}")

    # Check if promotion list is cleared
    print(f"   Pending promotions after promote_ready(): {len(scheduler._pending_promotions)}")

    print("\n4. Analysis:")
    if decision == "promoted" and len(promotions) == 1:
        print("   ✅ Seed was promoted and returned by promote_ready()")
        print("   ✅ Orchestrator should re-queue this for next shard")
    else:
        print(f"   ❌ Something went wrong:")
        print(f"      - Decision: {decision} (expected: promoted)")
        print(f"      - Promotions: {len(promotions)} (expected: 1)")


def test_minimum_shard_calculation():
    """Debug: Check min_shard calculation for different configs."""
    print("\n" + "=" * 80)
    print("DEADLOCK DEBUG: Minimum shard for mutation")
    print("=" * 80)

    from turbo_gepa.config import Config

    test_cases = [
        ((0.3, 1.0), 1.0),
        ((0.05, 0.20, 1.0), 0.20),
        ((1.0,), 1.0),
    ]

    for shards, expected_min in test_cases:
        config = Config(shards=shards)
        min_shard = config.shards[0] if len(config.shards) == 1 else config.shards[1]
        status = "✅" if min_shard == expected_min else "❌"
        print(f"  {status} shards={shards} → min_shard={min_shard} (expected: {expected_min})")


def test_mutation_spawn_conditions():
    """Debug: When should mutations spawn?"""
    print("\n" + "=" * 80)
    print("DEADLOCK DEBUG: Mutation spawn conditions")
    print("=" * 80)

    # Simulate orchestrator state
    queue_size = 1  # Promoted seed waiting
    buffer_size = 0
    mutation_buffer_min = 2
    evaluations_run = 1
    max_evaluations = None

    should_spawn = (queue_size + buffer_size) < mutation_buffer_min and (
        max_evaluations is None or evaluations_run < max_evaluations
    )

    print(f"  queue_size: {queue_size}")
    print(f"  buffer_size: {buffer_size}")
    print(f"  mutation_buffer_min: {mutation_buffer_min}")
    print(f"  evaluations_run: {evaluations_run}")
    print(f"  max_evaluations: {max_evaluations}")
    print(f"\n  should_spawn_mutations: {should_spawn}")

    if should_spawn:
        print("  ✅ Mutations SHOULD spawn (queue + buffer < min)")
        print("  ⚠️  But if no parents are eligible, spawn_mutations returns []")
        print("  ⚠️  This could cause deadlock if seed not launched from queue!")
    else:
        print("  ❌ Mutations will NOT spawn")


if __name__ == "__main__":
    try:
        test_seed_promotion_and_requeue()
        test_minimum_shard_calculation()
        test_mutation_spawn_conditions()

        print("\n" + "=" * 80)
        print("DEADLOCK DIAGNOSIS COMPLETE")
        print("=" * 80)
        print("\nPotential deadlock scenario:")
        print("1. Seed evaluated on shard 0 → promoted → re-queued for shard 1")
        print("2. Mutation spawn triggered (queue=1 < buffer_min=2)")
        print("3. No parents eligible yet (seed hasn't reached shard 1)")
        print("4. spawn_mutations returns [] (no mutations generated)")
        print("5. Seed sits in queue, not being launched")
        print("6. System hangs waiting for... nothing!")
        print("\nNext step: Check why seed isn't being launched from queue")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
