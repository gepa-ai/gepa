#!/usr/bin/env python3
"""
Test generation-based convergence detection.
"""

import pytest
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.interfaces import Candidate, EvalResult


def test_mid_rung_convergence_forces_promotion():
    """Test that mid-rung stagnation forces promotion of best candidate."""
    print("\n" + "=" * 80)
    print("TEST: Mid-rung convergence forces promotion")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,  # Converge after 3 generations without improvement
    )
    scheduler = BudgetedScheduler(config)

    # Generation 1: Seed with score 0.50
    seed = Candidate(text="seed", meta={})
    result = EvalResult(objectives={"quality": 0.50}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(seed, result, "quality")
    print(f"Gen 1: Seed (0.50) → {decision}")
    assert decision == "promoted", "Seed should promote"

    # Mark start of generation 2 (completes generation 1 which had improvement)
    scheduler.mark_generation_start(rung_idx=0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1, "Should have promoted seed"

    # Generation 2: 3 children, none improve (all 0.45 < 0.50 + 0.05)
    print("\nGen 2: 3 children, none improve...")
    for i in range(3):
        child = Candidate(
            text=f"child_gen2_{i}",
            meta={"parent_objectives": {"quality": 0.50}},
        )
        result = EvalResult(objectives={"quality": 0.45}, traces=[], n_examples=1, shard_fraction=0.2)
        decision = scheduler.record(child, result, "quality")
        print(f"  Child {i} (0.45) → {decision}")
        assert decision == "pruned"

    # Mark start of generation 3 (completes generation 2 which had NO improvement)
    scheduler.mark_generation_start(rung_idx=0)
    assert scheduler._rung_generations[0][0] == 1, "Should have 1 stagnant generation"

    # Generation 3: 3 more children, still no improvement
    print("\nGen 3: 3 children, still no improvement...")
    for i in range(3):
        child = Candidate(
            text=f"child_gen3_{i}",
            meta={"parent_objectives": {"quality": 0.50}},
        )
        result = EvalResult(objectives={"quality": 0.46}, traces=[], n_examples=1, shard_fraction=0.2)
        decision = scheduler.record(child, result, "quality")
        print(f"  Child {i} (0.46) → {decision}")
        assert decision == "pruned"

    # Mark start of generation 4 (completes generation 3 which had NO improvement)
    scheduler.mark_generation_start(rung_idx=0)
    assert scheduler._rung_generations[0][0] == 2, "Should have 2 stagnant generations"

    # Generation 4: 3 more children, still no improvement - THIS SHOULD TRIGGER CONVERGENCE
    print("\nGen 4: 3 children, still no improvement (should trigger force promotion)...")
    for i in range(3):
        child = Candidate(
            text=f"child_gen4_{i}",
            meta={"parent_objectives": {"quality": 0.50}},
        )
        result = EvalResult(objectives={"quality": 0.47}, traces=[], n_examples=1, shard_fraction=0.2)
        decision = scheduler.record(child, result, "quality")
        print(f"  Child {i} (0.47) → {decision}")
        assert decision == "pruned"

    # Mark start of generation 5 (completes generation 4 which had NO improvement)
    # This should trigger force promotion!
    print("\n🚀 Marking generation 5 start (should force promote best)...")
    scheduler.mark_generation_start(rung_idx=0)

    # Check that best candidate was force promoted
    promotions = scheduler.promote_ready()
    print(f"✅ Force promoted {len(promotions)} candidates")
    assert len(promotions) == 1, "Should have force promoted best candidate"
    best = promotions[0]
    assert best.text == "seed", "Should have promoted the seed (best on rung 0)"

    # Check that stagnation counter was reset
    assert scheduler._rung_generations[0][0] == 0, "Should have reset counter after force promotion"

    print("\n✅ PASS: Mid-rung convergence correctly forces promotion")


def test_final_rung_convergence_stops_optimization():
    """Test that final rung stagnation sets converged flag."""
    print("\n" + "=" * 80)
    print("TEST: Final rung convergence stops optimization")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.5, 1.0],  # Only 2 rungs
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Get a seed to final rung (rung 1)
    seed = Candidate(text="seed", meta={})

    # First eval on rung 0 (0.5)
    result = EvalResult(objectives={"quality": 0.60}, traces=[], n_examples=5, shard_fraction=0.5)
    decision = scheduler.record(seed, result, "quality")
    assert decision == "promoted"
    scheduler.mark_generation_start(0)
    scheduler.promote_ready()

    # Second eval on rung 1 (1.0) - final rung
    result = EvalResult(objectives={"quality": 0.62}, traces=[], n_examples=10, shard_fraction=1.0)
    decision = scheduler.record(seed, result, "quality")
    print(f"Seed on final rung (0.62) → {decision}")
    assert decision == "completed"

    assert not scheduler.converged, "Should not be converged yet"

    # Now do 3 generations on final rung with no improvement
    for gen in range(1, 4):
        print(f"\nGeneration {gen} on final rung: no improvement...")
        scheduler.mark_generation_start(1)

        for i in range(3):
            child = Candidate(
                text=f"child_final_{gen}_{i}",
                meta={"parent_objectives": {"quality": 0.62}},
            )
            # Pretend we're on final rung
            scheduler._candidate_levels[scheduler._sched_key(child)] = 1

            result = EvalResult(objectives={"quality": 0.60}, traces=[], n_examples=10, shard_fraction=1.0)
            decision = scheduler.record(child, result, "quality")
            assert decision == "completed"  # Final rung always returns "completed"

    # After 3 generations without improvement, mark next generation
    print("\n🛑 Marking generation 4 start (should trigger convergence)...")
    scheduler.mark_generation_start(1)

    print(f"✅ Scheduler converged: {scheduler.converged}")
    assert scheduler.converged, "Should have marked converged after 3 stagnant generations on final rung"

    print("\n✅ PASS: Final rung convergence correctly sets converged flag")


def test_improvement_resets_counter():
    """Test that any improvement resets the stagnation counter."""
    print("\n" + "=" * 80)
    print("TEST: Improvement resets stagnation counter")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Seed
    seed = Candidate(text="seed", meta={})
    result = EvalResult(objectives={"quality": 0.50}, traces=[], n_examples=1, shard_fraction=0.2)
    scheduler.record(seed, result, "quality")
    scheduler.mark_generation_start(0)
    scheduler.promote_ready()

    # Generation 1: No improvement
    print("\nGen 1: No improvement")
    child1 = Candidate(text="child1", meta={"parent_objectives": {"quality": 0.50}})
    result = EvalResult(objectives={"quality": 0.48}, traces=[], n_examples=1, shard_fraction=0.2)
    scheduler.record(child1, result, "quality")
    scheduler.mark_generation_start(0)
    assert scheduler._rung_generations[0][0] == 1, "Should have 1 stagnant gen"

    # Generation 2: No improvement
    print("Gen 2: No improvement")
    child2 = Candidate(text="child2", meta={"parent_objectives": {"quality": 0.50}})
    result = EvalResult(objectives={"quality": 0.49}, traces=[], n_examples=1, shard_fraction=0.2)
    scheduler.record(child2, result, "quality")
    scheduler.mark_generation_start(0)
    assert scheduler._rung_generations[0][0] == 2, "Should have 2 stagnant gens"

    # Generation 3: IMPROVEMENT! (0.56 >= 0.50 + 0.05)
    print("Gen 3: IMPROVEMENT! (should reset counter)")
    child3 = Candidate(text="child3", meta={"parent_objectives": {"quality": 0.50}})
    result = EvalResult(objectives={"quality": 0.56}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(child3, result, "quality")
    assert decision == "promoted", "Should promote on improvement"

    scheduler.mark_generation_start(0)
    print(f"Counter after improvement: {scheduler._rung_generations[0][0]}")
    assert scheduler._rung_generations[0][0] == 0, "Should have reset counter to 0"

    print("\n✅ PASS: Improvement correctly resets stagnation counter")


def test_ceiling_promotion():
    """Test that candidates matching parent at 100% quality are promoted."""
    print("\n" + "=" * 80)
    print("TEST: Ceiling promotion (100% quality)")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Parent achieves 100% on rung 0
    parent = Candidate(text="parent", meta={})
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(parent, result, "quality")
    print(f"Parent (100%) → {decision}")
    assert decision == "promoted", "Parent should promote"

    scheduler.mark_generation_start(0)
    scheduler.promote_ready()

    # Child also achieves 100% - should promote even though no improvement
    print("\nChild matches parent at ceiling (100%)...")
    child = Candidate(
        text="child",
        meta={"parent_objectives": {"quality": 1.0}},
    )
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(child, result, "quality")
    print(f"Child (100%) → {decision}")
    assert decision == "promoted", "Child should promote when matching parent at 100%"

    # Child with 99% should be pruned (doesn't meet eps_improve threshold)
    child2 = Candidate(
        text="child2",
        meta={"parent_objectives": {"quality": 1.0}},
    )
    result = EvalResult(objectives={"quality": 0.99}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(child2, result, "quality")
    print(f"Child (99%) → {decision}")
    assert decision == "pruned", "Child below ceiling should be pruned"

    print("\n✅ PASS: Ceiling promotion works correctly")


def test_ceiling_promotion_all_rungs():
    """Test that ceiling promotion works across all rungs."""
    print("\n" + "=" * 80)
    print("TEST: Ceiling promotion across all rungs")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Rung 0: Parent @ 100%, child @ 100% → should promote to rung 1
    print("\nRung 0 (0.2):")
    parent = Candidate(text="parent_r0", meta={})
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(parent, result, "quality")
    print(f"  Parent (100%) → {decision}")
    assert decision == "promoted"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    parent_r1 = promotions[0]
    assert scheduler.current_shard_index(parent_r1) == 1, "Should be on rung 1"

    # Rung 1: Evaluate parent @ 100%, child @ 100% → should promote to rung 2
    print("\nRung 1 (0.5):")
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=2, shard_fraction=0.5)
    decision = scheduler.record(parent_r1, result, "quality")
    print(f"  Parent (100%) → {decision}")
    assert decision == "promoted"

    scheduler.mark_generation_start(1)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    parent_r2 = promotions[0]
    assert scheduler.current_shard_index(parent_r2) == 2, "Should be on rung 2 (final)"

    # Rung 2 (final): Parent @ 100% → should complete (not promote beyond final)
    print("\nRung 2 (1.0) - FINAL:")
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=5, shard_fraction=1.0)
    decision = scheduler.record(parent_r2, result, "quality")
    print(f"  Parent (100%) → {decision}")
    assert decision == "completed", "Final rung should return 'completed'"

    # Test child on rung 1 also promotes
    print("\nRung 1 child @ 100%:")
    child_r1 = Candidate(text="child_r1", meta={"parent_objectives": {"quality": 1.0}})
    scheduler._candidate_levels[scheduler._sched_key(child_r1)] = 1  # Place on rung 1
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=2, shard_fraction=0.5)
    decision = scheduler.record(child_r1, result, "quality")
    print(f"  Child (100%) → {decision}")
    assert decision == "promoted", "Child at 100% on rung 1 should promote to rung 2"

    print("\n✅ PASS: Ceiling promotion works across all rungs")


if __name__ == "__main__":
    test_mid_rung_convergence_forces_promotion()
    test_final_rung_convergence_stops_optimization()
    test_improvement_resets_counter()
    test_ceiling_promotion()
    test_ceiling_promotion_all_rungs()
    print("\n" + "=" * 80)
    print("ALL CONVERGENCE TESTS PASSED!")
    print("=" * 80)
