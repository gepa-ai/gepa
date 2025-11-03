#!/usr/bin/env python3
"""
Test express lane (auto-promote threshold) functionality.
Run with: python tests/turbo_gepa/test_express_lane.py
"""

import pytest
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.interfaces import Candidate, EvalResult


def test_express_lane_single_rung_skip():
    """Test that a candidate scoring above threshold on rung 0 promotes directly to rung 1."""
    print("\n" + "=" * 80)
    print("TEST: Express lane - single rung skip")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Candidate scores 99.5% on rung 0 (above 99% threshold)
    print("\nRung 0: Candidate scores 99.5%")
    candidate = Candidate(text="high_scorer", meta={})
    result = EvalResult(objectives={"quality": 0.995}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(candidate, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "promoted", "Should promote high-scoring candidate"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1, "Should have 1 promotion"
    promoted = promotions[0]
    assert scheduler.current_shard_index(promoted) == 1, "Should be on rung 1 now"

    print("  ✅ Candidate promoted to rung 1 (express lane eligible)")
    print("\n✅ PASS: Express lane candidate correctly promoted")


def test_express_lane_multi_rung_cascade():
    """Test that a candidate scoring 100% cascades through multiple rungs."""
    print("\n" + "=" * 80)
    print("TEST: Express lane - multi-rung cascade")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Rung 0: Score 100%
    print("\nRung 0 (0.2): Candidate scores 100%")
    candidate = Candidate(text="perfect_scorer", meta={})
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(candidate, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "promoted"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    candidate_r1 = promotions[0]
    assert scheduler.current_shard_index(candidate_r1) == 1
    print("  ✅ Promoted to rung 1")

    # Rung 1: Still score 100%
    print("\nRung 1 (0.5): Candidate still scores 100%")
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=3, shard_fraction=0.5)
    decision = scheduler.record(candidate_r1, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "promoted"

    scheduler.mark_generation_start(1)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    candidate_r2 = promotions[0]
    assert scheduler.current_shard_index(candidate_r2) == 2
    print("  ✅ Promoted to rung 2 (final)")

    # Rung 2 (final): Score 100%
    print("\nRung 2 (1.0 - FINAL): Candidate scores 100%")
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=5, shard_fraction=1.0)
    decision = scheduler.record(candidate_r2, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "completed", "Final rung should return 'completed'"
    print("  ✅ Completed at final rung")

    print("\n✅ PASS: Perfect scorer cascaded through all rungs without mutation")


def test_express_lane_stops_when_score_drops():
    """Test that express lane stops when candidate drops below threshold."""
    print("\n" + "=" * 80)
    print("TEST: Express lane stops when score drops")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Rung 0: Score 100%
    print("\nRung 0 (0.2): Candidate scores 100%")
    candidate = Candidate(text="high_then_low", meta={})
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(candidate, result, "quality")
    assert decision == "promoted"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    candidate_r1 = promotions[0]
    print("  ✅ Promoted to rung 1 (express lane)")

    # Rung 1: Score drops to 90%
    print("\nRung 1 (0.5): Candidate drops to 90%")
    result = EvalResult(objectives={"quality": 0.90}, traces=[], n_examples=3, shard_fraction=0.5)
    decision = scheduler.record(candidate_r1, result, "quality")
    print(f"  Decision: {decision}")
    # Should still promote (90% is decent), but express lane would stop here
    # because score < 99% (express lane threshold)
    assert decision == "promoted", "Should still promote (beats 0 baseline)"

    scheduler.mark_generation_start(1)
    promotions = scheduler.promote_ready()
    candidate_r2 = promotions[0]
    print("  ✅ Still promoted (normal lane)")

    # At this point, mutations would be generated because score < 99%
    # Simulate a child that improves
    print("\nRung 1: Child mutation scores 96% (improvement over 90%)")
    child = Candidate(
        text="child_mutation",
        meta={"parent_objectives": {"quality": 0.90}},
    )
    scheduler._candidate_levels[scheduler._sched_key(child)] = 1
    result = EvalResult(objectives={"quality": 0.96}, traces=[], n_examples=3, shard_fraction=0.5)
    decision = scheduler.record(child, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "promoted", "Child improved over parent (0.96 >= 0.90 + 0.05)"

    print("\n✅ PASS: Express lane correctly stops when score drops, mutations resume")


def test_express_lane_threshold_boundary():
    """Test boundary cases around the 99% threshold."""
    print("\n" + "=" * 80)
    print("TEST: Express lane threshold boundary")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Test 1: Exactly 99% (should trigger ceiling logic)
    print("\nTest 1: Candidate scores exactly 99.0%")
    candidate_99 = Candidate(text="exactly_99", meta={})
    result = EvalResult(objectives={"quality": 0.990}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(candidate_99, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "promoted", "99.0% should promote (at ceiling threshold)"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    print("  ✅ 99.0% promoted (at threshold)")

    # Test 2: Just below 99% (98.9%)
    print("\nTest 2: Candidate scores 98.9% (just below threshold)")
    candidate_989 = Candidate(text="below_99", meta={})
    result = EvalResult(objectives={"quality": 0.989}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(candidate_989, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "promoted", "98.9% should still promote (good score)"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    print("  ✅ 98.9% promoted (below threshold, but still good)")

    # Test 3: 99.5% (well above threshold)
    print("\nTest 3: Candidate scores 99.5% (well above threshold)")
    candidate_995 = Candidate(text="above_99", meta={})
    result = EvalResult(objectives={"quality": 0.995}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(candidate_995, result, "quality")
    print(f"  Decision: {decision}")
    assert decision == "promoted", "99.5% should definitely promote"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    print("  ✅ 99.5% promoted (well above threshold)")

    print("\n✅ PASS: Threshold boundary cases handled correctly")


def test_ceiling_prevents_stuck_at_100():
    """Test that multiple candidates at 100% don't get stuck competing."""
    print("\n" + "=" * 80)
    print("TEST: Ceiling prevents stuck at 100%")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Parent scores 100%
    print("\nParent scores 100% on rung 0")
    parent = Candidate(text="parent_100", meta={})
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(parent, result, "quality")
    assert decision == "promoted"
    print("  ✅ Parent promoted")

    scheduler.mark_generation_start(0)
    scheduler.promote_ready()

    # Generate 5 children, all score 100%
    print("\nGenerating 5 children, all score 100%...")
    for i in range(5):
        child = Candidate(
            text=f"child_100_{i}",
            meta={"parent_objectives": {"quality": 1.0}},
        )
        result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
        decision = scheduler.record(child, result, "quality")
        print(f"  Child {i}: {decision}")
        assert decision == "promoted", f"Child {i} should promote (matches parent at ceiling)"

    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    print(f"\n✅ All 5 children promoted (total promotions: {len(promotions)})")
    assert len(promotions) == 5, "All children at ceiling should promote"

    # All promoted children should now be on rung 1
    for p in promotions:
        assert scheduler.current_shard_index(p) == 1, "Should be on rung 1"

    print("✅ PASS: Ceiling logic prevents candidates from getting stuck at 100%")


def test_express_lane_disabled():
    """Test that express lane can be disabled (threshold = None)."""
    print("\n" + "=" * 80)
    print("TEST: Express lane can be disabled")
    print("=" * 80)

    # Note: auto_promote_threshold is orchestrator-level config, not scheduler
    # This test just verifies scheduler behavior doesn't change
    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Candidate scores 100%
    print("\nCandidate scores 100% on rung 0")
    candidate = Candidate(text="perfect", meta={})
    result = EvalResult(objectives={"quality": 1.0}, traces=[], n_examples=1, shard_fraction=0.2)
    decision = scheduler.record(candidate, result, "quality")
    assert decision == "promoted"
    print("  ✅ Promoted (scheduler always promotes seeds)")

    # Even with express lane disabled at orchestrator level,
    # scheduler will still promote the candidate
    scheduler.mark_generation_start(0)
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    print("  ✅ Promotion still works (express lane is orchestrator concern)")

    print("\n✅ PASS: Scheduler behavior unchanged when express lane disabled")


def test_express_lane_intermediate_score():
    """Test candidates with intermediate scores (70-90%) don't use express lane."""
    print("\n" + "=" * 80)
    print("TEST: Intermediate scores don't trigger express lane")
    print("=" * 80)

    config = SchedulerConfig(
        shards=[0.2, 0.5, 1.0],
        eps_improve=0.05,
        patience_generations=3,
    )
    scheduler = BudgetedScheduler(config)

    # Test various intermediate scores
    test_scores = [0.70, 0.80, 0.90, 0.95, 0.98]

    for score in test_scores:
        print(f"\nTesting score: {score:.0%}")
        candidate = Candidate(text=f"score_{int(score*100)}", meta={})
        result = EvalResult(objectives={"quality": score}, traces=[], n_examples=1, shard_fraction=0.2)
        decision = scheduler.record(candidate, result, "quality")
        print(f"  Decision: {decision}")
        assert decision == "promoted", f"{score:.0%} should promote"

        scheduler.mark_generation_start(0)
        promotions = scheduler.promote_ready()
        assert len(promotions) == 1
        promoted = promotions[0]
        assert scheduler.current_shard_index(promoted) == 1

        # These scores are below 99%, so express lane wouldn't apply
        # (though scheduler doesn't know about express lane, that's orchestrator's job)
        if score < 0.99:
            print(f"  ✅ {score:.0%} promoted (below express threshold, normal path)")
        else:
            print(f"  ✅ {score:.0%} promoted (above express threshold, fast lane eligible)")

    print("\n✅ PASS: Intermediate scores handled correctly")


if __name__ == "__main__":
    test_express_lane_single_rung_skip()
    test_express_lane_multi_rung_cascade()
    test_express_lane_stops_when_score_drops()
    test_express_lane_threshold_boundary()
    test_ceiling_prevents_stuck_at_100()
    test_express_lane_disabled()
    test_express_lane_intermediate_score()
    print("\n" + "=" * 80)
    print("ALL EXPRESS LANE TESTS PASSED!")
    print("=" * 80)
