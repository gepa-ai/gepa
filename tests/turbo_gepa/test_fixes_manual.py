#!/usr/bin/env python3
"""
Manual test runner to verify our fixes work.
Run with: python tests/turbo_gepa/test_fixes_manual.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.archive import Archive
from turbo_gepa.config import Config
import asyncio


def test_promoted_candidates_stay_in_rung():
    """Test Fix #1: Promoted candidates should remain in rung."""
    print("\n" + "=" * 80)
    print("TEST 1: Promoted candidates stay in rung for threshold calculation")
    print("=" * 80)

    config = SchedulerConfig(shards=[0.05, 0.20, 1.0], eps_improve=0.01, quantile=0.6)
    scheduler = BudgetedScheduler(config)

    # Evaluate 5 candidates
    for i, score in enumerate([0.90, 0.85, 0.80, 0.75, 0.70]):
        cand = Candidate(text=f"Prompt {i}", meta={})
        result = EvalResult(objectives={"quality": score}, traces=[], n_examples=1, shard_fraction=0.05)
        decision = scheduler.record(cand, result, "quality")
        print(f"  Candidate {i} ({score:.0%}): {decision}")

    rung_size = len(scheduler.rungs[0].results)
    print(f"\nRung 0 population: {rung_size} (expected: 5)")
    assert rung_size == 5, f"FAIL: Expected 5, got {rung_size}"
    print("✅ PASS: All candidates stay in rung")


def test_parent_based_promotion():
    """Test Fix #2: Children beating parent should bypass quantile."""
    print("\n" + "=" * 80)
    print("TEST 2: Parent-based promotion bypasses quantile check")
    print("=" * 80)

    config = SchedulerConfig(shards=[0.05, 0.20, 1.0], eps_improve=0.01, quantile=0.6)
    scheduler = BudgetedScheduler(config)

    # Parent
    parent = Candidate(text="Parent", meta={})
    parent_result = EvalResult(objectives={"quality": 0.80}, traces=[], n_examples=1, shard_fraction=0.05)
    scheduler.record(parent, parent_result, "quality")
    print(f"  Parent: 80% quality → promoted")

    # Children
    test_cases = [
        (0.85, "promoted", "beats parent+eps"),
        (0.82, "promoted", "beats parent+eps"),
        (0.79, "pruned", "below parent+eps"),
    ]

    for score, expected, reason in test_cases:
        child = Candidate(text=f"Child {score}", meta={"parent": parent.fingerprint, "parent_score": 0.80})
        result = EvalResult(objectives={"quality": score}, traces=[], n_examples=1, shard_fraction=0.05)
        decision = scheduler.record(child, result, "quality")
        status = "✅" if decision == expected else "❌"
        print(f"  {status} Child ({score:.0%}): {decision} (expected: {expected}, {reason})")
        assert decision == expected, f"FAIL: {score} should {expected}, got {decision}"

    print("✅ PASS: Parent-based promotion works")


async def test_mutation_eligibility():
    """Test Fix #3: Seeds not eligible until shard 1."""
    print("\n" + "=" * 80)
    print("TEST 3: Mutation eligibility based on shard")
    print("=" * 80)

    config = Config(shards=(0.05, 0.20, 1.0))
    archive = Archive(bins_length=8, bins_bullets=6)

    seed = Candidate(text="Test seed", meta={"source": "seed"})

    # Shard 0 result
    result_shard0 = EvalResult(objectives={"quality": 1.0}, traces=[{"quality": 1.0}], n_examples=1, shard_fraction=0.05)
    await archive.insert(seed, result_shard0)

    min_shard = config.shards[1]
    entries = archive.pareto_entries()
    eligible = [e for e in entries if (e.result.shard_fraction or 0.0) >= min_shard]
    print(f"  After shard 0 (5%): eligible={len(eligible)} (expected: 0)")
    assert len(eligible) == 0, f"FAIL: Should not be eligible on shard 0"
    print("  ✅ Not eligible for mutation")

    # Shard 1 result
    result_shard1 = EvalResult(objectives={"quality": 0.70}, traces=[{"quality": 1.0}, {"quality": 0.0}], n_examples=2, shard_fraction=0.20)
    await archive.insert(seed, result_shard1)

    entries = archive.pareto_entries()
    eligible = [e for e in entries if (e.result.shard_fraction or 0.0) >= min_shard]
    print(f"  After shard 1 (20%): eligible={len(eligible)} (expected: 1)")
    assert len(eligible) == 1, f"FAIL: Should be eligible on shard 1"
    print("  ✅ NOW eligible for mutation")

    print("✅ PASS: Mutation eligibility filter works")


def test_seed_promotion():
    """Test: Seeds get promoted and re-queued."""
    print("\n" + "=" * 80)
    print("TEST 4: Seed promotion and re-queuing")
    print("=" * 80)

    config = SchedulerConfig(shards=[0.3, 1.0], eps_improve=0.01, quantile=0.6)
    scheduler = BudgetedScheduler(config)

    seed = Candidate(text="Seed prompt", meta={"source": "seed"})
    result = EvalResult(objectives={"quality": 0.67}, traces=[], n_examples=1, shard_fraction=0.3)

    decision = scheduler.record(seed, result, "quality")
    print(f"  Seed decision: {decision} (expected: promoted)")
    assert decision == "promoted", f"FAIL: Seed should promote"

    new_rung = scheduler.current_shard_index(seed)
    print(f"  Seed rung after promotion: {new_rung} (expected: 1)")
    assert new_rung == 1, f"FAIL: Should be at rung 1"

    promotions = scheduler.promote_ready()
    print(f"  Promotions available: {len(promotions)} (expected: 1)")
    assert len(promotions) == 1, f"FAIL: Should have 1 promotion"

    print("✅ PASS: Seed promotion works")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TURBOGEPA FIXES - MANUAL TEST SUITE")
    print("=" * 80)

    try:
        test_promoted_candidates_stay_in_rung()
        test_parent_based_promotion()
        asyncio.run(test_mutation_eligibility())
        test_seed_promotion()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
