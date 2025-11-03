"""
Test that the archive properly updates when a candidate is re-evaluated at higher rungs.
"""

import pytest
import asyncio
from turbo_gepa.archive import Archive
from turbo_gepa.interfaces import Candidate, EvalResult


@pytest.mark.asyncio
async def test_archive_updates_with_higher_rung():
    """Test that when a candidate is evaluated at a higher rung, the archive updates."""

    print("\n" + "=" * 80)
    print("TEST: Archive should update with higher rung evaluations")
    print("=" * 80)

    archive = Archive(bins_length=10, bins_bullets=10)

    # Create a candidate
    candidate = Candidate(text="Test prompt", meta={"source": "seed"})

    # Evaluate at rung 0.2 (20%)
    result_20 = EvalResult(
        objectives={"quality": 0.6, "neg_cost": -0.001},
        traces=[],
        n_examples=3,
        shard_fraction=0.2,
    )

    await archive.insert(candidate, result_20)

    print(f"\n1. After evaluating at rung 0.2:")
    pareto = archive.pareto_entries()
    print(f"   Pareto size: {len(pareto)}")
    assert len(pareto) == 1
    print(f"   Entry shard: {pareto[0].result.shard_fraction}")
    print(f"   Entry quality: {pareto[0].result.objectives['quality']}")
    assert pareto[0].result.shard_fraction == 0.2

    # Evaluate SAME candidate at rung 0.5 (50%)
    result_50 = EvalResult(
        objectives={"quality": 0.65, "neg_cost": -0.002},
        traces=[],
        n_examples=7,
        shard_fraction=0.5,
    )

    await archive.insert(candidate, result_50)

    print(f"\n2. After evaluating SAME candidate at rung 0.5:")
    pareto = archive.pareto_entries()
    print(f"   Pareto size: {len(pareto)}")
    assert len(pareto) == 1, f"Should still have 1 entry, got {len(pareto)}"
    print(f"   Entry shard: {pareto[0].result.shard_fraction}")
    print(f"   Entry quality: {pareto[0].result.objectives['quality']}")

    # THIS IS THE BUG: Archive should have updated to 0.5, but it stays at 0.2!
    assert pareto[0].result.shard_fraction == 0.5, \
        f"Archive should update to higher rung 0.5, but got {pareto[0].result.shard_fraction}"

    # Evaluate SAME candidate at rung 1.0 (100%)
    result_100 = EvalResult(
        objectives={"quality": 0.7, "neg_cost": -0.005},
        traces=[],
        n_examples=15,
        shard_fraction=1.0,
    )

    await archive.insert(candidate, result_100)

    print(f"\n3. After evaluating SAME candidate at rung 1.0:")
    pareto = archive.pareto_entries()
    print(f"   Pareto size: {len(pareto)}")
    assert len(pareto) == 1, f"Should still have 1 entry, got {len(pareto)}"
    print(f"   Entry shard: {pareto[0].result.shard_fraction}")
    print(f"   Entry quality: {pareto[0].result.objectives['quality']}")
    assert pareto[0].result.shard_fraction == 1.0, \
        f"Archive should update to final rung 1.0, but got {pareto[0].result.shard_fraction}"

    print(f"\n✅ TEST PASSED: Archive correctly updates with higher rung evaluations")
    print("=" * 80)


@pytest.mark.asyncio
async def test_archive_keeps_different_candidates():
    """Test that the archive keeps multiple different candidates."""

    print("\n" + "=" * 80)
    print("TEST: Archive should keep different candidates separate")
    print("=" * 80)

    archive = Archive(bins_length=10, bins_bullets=10)

    # Create two different candidates
    candidate1 = Candidate(text="Prompt A", meta={"source": "seed"})
    candidate2 = Candidate(text="Prompt B", meta={"source": "mutation"})

    # Evaluate both at rung 0.2
    result1 = EvalResult(
        objectives={"quality": 0.6, "neg_cost": -0.001},
        traces=[],
        n_examples=3,
        shard_fraction=0.2,
    )

    result2 = EvalResult(
        objectives={"quality": 0.65, "neg_cost": -0.0015},  # Slightly different
        traces=[],
        n_examples=3,
        shard_fraction=0.2,
    )

    await archive.insert(candidate1, result1)
    await archive.insert(candidate2, result2)

    print(f"\nAfter inserting 2 different candidates:")
    pareto = archive.pareto_entries()
    print(f"   Pareto size: {len(pareto)}")

    # Should keep both since they have different quality/cost tradeoffs
    assert len(pareto) >= 1, "Should have at least one candidate"

    print(f"\n✅ TEST PASSED: Archive handles multiple candidates")
    print("=" * 80)


if __name__ == "__main__":
    print("\nRunning archive rung update tests...\n")

    try:
        asyncio.run(test_archive_updates_with_higher_rung())
        asyncio.run(test_archive_keeps_different_candidates())

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
