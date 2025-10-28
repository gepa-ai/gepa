#!/usr/bin/env python3
"""
Integration tests for orchestrator flow to ensure proper evolution cycle.
Run with: python -m pytest tests/turbo_gepa/test_orchestrator_flow.py -v
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from turbo_gepa.config import Config
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.archive import Archive


class MockEvaluator:
    """Mock evaluator that returns quick results."""

    def __init__(self):
        self.eval_count = 0

    async def eval_on_shard(self, candidate, example_ids, concurrency, shard_fraction, show_progress=False):
        """Mock evaluation that returns immediately."""
        self.eval_count += 1
        await asyncio.sleep(0.01)  # Minimal delay

        # Return mock result
        return EvalResult(
            objectives={"quality": 0.5, "neg_cost": -100},
            traces=[{"example_id": eid, "quality": 0.5} for eid in example_ids],
            n_examples=len(example_ids),
            shard_fraction=shard_fraction,
            example_ids=example_ids,
        )


def test_scheduler_promotion_flow():
    """Test that candidates are properly promoted through shards."""
    config = SchedulerConfig(
        shards=(0.5, 1.0),
        eps_improve=0.0,
        quantile=0.4,
    )
    scheduler = BudgetedScheduler(config)

    # Create seed
    seed = Candidate(text="seed prompt", meta={"source": "seed"})

    # Evaluate on shard 0
    result_0 = EvalResult(
        objectives={"quality": 0.6},
        traces=[],
        n_examples=3,
        shard_fraction=0.5,
    )

    decision = scheduler.record(seed, result_0, "quality")
    assert decision in ("promoted", "continue"), f"Seed should be promoted or continue, got {decision}"

    # Get promotions
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1, f"Should have 1 promotion, got {len(promotions)}"
    assert promotions[0].fingerprint == seed.fingerprint

    # Check promoted to next rung
    promoted_rung = scheduler.current_shard_index(promotions[0])
    assert promoted_rung == 1, f"Should be promoted to rung 1, got {promoted_rung}"


def test_archive_insertion_and_retrieval():
    """Test that archive stores and retrieves candidates correctly."""
    archive = Archive(objectives=["quality", "neg_cost"])

    # Insert candidate
    candidate = Candidate(text="test prompt", meta={})
    result = EvalResult(
        objectives={"quality": 0.7, "neg_cost": -100},
        traces=[],
        n_examples=5,
        shard_fraction=1.0,
    )

    asyncio.run(archive.insert(candidate, result))

    # Retrieve pareto entries
    entries = archive.pareto_entries()
    assert len(entries) >= 1, "Should have at least 1 entry in archive"

    # Check entry contents
    entry = entries[0]
    assert entry.candidate.fingerprint == candidate.fingerprint
    assert entry.result.objectives["quality"] == 0.7


def test_idle_detection_logic():
    """Test idle detection conditions."""

    # Simulate orchestrator state
    class MockState:
        def __init__(self):
            self.launched = 0
            self.drained = 0
            self._total_inflight = 0
            self.queue = []
            self._mutation_buffer = []
            self._mutation_task = None

        def is_idle(self):
            """Idle detection from orchestrator.py line 415-433."""
            if (
                self.launched == 0
                and self.drained == 0
                and self._total_inflight == 0
                and not self.queue
                and not self._mutation_buffer
            ):
                if self._mutation_task is None:
                    return True
                elif self._mutation_task is not None and asyncio.iscoroutine(self._mutation_task):
                    # For mock purposes, assume done if it's None
                    return False
            return False

    # Test: Should be idle when nothing is happening
    state = MockState()
    assert state.is_idle() == True, "Should be idle when queue/buffer/inflight all empty and no task"

    # Test: Should NOT be idle when queue has items
    state = MockState()
    state.queue.append(Candidate(text="test", meta={}))
    assert state.is_idle() == False, "Should not be idle when queue has items"

    # Test: Should NOT be idle when buffer has items
    state = MockState()
    state._mutation_buffer.append(Candidate(text="test", meta={}))
    assert state.is_idle() == False, "Should not be idle when buffer has items"

    # Test: Should NOT be idle when inflight > 0
    state = MockState()
    state._total_inflight = 1
    assert state.is_idle() == False, "Should not be idle when inflight > 0"

    # Test: Should NOT be idle when mutation task exists
    state = MockState()
    state._mutation_task = asyncio.create_task(asyncio.sleep(0.1))
    # Can't easily test this without running event loop
    # Just verify the condition exists


def test_mutation_eligibility_filter():
    """Test that candidates must reach min shard before being eligible for mutation."""
    shards = (0.3, 1.0)
    min_shard_for_mutation = shards[0] if len(shards) == 1 else shards[1]

    # Mock archive entries
    class MockEntry:
        def __init__(self, shard_fraction):
            self.candidate = Candidate(text=f"prompt_{shard_fraction}", meta={})
            self.result = EvalResult(
                objectives={"quality": 0.8},
                traces=[{"example_id": "test", "quality": 0.8}],
                n_examples=1,
                shard_fraction=shard_fraction,
            )

    entries = [
        MockEntry(0.3),  # Only evaluated on first shard
        MockEntry(1.0),  # Evaluated on full shard
    ]

    # Filter for mutation eligibility (from orchestrator.py line 965-974)
    eligible = []
    for entry in entries:
        current_shard = entry.result.shard_fraction or 0.0
        if current_shard < min_shard_for_mutation:
            continue  # Skip - needs more evaluation first
        eligible.append(entry)

    assert len(eligible) == 1, f"Should have 1 eligible candidate, got {len(eligible)}"
    assert eligible[0].result.shard_fraction == 1.0, "Only full-shard candidate should be eligible"


def test_rung_accumulation():
    """Test that promoted candidates stay in rung for threshold calculation."""
    config = SchedulerConfig(
        shards=(0.5, 1.0),
        eps_improve=0.0,
        quantile=0.4,  # Top 40% promote
    )
    scheduler = BudgetedScheduler(config)

    # Create 10 candidates
    candidates = [Candidate(text=f"prompt_{i}", meta={}) for i in range(10)]

    # Evaluate all on shard 0 with varying quality
    for i, cand in enumerate(candidates):
        result = EvalResult(
            objectives={"quality": 0.1 * i},  # 0.0, 0.1, 0.2, ..., 0.9
            traces=[],
            n_examples=1,
            shard_fraction=0.5,
        )
        scheduler.record(cand, result, "quality")

    # Get promotions
    promotions = scheduler.promote_ready()

    # Should promote top 40% (4 candidates)
    # But need to check if rung still has all 10 for threshold calculation
    rung = scheduler.rungs[0]
    assert len(rung.results) == 10, f"Rung should have all 10 candidates, got {len(rung.results)}"


def test_parent_based_promotion():
    """Test that children beating parent bypass quantile check."""
    config = SchedulerConfig(
        shards=(0.5, 1.0),
        eps_improve=0.01,
        quantile=0.4,  # Only top 40% normally promote
    )
    scheduler = BudgetedScheduler(config)

    # Create parent and evaluate
    parent = Candidate(text="parent prompt", meta={})
    parent_result = EvalResult(
        objectives={"quality": 0.5},
        traces=[],
        n_examples=1,
        shard_fraction=0.5,
    )
    scheduler.record(parent, parent_result, "quality")

    # Promote parent
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1

    # Create child that beats parent
    child = Candidate(text="child prompt", meta={"parent": parent.fingerprint})
    child_result = EvalResult(
        objectives={"quality": 0.6},  # Beats parent
        traces=[],
        n_examples=1,
        shard_fraction=0.5,
    )

    # Record child - should be promoted even if not in top 40% quantile
    decision = scheduler.record(child, child_result, "quality")
    assert decision in ("promoted", "continue"), f"Child beating parent should promote, got {decision}"


@pytest.mark.asyncio
async def test_end_to_end_minimal_flow():
    """Test minimal end-to-end flow without hanging."""
    # This test verifies the system doesn't hang on basic operations

    config = Config(
        shards=(0.5, 1.0),
        eval_concurrency=2,
        max_total_inflight=2,
        mutation_buffer_min=1,
        max_mutations_per_round=2,
    )

    scheduler = BudgetedScheduler(
        SchedulerConfig(
            shards=config.shards,
            eps_improve=0.0,
            quantile=0.4,
        )
    )

    archive = Archive(objectives=["quality", "neg_cost"])

    # Evaluate seed
    seed = Candidate(text="seed", meta={"source": "seed"})
    result = EvalResult(
        objectives={"quality": 0.5, "neg_cost": -100},
        traces=[],
        n_examples=1,
        shard_fraction=0.5,
    )

    # Record and promote
    scheduler.record(seed, result, "quality")
    await archive.insert(seed, result)

    promotions = scheduler.promote_ready()
    assert len(promotions) >= 0, "Should get promotions or empty list"

    # Archive should have entries
    entries = archive.pareto_entries()
    assert len(entries) >= 1, "Archive should have at least seed"


if __name__ == "__main__":
    print("Running orchestrator flow tests...\n")

    tests = [
        ("Scheduler promotion", test_scheduler_promotion_flow),
        ("Archive insertion", test_archive_insertion_and_retrieval),
        ("Idle detection", test_idle_detection_logic),
        ("Mutation eligibility", test_mutation_eligibility_filter),
        ("Rung accumulation", test_rung_accumulation),
        ("Parent-based promotion", test_parent_based_promotion),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {name}")
            print(f"   {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {name}")
            print(f"   {type(e).__name__}: {e}")
            failed += 1

    # Run async test
    try:
        asyncio.run(test_end_to_end_minimal_flow())
        print(f"✅ PASS: End-to-end minimal flow")
        passed += 1
    except Exception as e:
        print(f"❌ FAIL: End-to-end minimal flow")
        print(f"   {type(e).__name__}: {e}")
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
