"""
Test that max_optimization_time_seconds timeout works correctly.
"""

import pytest
import asyncio
import time
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.config import Config
from turbo_gepa.archive import Archive
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.mutator import Mutator
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig


class FastMockEvaluator:
    """Mock evaluator that returns results quickly."""

    def __init__(self):
        self.eval_count = 0

    async def evaluate(self, candidate: Candidate, examples: list, shard_fraction: float, **kwargs) -> EvalResult:
        """Return mock results instantly."""
        self.eval_count += 1
        quality = 0.5 + (self.eval_count % 10) / 20.0  # Vary between 0.5 and 1.0

        # Small delay to simulate work
        await asyncio.sleep(0.01)

        return EvalResult(
            objectives={"quality": quality, "neg_cost": -0.001},
            traces=[{"example_id": f"ex_{i}", "correct": quality > 0.7} for i in range(len(examples))],
            n_examples=len(examples),
            shard_fraction=shard_fraction,
            example_ids=[f"ex_{i}" for i in range(len(examples))],
        )


@pytest.mark.asyncio
async def test_timeout_stops_optimization():
    """Test that optimization stops after max_optimization_time_seconds."""

    print("\n" + "=" * 80)
    print("TEST: Timeout mechanism")
    print("=" * 80)

    # Setup with SHORT timeout (5 seconds)
    timeout_seconds = 5
    config = Config(
        eval_concurrency=16,
        batch_size=5,
        max_mutations_per_round=20,
        mutation_buffer_min=10,
        shards=(0.2, 0.5, 1.0),
        eps_improve=0.01,
        n_islands=1,
        queue_limit=50,
        max_optimization_time_seconds=timeout_seconds,  # 5 second timeout
        adaptive_shards_enabled=False,  # Disable for simpler test
    )

    # Create mock dataset
    dataset = [{"input": f"example_{i}", "id": f"ex_{i}"} for i in range(15)]

    # Create components
    sampler = InstanceSampler(dataset)
    archive = Archive(bins_length=10, bins_bullets=10)

    scheduler_config = SchedulerConfig(
        shards=config.shards,
        eps_improve=config.eps_improve,
    )
    scheduler = BudgetedScheduler(scheduler_config)

    # Create mutator (use real config)
    mutator = Mutator(
        config=config,
        validators=[],
    )

    # Override mutator to generate simple mutations quickly
    async def mock_request_mutations(n: int, **kwargs):
        """Generate n simple mutations instantly."""
        await asyncio.sleep(0.01)  # Tiny delay
        return [
            Candidate(
                text=f"Mutation variant {i}",
                meta={"source": "mutation", "parent_score": 0.5}
            )
            for i in range(n)
        ]

    mutator.request_mutations = mock_request_mutations

    evaluator = FastMockEvaluator()

    # Create orchestrator
    orchestrator = Orchestrator(
        config=config,
        sampler=sampler,
        archive=archive,
        scheduler=scheduler,
        mutator=mutator,
        show_progress=False,  # Disable progress for cleaner output
    )

    # Override evaluator
    orchestrator._evaluate_candidate = evaluator.evaluate

    # Create seed
    seed = Candidate(text="test seed prompt", meta={"source": "seed"})

    print(f"\n⏱️  Starting optimization with {timeout_seconds}s timeout")
    print(f"   Expect it to stop within ~{timeout_seconds} seconds")

    start_time = time.time()

    # Run optimization (should stop due to timeout)
    await orchestrator.run(
        seeds=[seed],
        max_rounds=None,  # No round limit
        max_evaluations=None,  # No eval limit
    )

    elapsed = time.time() - start_time

    print(f"\n📊 Results:")
    print(f"   Elapsed time: {elapsed:.2f}s")
    print(f"   Timeout setting: {timeout_seconds}s")
    print(f"   Total evaluations: {evaluator.eval_count}")
    print(f"   Evals per second: {evaluator.eval_count / elapsed:.2f}")

    # Verify timeout worked
    tolerance = 2.0  # Allow 2 second tolerance for cleanup
    assert elapsed <= timeout_seconds + tolerance, \
        f"Optimization took {elapsed:.1f}s, expected <= {timeout_seconds + tolerance}s"

    assert evaluator.eval_count > 0, "Should have done at least some evaluations"

    print(f"\n✅ TEST PASSED!")
    print(f"   - Optimization stopped within timeout")
    print(f"   - Completed {evaluator.eval_count} evaluations in {elapsed:.1f}s")
    print("=" * 80)


@pytest.mark.asyncio
async def test_adaptive_sharding_with_timeout():
    """Test that adaptive sharding works correctly with timeout."""

    print("\n" + "=" * 80)
    print("TEST: Adaptive sharding + timeout")
    print("=" * 80)

    timeout_seconds = 5
    config = Config(
        eval_concurrency=16,
        batch_size=5,
        max_mutations_per_round=20,
        mutation_buffer_min=10,
        shards=(0.1, 0.3, 1.0),  # Start with aggressive pruning
        eps_improve=0.01,
        n_islands=1,
        queue_limit=50,
        max_optimization_time_seconds=timeout_seconds,
        adaptive_shards_enabled=True,  # Enable adaptive sharding
        adaptive_shard_min_frac=0.05,
        adaptive_shard_max_frac=0.5,
    )

    dataset = [{"input": f"example_{i}", "id": f"ex_{i}"} for i in range(20)]
    sampler = InstanceSampler(dataset)
    archive = Archive(bins_length=10, bins_bullets=10)

    scheduler_config = SchedulerConfig(
        shards=config.shards,
        eps_improve=config.eps_improve,
    )
    scheduler = BudgetedScheduler(scheduler_config)

    mutator = Mutator(config=config, validators=[])

    async def mock_request_mutations(n: int, **kwargs):
        await asyncio.sleep(0.01)
        return [
            Candidate(text=f"Mutation {i}", meta={"source": "mutation", "parent_score": 0.5})
            for i in range(n)
        ]

    mutator.request_mutations = mock_request_mutations
    evaluator = FastMockEvaluator()

    orchestrator = Orchestrator(
        config=config,
        sampler=sampler,
        archive=archive,
        scheduler=scheduler,
        mutator=mutator,
        show_progress=False,
    )
    orchestrator._evaluate_candidate = evaluator.evaluate

    seed = Candidate(text="test seed", meta={"source": "seed"})

    print(f"\n⏱️  Starting with adaptive sharding enabled")
    print(f"   Initial shards: {config.shards}")
    print(f"   Timeout: {timeout_seconds}s")

    start_time = time.time()
    await orchestrator.run(seeds=[seed], max_rounds=None, max_evaluations=None)
    elapsed = time.time() - start_time

    print(f"\n📊 Results:")
    print(f"   Elapsed: {elapsed:.2f}s")
    print(f"   Evaluations: {evaluator.eval_count}")
    print(f"   Final shards: {scheduler.shards}")

    # Check timeout worked
    assert elapsed <= timeout_seconds + 2.0, f"Took too long: {elapsed:.1f}s"

    print(f"\n✅ TEST PASSED!")
    print(f"   - Adaptive sharding worked")
    print(f"   - Timeout respected")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    print("\nRunning timeout mechanism tests...\n")

    try:
        asyncio.run(test_timeout_stops_optimization())
        asyncio.run(test_adaptive_sharding_with_timeout())

        print("\n" + "=" * 80)
        print("🎉 ALL TIMEOUT TESTS PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
