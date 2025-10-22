"""Test that evaluator handles LLM failures gracefully."""

import asyncio
import pytest

from turbo_gepa.cache import DiskCache
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate


@pytest.fixture
def cache(tmp_path):
    """Create a temporary cache."""
    return DiskCache(str(tmp_path / "cache"))


def test_evaluator_handles_task_runner_failure(cache):
    """Test that evaluator returns zero scores when task runner fails."""

    # Create a task runner that always fails
    async def failing_task_runner(candidate: Candidate, example_id: str):
        raise RuntimeError("Simulated LLM API failure")

    evaluator = AsyncEvaluator(cache=cache, task_runner=failing_task_runner)

    candidate = Candidate(text="Test prompt", meta={})
    example_ids = ["ex1", "ex2", "ex3"]

    async def run_test():
        # This should not crash, but return zero scores
        result = await evaluator.eval_on_shard(
            candidate=candidate,
            example_ids=example_ids,
            concurrency=2,
            shard_fraction=1.0,
        )

        # Verify we got a result (not a crash)
        assert result is not None
        assert result.n_examples == len(example_ids)

        # Verify all scores are zero (failure case)
        assert result.objectives.get("quality", -1.0) == 0.0
        assert result.objectives.get("neg_cost", -1.0) == 0.0

        # Verify traces contain error information
        assert len(result.traces) == len(example_ids)
        for trace in result.traces:
            assert "error" in trace
            assert "Simulated LLM API failure" in trace["error"]

    asyncio.run(run_test())


def test_evaluator_handles_partial_failures(cache):
    """Test that evaluator handles some successes and some failures."""

    call_count = 0

    async def flaky_task_runner(candidate: Candidate, example_id: str):
        nonlocal call_count
        call_count += 1

        # Fail on even numbered calls
        if call_count % 2 == 0:
            raise RuntimeError("Simulated intermittent failure")

        # Succeed on odd numbered calls
        return {
            "quality": 1.0,
            "neg_cost": -100.0,
            "tokens": 100.0,
        }

    evaluator = AsyncEvaluator(cache=cache, task_runner=flaky_task_runner)

    candidate = Candidate(text="Test prompt", meta={})
    example_ids = ["ex1", "ex2", "ex3", "ex4"]

    async def run_test():
        result = await evaluator.eval_on_shard(
            candidate=candidate,
            example_ids=example_ids,
            concurrency=2,
            shard_fraction=1.0,
        )

        # Should get result with partial success
        assert result is not None
        assert result.n_examples == len(example_ids)

        # Should have averaged scores (2 successes, 2 failures)
        # Average quality = (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5
        assert result.objectives.get("quality") == pytest.approx(0.5)

        # Verify traces show both successes and failures
        errors_found = sum(1 for trace in result.traces if "error" in trace)
        successes_found = sum(1 for trace in result.traces if "error" not in trace)
        assert errors_found == 2
        assert successes_found == 2

    asyncio.run(run_test())


def test_evaluator_uses_cache_on_retry(cache):
    """Test that failed evaluations are not cached and will be retried."""

    attempt_count = {}

    async def eventually_succeeds_task_runner(candidate: Candidate, example_id: str):
        # Track attempt count per example
        if example_id not in attempt_count:
            attempt_count[example_id] = 0
        attempt_count[example_id] += 1

        # Fail on first attempt, succeed on second
        if attempt_count[example_id] == 1:
            raise RuntimeError("First attempt fails")

        return {
            "quality": 1.0,
            "neg_cost": -100.0,
            "tokens": 100.0,
        }

    evaluator = AsyncEvaluator(cache=cache, task_runner=eventually_succeeds_task_runner)

    candidate = Candidate(text="Test prompt", meta={})
    example_ids = ["ex1"]

    async def run_test():
        # First evaluation - should fail
        result1 = await evaluator.eval_on_shard(
            candidate=candidate,
            example_ids=example_ids,
            concurrency=1,
            shard_fraction=1.0,
        )

        # Should get zero score due to failure
        assert result1.objectives.get("quality") == 0.0
        assert result1.traces[0].get("error") is not None

        # Second evaluation - should retry (not use cache) and succeed
        result2 = await evaluator.eval_on_shard(
            candidate=candidate,
            example_ids=example_ids,
            concurrency=1,
            shard_fraction=1.0,
        )

        # Should get success score
        assert result2.objectives.get("quality") == 1.0
        assert "error" not in result2.traces[0]

        # Verify we actually retried (attempt count should be 2)
        assert attempt_count["ex1"] == 2

    asyncio.run(run_test())
