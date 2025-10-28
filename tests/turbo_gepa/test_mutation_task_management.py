#!/usr/bin/env python3
"""
Tests for mutation task management to prevent spam and ensure proper lifecycle.
Run with: python -m pytest tests/turbo_gepa/test_mutation_task_management.py -v
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from turbo_gepa.config import Config
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig


@pytest.fixture
def mock_config():
    """Create a minimal config for testing."""
    return Config(
        shards=(0.5, 1.0),
        eval_concurrency=4,
        max_total_inflight=4,
        mutation_buffer_min=2,
        max_mutations_per_round=2,
    )


@pytest.fixture
def mock_dataset():
    """Create a minimal dataset."""
    return [{"input": "test", "answer": "test", "id": "test_1"}]


class MockOrchestrator:
    """Mock orchestrator to test mutation task management."""

    def __init__(self, config):
        self.config = config
        self._mutation_task = None
        self._mutation_buffer = []
        self.queue = []
        self._total_inflight = 0
        self.evaluations_run = 0
        self.mutation_task_created_count = 0
        self.spawn_mutations_called = 0

    async def _spawn_mutations(self, callback):
        """Mock spawn mutations that takes some time."""
        self.spawn_mutations_called += 1
        await asyncio.sleep(0.1)  # Simulate LLM call
        # Add some mock mutations
        for i in range(2):
            callback(Candidate(text=f"mutation_{i}", meta={}))

    async def run_mutation_management_cycle(self, max_evaluations=None):
        """Simulate one cycle of the main loop."""
        should_spawn_mutations = (
            len(self.queue) + len(self._mutation_buffer)
        ) < self.config.mutation_buffer_min and (
            max_evaluations is None or self.evaluations_run < max_evaluations
        )

        # This is the logic from orchestrator.py
        if should_spawn_mutations:
            if self._mutation_task is None:
                # No task exists - create one
                self.mutation_task_created_count += 1
                self._mutation_task = asyncio.create_task(
                    self._spawn_mutations(callback=self._mutation_buffer.append)
                )
            elif self._mutation_task.done():
                # Previous task completed - create new one
                self.mutation_task_created_count += 1
                self._mutation_task = asyncio.create_task(
                    self._spawn_mutations(callback=self._mutation_buffer.append)
                )
            # else: task is running, don't create duplicate


@pytest.mark.asyncio
async def test_mutation_task_no_spam(mock_config):
    """Test that mutation tasks are not created multiple times in rapid succession."""
    orch = MockOrchestrator(mock_config)

    # Run 100 cycles rapidly (simulating tight loop)
    for _ in range(100):
        await orch.run_mutation_management_cycle()
        await asyncio.sleep(0.001)  # Minimal delay

    # Should have created exactly 1 task (not 100!)
    assert orch.mutation_task_created_count == 1, (
        f"Expected 1 mutation task, got {orch.mutation_task_created_count}. "
        "This indicates mutation task spam bug!"
    )

    # Wait for task to complete
    if orch._mutation_task:
        await orch._mutation_task

    # Should have called spawn_mutations exactly once
    assert orch.spawn_mutations_called == 1


@pytest.mark.asyncio
async def test_mutation_task_recreation_after_completion(mock_config):
    """Test that a new task is created after the previous one completes."""
    orch = MockOrchestrator(mock_config)

    # First cycle - create task
    await orch.run_mutation_management_cycle()
    assert orch.mutation_task_created_count == 1
    first_task = orch._mutation_task

    # Wait for first task to complete
    await first_task
    assert first_task.done()

    # Clear buffer so should_spawn is True again
    orch._mutation_buffer.clear()

    # Second cycle - should create new task
    await orch.run_mutation_management_cycle()
    assert orch.mutation_task_created_count == 2, "Should create new task after previous completes"

    # Wait for second task
    if orch._mutation_task:
        await orch._mutation_task


@pytest.mark.asyncio
async def test_mutation_task_waits_for_running_task(mock_config):
    """Test that no new task is created while one is running."""
    orch = MockOrchestrator(mock_config)

    # Start first task
    await orch.run_mutation_management_cycle()
    assert orch.mutation_task_created_count == 1

    # Run many more cycles while task is running
    for _ in range(50):
        await orch.run_mutation_management_cycle()
        await asyncio.sleep(0.001)

    # Should still be only 1 task
    assert orch.mutation_task_created_count == 1, "Should not create new task while one is running"

    # Wait for task to complete
    if orch._mutation_task:
        await orch._mutation_task


@pytest.mark.asyncio
async def test_mutation_task_respects_buffer_min(mock_config):
    """Test that mutations are not spawned when buffer is full."""
    orch = MockOrchestrator(mock_config)

    # Fill buffer to mutation_buffer_min
    for i in range(mock_config.mutation_buffer_min):
        orch._mutation_buffer.append(Candidate(text=f"existing_{i}", meta={}))

    # Run cycle - should NOT create task
    await orch.run_mutation_management_cycle()
    assert orch.mutation_task_created_count == 0, "Should not create task when buffer is full"

    # Clear buffer
    orch._mutation_buffer.clear()

    # Now should create task
    await orch.run_mutation_management_cycle()
    assert orch.mutation_task_created_count == 1, "Should create task when buffer is empty"

    if orch._mutation_task:
        await orch._mutation_task


@pytest.mark.asyncio
async def test_mutation_task_respects_eval_budget(mock_config):
    """Test that mutations are not spawned when evaluation budget is exhausted."""
    orch = MockOrchestrator(mock_config)
    max_evaluations = 10
    orch.evaluations_run = 10  # Budget exhausted

    # Run cycle - should NOT create task
    await orch.run_mutation_management_cycle(max_evaluations=max_evaluations)
    assert orch.mutation_task_created_count == 0, "Should not create task when eval budget exhausted"

    # Reset evaluations
    orch.evaluations_run = 5  # Still has budget

    # Now should create task
    await orch.run_mutation_management_cycle(max_evaluations=max_evaluations)
    assert orch.mutation_task_created_count == 1, "Should create task when budget available"

    if orch._mutation_task:
        await orch._mutation_task


@pytest.mark.asyncio
async def test_mutation_buffer_fills_correctly(mock_config):
    """Test that mutations are added to buffer when task completes."""
    orch = MockOrchestrator(mock_config)

    # Create and wait for task
    await orch.run_mutation_management_cycle()
    assert len(orch._mutation_buffer) == 0, "Buffer should be empty before task completes"

    # Wait for task to complete
    if orch._mutation_task:
        await orch._mutation_task

    # Buffer should be filled
    assert len(orch._mutation_buffer) == 2, f"Buffer should have 2 mutations, got {len(orch._mutation_buffer)}"


@pytest.mark.asyncio
async def test_mutation_task_lifecycle_multiple_rounds(mock_config):
    """Test complete lifecycle over multiple rounds."""
    orch = MockOrchestrator(mock_config)

    rounds = 3
    for round_num in range(rounds):
        # Start task
        await orch.run_mutation_management_cycle()

        # Task should be running
        assert orch._mutation_task is not None
        assert not orch._mutation_task.done()

        # Wait for completion
        await orch._mutation_task

        # Buffer should be filled
        assert len(orch._mutation_buffer) >= 2

        # Clear buffer for next round
        orch._mutation_buffer.clear()

        # Small delay
        await asyncio.sleep(0.01)

    # Should have created exactly 'rounds' tasks
    assert orch.mutation_task_created_count == rounds


if __name__ == "__main__":
    # Run tests manually
    async def run_all_tests():
        config = Config(
            shards=(0.5, 1.0),
            eval_concurrency=4,
            max_total_inflight=4,
            mutation_buffer_min=2,
            max_mutations_per_round=2,
        )

        print("Running mutation task management tests...\n")

        tests = [
            ("No spam", test_mutation_task_no_spam(config)),
            ("Recreation after completion", test_mutation_task_recreation_after_completion(config)),
            ("Waits for running task", test_mutation_task_waits_for_running_task(config)),
            ("Respects buffer min", test_mutation_task_respects_buffer_min(config)),
            ("Respects eval budget", test_mutation_task_respects_eval_budget(config)),
            ("Buffer fills correctly", test_mutation_buffer_fills_correctly(config)),
            ("Multiple rounds lifecycle", test_mutation_task_lifecycle_multiple_rounds(config)),
        ]

        passed = 0
        failed = 0

        for name, test_coro in tests:
            try:
                await test_coro
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

        print(f"\n{passed} passed, {failed} failed")
        return failed == 0

    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
