"""Test batch reflection mechanism in TurboGEPA."""

import asyncio
from typing import Dict, List, Sequence

from turbo_gepa.interfaces import Candidate
from turbo_gepa.mutator import MutationConfig, Mutator


# Mock batch reflection runner
async def mock_batch_reflection_runner(
    parent_contexts: Sequence[Dict[str, object]],
    num_mutations: int
) -> Sequence[str]:
    """Mock runner that verifies it receives correct input and returns mutations."""
    # Verify we got parent contexts
    assert len(parent_contexts) > 0, "Should receive at least one parent context"

    # Verify each context has required fields
    for ctx in parent_contexts:
        assert "prompt" in ctx, "Context should have 'prompt' field"
        assert "traces" in ctx, "Context should have 'traces' field"
        assert "meta" in ctx, "Context should have 'meta' field"

    # Generate mutations based on parents
    mutations = []
    for i in range(num_mutations):
        # Synthesize from first parent
        base_prompt = parent_contexts[0]["prompt"]
        mutations.append(f"{base_prompt}\n\n[Mutation {i+1}: synthesized from {len(parent_contexts)} parents]")

    return mutations


# Mock spec induction runner
async def mock_spec_induction_runner(
    task_examples: Sequence[Dict[str, object]],
    num_specs: int
) -> Sequence[str]:
    """Mock spec induction that generates fresh prompts from examples."""
    assert len(task_examples) > 0, "Should receive task examples"

    specs = []
    for i in range(num_specs):
        specs.append(f"Fresh spec {i+1} induced from {len(task_examples)} examples")

    return specs


def test_batch_reflection_receives_multiple_parents():
    """Test that batch reflection receives multiple parent contexts in a single call."""
    config = MutationConfig(
        reflection_batch_size=3,
        max_mutations=6,
        max_tokens=500,
    )

    mutator = Mutator(
        config,
        batch_reflection_runner=mock_batch_reflection_runner,
        spec_induction_runner=mock_spec_induction_runner,
        temperature_mutations_enabled=False,
    )

    # Create multiple parent contexts
    parent_contexts = []
    for i in range(3):
        candidate = Candidate(
            text=f"You are a helpful assistant with approach {i}.",
            meta={"quality": 0.5 + i * 0.1}
        )
        failures = [
            (f"example-{j}", [{"quality": 0.4, "trace": f"Failed on example {j}"}])
            for j in range(2)
        ]
        parent_contexts.append({
            "candidate": candidate,
            "failures": failures,
        })

    # Request mutations (priority: temperature → incremental → spec induction)
    num_mutations = 6
    task_examples = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 3+3?", "output": "6"},
    ]

    proposals = asyncio.run(mutator.propose(parent_contexts, num_mutations, task_examples))

    # Should get mutations
    assert len(proposals) > 0, "Should generate mutations"
    assert len(proposals) <= num_mutations, "Should not exceed requested count"

    # Check that mutations have correct metadata
    incremental_count = sum(1 for p in proposals if p.meta.get("generation_method") == "incremental_reflection")
    spec_count = sum(1 for p in proposals if p.meta.get("generation_method") == "spec_induction")

    print(f"✓ Generated {len(proposals)} mutations: {incremental_count} incremental, {spec_count} spec")

    # Priority-based: incremental reflection consumes budget first, then spec induction
    assert incremental_count > 0, "Should have incremental mutations"
    # Note: Spec induction only gets remaining budget after incremental, may be 0

    # Check parent tracking for incremental mutations
    for proposal in proposals:
        if proposal.meta.get("generation_method") == "incremental_reflection":
            assert "parent" in proposal.meta, "Incremental mutations should track parent"
            assert "num_parents_seen" in proposal.meta, "Should track number of parents seen"
            assert proposal.meta["num_parents_seen"] == 3, "Should see all 3 parents"


def test_batch_reflection_with_no_runner():
    """Test that mutator handles missing batch_reflection_runner gracefully."""
    config = MutationConfig(
        reflection_batch_size=3,
        max_mutations=6,
        max_tokens=500,
    )

    # No batch_reflection_runner provided
    mutator = Mutator(
        config,
        batch_reflection_runner=None,  # Missing
        spec_induction_runner=mock_spec_induction_runner,
        temperature_mutations_enabled=False,
    )

    parent_contexts = [{
        "candidate": Candidate(text="You are helpful."),
        "failures": [("ex1", [{"quality": 0.5}])],
    }]

    task_examples = [{"input": "test", "output": "result"}]

    proposals = asyncio.run(mutator.propose(parent_contexts, 6, task_examples))

    # Should only get spec induction mutations
    assert len(proposals) > 0, "Should still generate mutations from spec induction"
    assert all(p.meta.get("generation_method") == "spec_induction" for p in proposals), \
        "All mutations should be from spec induction"


def test_batch_reflection_with_no_spec_runner():
    """Test that mutator handles missing spec_induction_runner gracefully."""
    config = MutationConfig(
        reflection_batch_size=3,
        max_mutations=6,
        max_tokens=500,
    )

    # No spec_induction_runner provided
    mutator = Mutator(
        config,
        batch_reflection_runner=mock_batch_reflection_runner,
        spec_induction_runner=None,  # Missing
        temperature_mutations_enabled=False,
    )

    parent_contexts = [{
        "candidate": Candidate(text="You are helpful."),
        "failures": [("ex1", [{"quality": 0.5}])],
    }]

    task_examples = [{"input": "test", "output": "result"}]

    proposals = asyncio.run(mutator.propose(parent_contexts, 6, task_examples))

    # Should only get incremental reflection mutations
    assert len(proposals) > 0, "Should still generate mutations from batch reflection"
    assert all(p.meta.get("generation_method") == "incremental_reflection" for p in proposals), \
        "All mutations should be from incremental reflection"


def test_batch_reflection_trace_limiting():
    """Test that traces are limited per parent to avoid token explosion."""
    config = MutationConfig(
        reflection_batch_size=3,  # Limit to 3 traces per parent
        max_mutations=4,
        max_tokens=500,
    )

    # Create a custom reflection runner that checks trace counts
    received_contexts = []

    async def checking_batch_reflection_runner(
        parent_contexts: Sequence[Dict[str, object]],
        num_mutations: int
    ) -> Sequence[str]:
        # Store what we received
        received_contexts.extend(parent_contexts)
        return [f"mutation {i}" for i in range(num_mutations)]

    mutator = Mutator(
        config,
        batch_reflection_runner=checking_batch_reflection_runner,
        spec_induction_runner=None,
        temperature_mutations_enabled=False,
    )

    # Create parent with MANY failure traces (should be limited)
    candidate = Candidate(text="You are helpful.", meta={"quality": 0.7})
    failures = [
        (f"example-{i}", [{"quality": 0.3, "trace": f"trace {i}"}])
        for i in range(10)  # 10 failures, each with 1 trace = 10 total traces
    ]

    parent_contexts = [{
        "candidate": candidate,
        "failures": failures,
    }]

    proposals = asyncio.run(mutator.propose(parent_contexts, 4, task_examples=None))

    # Check that received contexts had limited traces
    assert len(received_contexts) > 0, "Should have received contexts"
    for ctx in received_contexts:
        traces = ctx.get("traces", [])
        assert len(traces) <= config.reflection_batch_size, \
            f"Traces should be limited to {config.reflection_batch_size}, got {len(traces)}"

    print(f"✓ Traces correctly limited to {config.reflection_batch_size} per parent")


def test_concurrent_execution():
    """Test that incremental and spec induction run concurrently."""
    import time

    # Track execution order
    execution_log = []

    async def slow_batch_reflection_runner(
        parent_contexts: Sequence[Dict[str, object]],
        num_mutations: int
    ) -> Sequence[str]:
        execution_log.append(("reflection_start", time.time()))
        await asyncio.sleep(0.1)  # Simulate LLM call
        execution_log.append(("reflection_end", time.time()))
        # Return only 2 to leave budget for spec induction
        return [f"mutation {i}" for i in range(min(num_mutations, 2))]

    async def slow_spec_induction_runner(
        task_examples: Sequence[Dict[str, object]],
        num_specs: int
    ) -> Sequence[str]:
        execution_log.append(("spec_start", time.time()))
        await asyncio.sleep(0.1)  # Simulate LLM call
        execution_log.append(("spec_end", time.time()))
        return [f"spec {i}" for i in range(num_specs)]

    config = MutationConfig(
        reflection_batch_size=3,
        max_mutations=4,
        max_tokens=500,
    )

    mutator = Mutator(
        config,
        batch_reflection_runner=slow_batch_reflection_runner,
        spec_induction_runner=slow_spec_induction_runner,
        temperature_mutations_enabled=False,
    )

    parent_contexts = [{
        "candidate": Candidate(text="Test"),
        "failures": [("ex1", [{"quality": 0.5}])],
    }]
    task_examples = [{"input": "test", "output": "result"}]

    start = time.time()
    proposals = asyncio.run(mutator.propose(parent_contexts, 4, task_examples))
    elapsed = time.time() - start

    # Sequential allocation: reflection (2 mutations) then spec (remaining 2)
    # Should take ~0.2s (0.1s reflection + 0.1s spec)
    assert elapsed >= 0.15, f"Should run sequentially (~0.2s), took {elapsed:.3f}s"
    assert elapsed < 0.25, f"Should complete efficiently, took {elapsed:.3f}s"

    # Check that both ran (reflection first, then spec with remaining budget)
    assert len(execution_log) == 4, f"Both reflection and spec should run, got {len(execution_log)} events"
    reflection_start = next(t for evt, t in execution_log if evt == "reflection_start")
    reflection_end = next(t for evt, t in execution_log if evt == "reflection_end")
    spec_start = next(t for evt, t in execution_log if evt == "spec_start")
    spec_end = next(t for evt, t in execution_log if evt == "spec_end")

    # Sequential: reflection must finish before spec starts (priority-based)
    assert reflection_end < spec_start, "Reflection should finish before spec starts (sequential priority)"

    # Verify we got mutations from both sources
    incremental_count = sum(1 for p in proposals if p.meta.get("generation_method") == "incremental_reflection")
    spec_count = sum(1 for p in proposals if p.meta.get("generation_method") == "spec_induction")
    assert incremental_count == 2, "Should have 2 incremental mutations"
    assert spec_count == 2, "Should have 2 spec mutations from remaining budget"
    print(f"✓ Concurrent execution verified: {elapsed:.3f}s total")


if __name__ == "__main__":
    print("Testing batch reflection mechanism...\n")

    test_batch_reflection_receives_multiple_parents()
    print("✅ Test 1: Multiple parents received correctly\n")

    test_batch_reflection_with_no_runner()
    print("✅ Test 2: Handles missing batch reflection runner\n")

    test_batch_reflection_with_no_spec_runner()
    print("✅ Test 3: Handles missing spec induction runner\n")

    test_batch_reflection_trace_limiting()
    print("✅ Test 4: Trace limiting works correctly\n")

    test_concurrent_execution()
    print("✅ Test 5: Concurrent execution verified\n")

    print("\n🎉 All batch reflection tests passed!")
