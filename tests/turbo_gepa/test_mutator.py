import asyncio

from turbo_gepa.interfaces import Candidate
from turbo_gepa.mutator import MutationConfig, Mutator


async def fake_reflection(traces, parent_prompt: str, parent_meta=None):
    return [parent_prompt + "\n\nAdd explicit verification steps."]


def test_mutator_produces_mutations_with_token_guard():
    """Test that mutator produces LLM-reflection-based mutations only."""
    config = MutationConfig(
        reflection_batch_size=4,
        max_mutations=5,
        max_tokens=200,
    )

    async def fake_batch_reflection(contexts, num_mutations):
        return [ctx["prompt"] + "\n\nAdd explicit verification steps." for ctx in contexts[:num_mutations]]

    mutator = Mutator(
        config,
        batch_reflection_runner=fake_batch_reflection,
        temperature_mutations_enabled=False,  # Disable temperature for this test
        seed=3,
    )
    candidate = Candidate(text="You are a helpful assistant.")

    # Build parent contexts in new format
    parent_contexts = [
        {
            "candidate": candidate,
            "failures": [
                ("example-1", [{"quality": 0.5, "trace": "Failed on example 1"}]),
                ("example-2", [{"quality": 0.3, "trace": "Failed on example 2"}]),
            ]
        }
    ]

    proposals = asyncio.run(mutator.propose(parent_contexts, num_mutations=5))

    # Should get mutations from reflection
    assert proposals
    assert len(proposals) > 0

    for proposal in proposals:
        # Check token limit
        assert len(proposal.text.split()) <= config.max_tokens
        # Check parent tracking
        assert proposal.meta.get("parent") == candidate.fingerprint
        # Check that it's marked as incremental reflection
        assert proposal.meta.get("generation_method") == "incremental_reflection"


def test_mutator_operator_credit_adjusts_spec_quota():
    config = MutationConfig(
        reflection_batch_size=2,
        max_mutations=5,
        max_tokens=200,
    )

    async def fake_batch_reflection(contexts, num_mutations):
        return [ctx["prompt"] + "\nReflection mutation" for ctx in contexts[:num_mutations]]

    call_counter = {"calls": 0}

    async def fake_spec_runner(examples, num_specs):
        call_counter["calls"] += 1
        base = call_counter["calls"] * 10
        return [f"Spec mutation {base + i}" for i in range(num_specs)]

    mutator = Mutator(
        config,
        batch_reflection_runner=fake_batch_reflection,
        spec_induction_runner=fake_spec_runner,
        temperature_mutations_enabled=False,
        seed=7,
    )

    candidate = Candidate(text="Analyze the prompt carefully.")
    parent_contexts = [{"candidate": candidate, "failures": []}]
    task_examples = [{"input": "example", "expected_answer": "42"}]

    proposals = asyncio.run(mutator.propose(parent_contexts, num_mutations=4, task_examples=task_examples))
    initial_spec = sum(1 for p in proposals if p.meta.get("generation_method") == "spec_induction")
    assert initial_spec > 0

    # Penalize spec induction and reward reflection
    for _ in range(3):
        mutator.report_outcome("spec_induction", success=False)
        mutator.report_outcome("incremental_reflection", success=True)

    proposals_after = asyncio.run(mutator.propose(parent_contexts, num_mutations=4, task_examples=task_examples))
    later_spec = sum(1 for p in proposals_after if p.meta.get("generation_method") == "spec_induction")

    assert later_spec <= initial_spec
    assert call_counter["calls"] >= 1

    # Subsequent call should draw from cache without additional runner invocation when enough specs cached
    prev_calls = call_counter["calls"]
    proposals_cached = asyncio.run(mutator.propose(parent_contexts, num_mutations=2, task_examples=task_examples))
    cached_spec = sum(1 for p in proposals_cached if p.meta.get("generation_method") == "spec_induction")
    assert cached_spec >= 0
    assert call_counter["calls"] == prev_calls
