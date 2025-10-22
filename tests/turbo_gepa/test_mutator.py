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
