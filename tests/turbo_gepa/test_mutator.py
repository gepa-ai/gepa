import asyncio

from turbo_gepa.interfaces import Candidate
from turbo_gepa.mutator import MutationConfig, Mutator


async def fake_reflection(traces, parent_prompt: str):
    return [parent_prompt + "\n\nAdd explicit verification steps."]


def test_mutator_produces_mutations_with_token_guard():
    config = MutationConfig(
        amortized_rate=1.0,
        reflection_batch_size=4,
        max_mutations=5,
        max_tokens=200,
    )
    mutator = Mutator(config, reflection_runner=fake_reflection, seed=3)
    candidate = Candidate(text="You are a helpful assistant.")
    proposals = asyncio.run(mutator.propose(candidate, []))
    assert proposals
    for proposal in proposals:
        assert len(proposal.text.split()) <= config.max_tokens
        assert proposal.meta.get("parent") == candidate.fingerprint
