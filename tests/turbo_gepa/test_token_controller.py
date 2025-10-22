import asyncio

from turbo_gepa.interfaces import Candidate
from turbo_gepa.token_controller import TokenCostController


def test_token_controller_compresses_text():
    controller = TokenCostController(max_tokens=8)
    candidate = Candidate(text="one two three four five six seven nine.\n\nshort clause")

    async def evaluate(_: Candidate) -> float:
        return 1.0

    compressed = asyncio.run(controller.compress(candidate, delta=0.01, evaluate=evaluate))
    assert len(compressed.text.split()) <= controller.max_tokens
    assert compressed.meta.get("compressed")
