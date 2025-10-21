"""
Token-cost controller for clause pruning and compression.

The implementation applies simple heuristics to remove or tighten clauses
while preserving evaluation quality, delegating the final approval to the
caller via a callback.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List

from .interfaces import Candidate


class TokenCostController:
    """Greedy clause pruning helper."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    async def compress(
        self,
        candidate: Candidate,
        delta: float,
        evaluate: Callable[[Candidate], Awaitable[float]],
    ) -> Candidate:
        """
        Remove clauses while preserving quality within ``delta``.

        The controller splits text into paragraphs, heuristically removes the
        lowest-value ones, and keeps the first configuration within tolerance.
        """
        paragraphs = [p for p in candidate.text.split("\n\n") if p.strip()]
        base_score = await evaluate(candidate)
        best = candidate
        for idx in range(len(paragraphs) - 1, -1, -1):
            trimmed = paragraphs[:idx] + paragraphs[idx + 1 :]
            new_text = "\n\n".join(trimmed)
            if not new_text.strip():
                continue
            new_candidate = Candidate(text=new_text, meta={**candidate.meta, "compressed": True})
            score = await evaluate(new_candidate)
            if len(new_text.split()) <= self.max_tokens and score >= base_score - delta:
                best = new_candidate
                base_score = score
        return best
