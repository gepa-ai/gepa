"""
Mutation utilities blending deterministic edits with speculative reflection.

Real reflection calls are pluggable; the default implementation focuses on
rule-based edits and provides extension hooks for batched reflection.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable, List, Sequence, Set, Tuple

from .interfaces import Candidate

ReflectionRunner = Callable[[Sequence[Dict[str, object]], str], Awaitable[Sequence[str]]]
Validator = Callable[[Candidate], None]


def _default_token_validator(max_tokens: int) -> Validator:
    def _validate(candidate: Candidate) -> None:
        if len(candidate.text.split()) > max_tokens:
            raise ValueError("candidate exceeds token budget")

    return _validate


@dataclass
class MutationConfig:
    amortized_rate: float
    reflection_batch_size: int
    max_mutations: int
    max_tokens: int


class Mutator:
    """Combine amortized rule edits with optional reflection proposals."""

    def __init__(
        self,
        config: MutationConfig,
        *,
        validators: Iterable[Validator] | None = None,
        reflection_runner: ReflectionRunner | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        extra_validators = list(validators or [])
        extra_validators.append(_default_token_validator(config.max_tokens))
        self.validators = extra_validators
        self.reflection_runner = reflection_runner
        self.random = random.Random(seed)

    async def propose(
        self,
        candidate: Candidate,
        recent_failures: Iterable[Tuple[str, List[Dict[str, object]]]] | None = None,
    ) -> List[Candidate]:
        """
        Generate mutated candidates.

        Deterministic edits are always attempted; reflection proposals are
        included when a runner is supplied. Results are deduplicated and
        token-guarded.
        """
        proposals: List[Candidate] = []
        amortized = self._amortized_edits(candidate)
        if amortized and self.random.random() <= self.config.amortized_rate:
            proposals.extend(amortized)

        if self.reflection_runner and recent_failures:
            collected = []
            for _example_id, traces in recent_failures:
                collected.extend(traces)
            batch = collected[: self.config.reflection_batch_size]
            if batch:
                mutated_texts = await self.reflection_runner(batch, candidate.text)
                for idx, text in enumerate(mutated_texts):
                    meta = dict(candidate.meta)
                    meta.update({"edit": "reflection", "parent": candidate.fingerprint, "proposal_idx": idx})
                    proposals.append(Candidate(text=text, meta=meta))

        filtered = self._filter(proposals)
        return filtered[: self.config.max_mutations]

    def _amortized_edits(self, candidate: Candidate) -> List[Candidate]:
        text = candidate.text
        edits: List[Candidate] = []

        if "Think through" not in text:
            meta = dict(candidate.meta, edit="reasoning_hint", parent=candidate.fingerprint)
            edits.append(Candidate(text=f"{text}\n\nThink through the solution before answering.", meta=meta))

        if "Avoid" not in text:
            meta = dict(candidate.meta, edit="brevity_clause", parent=candidate.fingerprint)
            edits.append(Candidate(text=f"{text}\n\nAvoid unnecessary verbosity.", meta=meta))

        if "Example" not in text:
            snippet = "\n\nExample:\n- Input: ...\n- Output: ..."
            meta = dict(candidate.meta, edit="example", parent=candidate.fingerprint)
            edits.append(Candidate(text=f"{text}{snippet}", meta=meta))

        tightened = text.replace("You are a helpful assistant.", "You are a meticulous assistant.")
        if tightened != text:
            meta = dict(candidate.meta, edit="tone_adjust", parent=candidate.fingerprint)
            edits.append(Candidate(text=tightened, meta=meta))

        return edits

    def _filter(self, candidates: Iterable[Candidate]) -> List[Candidate]:
        seen: Set[str] = set()
        valid: List[Candidate] = []
        for candidate in candidates:
            fingerprint = hashlib.sha1(candidate.text.encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                for validator in self.validators:
                    validator(candidate)
            except ValueError:
                continue
            valid.append(candidate)
        return valid
