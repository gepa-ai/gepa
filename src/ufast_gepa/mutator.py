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

ReflectionRunner = Callable[[Sequence[Dict[str, object]], str, Dict[str, object] | None], Awaitable[Sequence[str]]]
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
        temperature_mutations_enabled: bool = True,
    ) -> None:
        self.config = config
        extra_validators = list(validators or [])
        extra_validators.append(_default_token_validator(config.max_tokens))
        self.validators = extra_validators
        self.reflection_runner = reflection_runner
        self.random = random.Random(seed)
        self.temperature_mutations_enabled = temperature_mutations_enabled

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
                # Pass parent metadata to reflection runner (for temperature context)
                mutated_texts = await self.reflection_runner(batch, candidate.text, candidate.meta)
                for idx, text in enumerate(mutated_texts):
                    meta = dict(candidate.meta)
                    meta.update({"edit": "reflection", "parent": candidate.fingerprint, "proposal_idx": idx})
                    proposals.append(Candidate(text=text, meta=meta))

        filtered = self._filter(proposals)
        return filtered[: self.config.max_mutations]

    def _amortized_edits(self, candidate: Candidate) -> List[Candidate]:
        text = candidate.text
        edits: List[Candidate] = []

        # Text mutations (keep existing temperature if present)
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

        # Temperature mutations (20% chance - keep text same, vary temperature)
        # Only mutate temperature if:
        # 1. Model supports temperature mutations (checked upfront)
        # 2. Parent candidate has temperature (opt-in behavior)
        if self.temperature_mutations_enabled and "temperature" in candidate.meta and self.random.random() < 0.2:
            current_temp = candidate.meta["temperature"]
            # Try different temperatures: lower, higher, and standard values
            # Keep within valid range [0.0, 1.0]
            temp_variants = []

            # Lower creativity (move toward deterministic)
            if current_temp > 0.3:
                temp_variants.append(max(0.0, current_temp - 0.3))

            # Higher creativity (move toward diverse)
            if current_temp < 0.7:
                temp_variants.append(min(1.0, current_temp + 0.3))

            # Anchors: try standard values if not already there
            for anchor in [0.0, 0.5, 1.0]:
                if abs(current_temp - anchor) > 0.2:
                    temp_variants.append(anchor)

            # Create variants (limit to 2 to avoid explosion)
            for temp in self.random.sample(temp_variants, min(2, len(temp_variants))):
                # Clamp to valid range [0.0, 1.0]
                temp = max(0.0, min(1.0, temp))
                meta = dict(candidate.meta, temperature=temp, edit="temperature_shift", parent=candidate.fingerprint)
                edits.append(Candidate(text=text, meta=meta))

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
