"""
Pareto archive for TurboGEPA.

This module maintains a multi-objective Pareto frontier of non-dominated candidates.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable

from .cache import candidate_key
from .interfaces import Candidate, EvalResult


@dataclass
class ArchiveEntry:
    candidate: Candidate
    result: EvalResult


class Archive:
    """Archive maintaining Pareto frontier of non-dominated candidates."""

    def __init__(self) -> None:
        self.pareto: dict[str, ArchiveEntry] = {}
        # Lock to prevent concurrent modification
        self._pareto_lock = asyncio.Lock()

    async def insert(self, candidate: Candidate, result: EvalResult) -> None:
        """Insert or update the Pareto frontier."""
        cand_hash = candidate_key(candidate)
        entry = ArchiveEntry(candidate=candidate, result=result)
        async with self._pareto_lock:
            self._maintain_pareto(cand_hash, entry)

    async def batch_insert(self, pairs: Iterable[tuple[Candidate, EvalResult]]) -> None:
        """Insert multiple candidates concurrently."""
        tasks = [self.insert(candidate, result) for candidate, result in pairs]
        await asyncio.gather(*tasks)

    def pareto_candidates(self) -> list[Candidate]:
        return [entry.candidate for entry in self.pareto.values()]

    def pareto_entries(self) -> list[ArchiveEntry]:
        return list(self.pareto.values())

    def select_for_generation(self, k: int, objective: str = "quality") -> list[Candidate]:
        """Return top k candidates sorted by objective."""
        pareto_sorted = sorted(
            self.pareto.values(),
            key=lambda entry: entry.result.objectives.get(objective, float("-inf")),
            reverse=True,
        )
        return [entry.candidate for entry in pareto_sorted[:k]]

    def top_modules(self, limit: int = 3) -> list[str]:
        """Extract representative modules/sections from top Pareto candidates."""
        sections: list[str] = []
        for entry in self.pareto.values():
            parts = [segment.strip() for segment in entry.candidate.text.split("\n\n") if segment.strip()]
            sections.extend(parts[:limit])
        uniq = []
        seen = set()
        for section in sections:
            key = section.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(section)
            if len(uniq) >= limit:
                break
        return uniq

    def _maintain_pareto(self, cand_hash: str, entry: ArchiveEntry) -> None:
        dominated = []
        for key, existing in self.pareto.items():
            # Special case: If this is the same candidate (same key), prefer larger shard
            # This ensures full-dataset evaluations replace partial-shard evaluations
            if key == cand_hash:
                new_shard = entry.result.shard_fraction or 0.0
                existing_shard = existing.result.shard_fraction or 0.0

                if new_shard >= existing_shard:
                    # New evaluation is on equal or larger shard - replace the old one
                    dominated.append(key)
                    continue
                else:
                    # Old evaluation was on larger shard - keep it, reject new one
                    return

            # Normal Pareto domination for different candidates
            if dominates(entry.result, existing.result):
                dominated.append(key)
            elif dominates(existing.result, entry.result):
                return
        for key in dominated:
            del self.pareto[key]
        self.pareto[cand_hash] = entry


def dominates(lhs: EvalResult, rhs: EvalResult) -> bool:
    """Return True if ``lhs`` dominates ``rhs`` across all objectives."""
    lhs_keys = set(lhs.objectives)
    rhs_keys = set(rhs.objectives)
    if lhs_keys != rhs_keys:
        return False
    better_or_equal = all(lhs.objectives[k] >= rhs.objectives[k] for k in lhs_keys)
    strictly_better = any(lhs.objectives[k] > rhs.objectives[k] for k in lhs_keys)
    return better_or_equal and strictly_better
