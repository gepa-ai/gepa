"""
Pareto + QD archive for TurboGEPA.

This module maintains both a multi-objective frontier and a quality-diversity
grid keyed off simple textual features derived from candidate text.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .cache import candidate_key
from .interfaces import Candidate, EvalResult


@dataclass
class ArchiveEntry:
    candidate: Candidate
    result: EvalResult


class Archive:
    """Hybrid archive exposing Pareto- and QD-based sampling utilities."""

    def __init__(self, bins_length: int, bins_bullets: int, *, flags: Sequence[str] | None = None) -> None:
        self.pareto: Dict[str, ArchiveEntry] = {}
        self.qd_grid: Dict[Tuple[int, int, frozenset[str]], ArchiveEntry] = {}
        self.bins_length = bins_length
        self.bins_bullets = bins_bullets
        self.flags = tuple(flags or ())
        # Locks to prevent concurrent modification of archives
        self._pareto_lock = asyncio.Lock()
        self._qd_lock = asyncio.Lock()

    async def insert(self, candidate: Candidate, result: EvalResult) -> None:
        """Insert or update both the Pareto frontier and QD grid."""
        cand_hash = candidate_key(candidate)
        entry = ArchiveEntry(candidate=candidate, result=result)
        async with self._pareto_lock:
            self._maintain_pareto(cand_hash, entry)
        async with self._qd_lock:
            self._maintain_qd(entry)

    async def batch_insert(self, pairs: Iterable[Tuple[Candidate, EvalResult]]) -> None:
        """Insert multiple candidates concurrently."""
        tasks = [self.insert(candidate, result) for candidate, result in pairs]
        await asyncio.gather(*tasks)

    def pareto_candidates(self) -> List[Candidate]:
        return [entry.candidate for entry in self.pareto.values()]

    def pareto_entries(self) -> List[ArchiveEntry]:
        return list(self.pareto.values())

    def sample_qd(self, limit: int) -> List[Candidate]:
        elites = list(self.qd_grid.values())
        random.shuffle(elites)
        return [entry.candidate for entry in elites[:limit]]

    def select_for_generation(self, k_exploit: int, k_explore: int, objective: str = "quality") -> List[Candidate]:
        """Return a mixed batch of exploit/explore candidates."""
        pareto_sorted = sorted(
            self.pareto.values(),
            key=lambda entry: entry.result.objectives.get(objective, float("-inf")),
            reverse=True,
        )
        exploit = [entry.candidate for entry in pareto_sorted[:k_exploit]]
        explore = self.sample_qd(k_explore)
        missing = max(0, k_exploit + k_explore - len(exploit) - len(explore))
        if missing > 0:
            explore.extend([entry.candidate for entry in pareto_sorted[len(exploit) : len(exploit) + missing]])
        return exploit + explore

    def top_modules(self, limit: int = 3) -> List[str]:
        """Extract representative modules/sections from top Pareto candidates."""
        sections: List[str] = []
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
            if dominates(entry.result, existing.result):
                dominated.append(key)
            elif dominates(existing.result, entry.result):
                return
        for key in dominated:
            del self.pareto[key]
        self.pareto[cand_hash] = entry

    def _maintain_qd(self, entry: ArchiveEntry) -> None:
        cell = qd_cell(entry.candidate, self.bins_length, self.bins_bullets, self.flags)
        elite = self.qd_grid.get(cell)
        if elite is None or dominates(entry.result, elite.result):
            self.qd_grid[cell] = entry


def dominates(lhs: EvalResult, rhs: EvalResult) -> bool:
    """Return True if ``lhs`` dominates ``rhs`` across all objectives."""
    lhs_keys = set(lhs.objectives)
    rhs_keys = set(rhs.objectives)
    if lhs_keys != rhs_keys:
        return False
    better_or_equal = all(lhs.objectives[k] >= rhs.objectives[k] for k in lhs_keys)
    strictly_better = any(lhs.objectives[k] > rhs.objectives[k] for k in lhs_keys)
    return better_or_equal and strictly_better


def qd_cell(
    candidate: Candidate,
    bins_length: int,
    bins_bullets: int,
    flags: Sequence[str],
) -> Tuple[int, int, frozenset[str]]:
    """Bucket the candidate into a QD cell based on cheap textual features."""
    length = len(candidate.text.split())
    bullets = candidate.text.count("- ")
    flag_hits = set()
    lowered = candidate.text.lower()
    for flag in flags:
        if flag in lowered:
            flag_hits.add(flag)
    length_bin = min(length // 50, bins_length - 1)
    bullet_bin = min(bullets, bins_bullets - 1)
    return (length_bin, bullet_bin, frozenset(flag_hits))
