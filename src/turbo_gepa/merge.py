"""
DEPRECATED: Merge functionality has been removed from TurboGEPA.

Previously, this module combined sections from multiple prompts through text-splicing.
Now, the reflection LLM handles all prompt improvements, including intelligently
synthesizing ideas from multiple good prompts.

This file is kept for historical reference only and is not used in the codebase.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .interfaces import Candidate


class MergeManager:
    """Generate merged candidates by splicing salient sections."""

    def __init__(self) -> None:
        self.section_pattern = re.compile(r"(?:^|\n\n)([^:\n]{1,40}:.*?)\n\n", re.DOTALL)

    def propose(self, candidates: Iterable[Candidate]) -> List[Candidate]:
        """Produce merged candidates by reusing noteworthy sections."""
        candidates = [cand for cand in candidates if cand.text.strip()]
        if len(candidates) < 2:
            return []
        sections = [self._extract_sections(cand.text) for cand in candidates]
        merged_sections = self._best_sections(sections)
        if not merged_sections:
            return []
        merged_text = "\n\n".join(merged_sections)
        meta = {
            "merge_parents": [cand.fingerprint for cand in candidates],
            "merge_of": [cand.meta for cand in candidates],
        }
        merged_candidate = Candidate(text=merged_text, meta=meta)
        return [merged_candidate]

    def _extract_sections(self, text: str) -> List[str]:
        matches = self.section_pattern.findall(text + "\n\n")
        return matches or [text]

    def _best_sections(self, section_groups: List[List[str]]) -> List[str]:
        seen = set()
        selected = []
        for sections in section_groups:
            for section in sections:
                key = section.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                selected.append(section.strip())
                if len(selected) >= 6:
                    return selected
        return selected
