"""Persistent append-only reflection journal and note updater for CC agentic mode."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gepa.proposer.reflective_mutation.base import LanguageModel


class GlobalNote:
    """Append-only, thread-safe reflection journal stored at run_dir/note.md."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> str:
        return self.path.read_text() if self.path.exists() else ""

    def append(self, entry: str) -> None:
        with self._lock:
            with self.path.open("a") as f:
                f.write("\n" + entry.strip() + "\n")


class NoteUpdater:
    """Uses CC (via cc_lm) to generate and append one reflection entry per iteration."""

    def __init__(self, global_note: GlobalNote, cc_lm: LanguageModel) -> None:
        self.global_note = global_note
        self.cc_lm = cc_lm

    def update(
        self,
        iteration: int,
        parent_candidate: str,
        child_candidate: str,
        parent_scores: list[float],
        child_scores: list[float],
        parent_outputs: list[Any],
        child_outputs: list[Any],
        objective: str | None = None,
    ) -> None:
        current_note = self.global_note.read()
        prompt = _build_note_update_prompt(
            current_note=current_note,
            iteration=iteration,
            parent_candidate=parent_candidate,
            child_candidate=child_candidate,
            parent_scores=parent_scores,
            child_scores=child_scores,
            parent_outputs=parent_outputs,
            child_outputs=child_outputs,
            objective=objective,
        )
        entry = self.cc_lm(prompt)
        self.global_note.append(entry)


def _build_note_update_prompt(
    current_note: str,
    iteration: int,
    parent_candidate: str,
    child_candidate: str,
    parent_scores: list[float],
    child_scores: list[float],
    parent_outputs: list[Any],
    child_outputs: list[Any],
    objective: str | None,
) -> str:
    parent_avg = sum(parent_scores) / len(parent_scores) if parent_scores else 0.0
    child_avg = sum(child_scores) / len(child_scores) if child_scores else 0.0
    delta = child_avg - parent_avg
    direction = "improvement" if delta > 0 else "regression" if delta < 0 else "no change"

    return f"""\
You are updating a reflection journal for an optimization run.

## Current note
{current_note if current_note else "(empty — this is the first entry)"}

## Iteration {iteration}
Objective: {objective or "not specified"}

Parent score: {parent_avg:.4f}
Child score:  {child_avg:.4f}
Delta: {delta:+.4f} ({direction})

Parent candidate:
{parent_candidate[:1500]}{"..." if len(parent_candidate) > 1500 else ""}

Child candidate:
{child_candidate[:1500]}{"..." if len(child_candidate) > 1500 else ""}

Parent outputs: {str(parent_outputs)[:500]}
Child outputs:  {str(child_outputs)[:500]}

## Instructions
Append one concise reflection entry (3-5 sentences). Start with "## Iteration {iteration}".
Focus on:
- What the child changed vs the parent
- Whether it helped or hurt and why
- Any generalizable insight for future mutations
Output ONLY the new entry text. Do not repeat the existing note.
"""
