"""Task and Example definitions for the omni API.

A :class:`Task` is **pure data** describing *what* to optimize: a name, an
initial candidate, optional natural-language ``objective`` / ``background``
for backend prompts, and optional ``train``/``val``/``test`` splits. Every
field is YAML-friendly so a Task can be instantiated directly from a Hydra
config (``_target_: gepa.omni.task.Task``) and individual fields like
``objective`` overridden from the CLI.

The eval *function* — how to score a candidate — is **not** part of Task. It
is passed separately to :func:`gepa.omni.optimize_anything` as a required
``evaluate`` parameter. This split keeps code (the function) in Python and
data (the spec) in config, and avoids the "Task carries a callable that may
or may not match the api's signature" ambiguity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

EvalFn = Callable[..., tuple[float, dict[str, Any]]]
"""Eval function signature.

- Single-task:  ``(candidate: str) -> (score, info)``
- Dataset task: ``(candidate: str, example: Example) -> (score, info)``
"""


@dataclass
class Example:
    """A single dataset example for train/val/test sets."""

    id: str
    inputs: dict[str, Any]
    expected: Any | None = None


@dataclass
class Task:
    """A task definition for the omni API.

    Attributes:
        name: Unique identifier (e.g. ``"circle_packing"``).
        initial_candidate: Seed text to evolve from.
        objective: Short goal statement (e.g. "Maximize sum of circle radii").
            Surfaced verbatim by every backend as the optimization goal.
        background: Long-form context — problem statement, evaluation rules,
            domain notes. Surfaced verbatim by every backend.
        train_set: Training examples (used for optimization).
        val_set: Validation examples (used for candidate selection).
        test_set: Held-out test examples (evaluated outside the budget).
        metadata: Extra typed context (run mode, candidate type, etc.).
    """

    name: str
    initial_candidate: str
    objective: str = ""
    background: str = ""
    train_set: list[Example] | None = None
    val_set: list[Example] | None = None
    test_set: list[Example] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_dataset(self) -> bool:
        return self.train_set is not None or self.val_set is not None or self.test_set is not None
