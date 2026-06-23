"""Task definition for the optimize_anything API.

A :class:`Task` is **pure data** describing *what* to optimize: a name, an
initial candidate, optional natural-language ``objective`` / ``background``
for engine prompts, and optional ``train``/``val``/``test`` splits. Dataset
items are opaque — any Python object the user's evaluator understands. If an
item has a stable ``id`` (attribute or ``"id"`` mapping key), the eval server
uses it; otherwise the server assigns one.

The eval *function* — how to score a candidate — is **not** part of Task. It
is passed separately to :func:`gepa.optimize_anything.optimize_anything` as a
required ``evaluate`` parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

EvalFn = Callable[..., tuple[float, dict[str, Any]]]
"""Eval function signature.

- Single-task:  ``(candidate: str) -> (score, info)``
- Dataset task: ``(candidate: str, example: Any) -> (score, info)``
  ``example`` is whatever the caller put in ``train_set`` / ``val_set`` / ``test_set``.
"""


@dataclass
class Task:
    """A task definition for the optimize_anything API.

    Attributes:
        name: Unique identifier (e.g. ``"circle_packing"``).
        initial_candidate: Seed text to evolve from.
        objective: Short goal statement (e.g. "Maximize sum of circle radii").
            Surfaced verbatim by every engine as the optimization goal.
        background: Long-form context — problem statement, evaluation rules,
            domain notes. Surfaced verbatim by every engine.
        train_set: Training examples (used for optimization). Items are
            opaque — any object the evaluator understands.
        val_set: Validation examples (used for candidate selection).
        test_set: Held-out test examples (evaluated outside the budget).
    """

    name: str
    initial_candidate: str
    objective: str = ""
    background: str = ""
    train_set: list[Any] | None = None
    val_set: list[Any] | None = None
    test_set: list[Any] | None = None

    @property
    def has_dataset(self) -> bool:
        return self.train_set is not None or self.val_set is not None or self.test_set is not None
