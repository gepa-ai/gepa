"""In-process budget enforcement.

The :class:`BudgetTracker` lives inside the omni process and wraps the
real eval function. Backends — including external black-box ones — can only
evaluate through the eval server, so they cannot modify the counter.

Two independent limits are supported — ``max_evals`` (number of eval calls)
and ``max_token_cost`` (cumulative USD spent on backend LLM tokens). At least
one must be specified.

``max_evals`` is enforced centrally by the ``BudgetTracker`` (every eval goes
through it). ``max_token_cost`` is enforced by each backend in its own way
(GEPA via ``max_reflection_cost``, Claude Code via ``--max-budget-usd``).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(Exception):  # noqa: N818 — load-bearing public name
    """Raised when the eval budget has been used up."""


@dataclass
class BudgetTracker:
    """Thread-safe, in-process eval budget enforcer.

    Args:
        max_evals: Maximum number of evaluation calls allowed. ``None`` means
            unlimited (token-cost-only mode).
        max_token_cost: Maximum cumulative USD spent on backend LLM tokens.
            ``None`` means unlimited (eval-count-only mode). Enforcement is
            backend-side; the tracker holds the config value for propagation.

    Raises:
        ValueError: If both ``max_evals`` and ``max_token_cost`` are ``None``.
    """

    max_evals: int | None = None
    max_token_cost: float | None = None

    _used: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_evals is None and self.max_token_cost is None:
            raise ValueError("At least one of max_evals or max_token_cost must be specified")

    def record(self, score: float) -> None:
        """Record one eval call. Raises BudgetExhausted if over eval limit."""
        with self._lock:
            if self.max_evals is not None and self._used >= self.max_evals:
                raise BudgetExhausted(f"Eval budget exhausted: {self._used}/{self.max_evals} used")
            self._used += 1
            self._log.append({"eval": self._used, "score": score, "time": time.time()})

    def check(self) -> None:
        """Raise BudgetExhausted if the eval budget is used up."""
        if self.max_evals is not None and self._used >= self.max_evals:
            raise BudgetExhausted(f"Eval budget exhausted: {self._used}/{self.max_evals} used")

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int | None:
        if self.max_evals is None:
            return None
        return max(0, self.max_evals - self._used)

    @property
    def exhausted(self) -> bool:
        if self.max_evals is not None and self._used >= self.max_evals:
            return True
        return False

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {"exhausted": self.exhausted}
        if self.max_evals is not None:
            result["max_evals"] = self.max_evals
            result["used"] = self._used
            result["remaining_evals"] = self.remaining
        if self.max_token_cost is not None:
            result["max_token_cost"] = self.max_token_cost
        return result
