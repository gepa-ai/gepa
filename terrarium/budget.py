"""In-process budget enforcement.

The BudgetTracker lives inside the terrarium process and wraps the real eval
function. Evolution systems (including external black-box ones) can only evaluate
through terrarium — they cannot modify the counter.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(Exception):
    """Raised when the eval budget has been used up."""


@dataclass
class BudgetTracker:
    """Thread-safe, in-process eval budget enforcer.

    This is NOT exposed to evolution systems. Terrarium wraps the task's
    eval_fn with this tracker, so every call is counted and hard-limited.
    """

    max_evals: int
    _used: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def record(self, score: float) -> None:
        """Record one eval call. Raises BudgetExhausted if over limit."""
        with self._lock:
            if self._used >= self.max_evals:
                raise BudgetExhausted(f"Budget exhausted: {self.max_evals}/{self.max_evals} used")
            self._used += 1
            self._log.append({"eval": self._used, "score": score, "time": time.time()})

    def check(self) -> None:
        """Raise BudgetExhausted if budget is used up."""
        if self._used >= self.max_evals:
            raise BudgetExhausted(f"Budget exhausted: {self.max_evals}/{self.max_evals} used")

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self._used)

    @property
    def exhausted(self) -> bool:
        return self._used >= self.max_evals

    def status(self) -> dict[str, Any]:
        return {
            "max_evals": self.max_evals,
            "used": self._used,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
        }
