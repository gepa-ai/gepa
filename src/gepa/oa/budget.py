"""In-process eval-budget enforcement.

The :class:`BudgetTracker` is the **eval ledger**: it counts (and caps) calls
to the eval function and nothing else. It lives inside the optimize_anything
process and wraps the real eval function. Engines — including external
black-box ones — can only evaluate through the eval server, so they cannot
modify the counter.

It does **not** track proposer cost. The dollars an optimizer spends *thinking
up* candidates (GEPA's reflection LM, a ``claude --print`` subprocess) are
spent out-of-band — the eval server never sees them, and for a subprocess
there is nothing to observe until the process exits. That cap is
:attr:`OptimizeAnythingConfig.max_token_cost`, read by each engine at
construction and enforced engine-side (GEPA via ``max_reflection_cost``, Claude
engines via ``--max-budget-usd``). See :class:`gepa.oa.engine.Engine`.

So the two budgets have two distinct owners:

- **eval budget** (call count, here) — enforced centrally by this tracker.
- **proposer budget** (USD on optimizer LLM tokens) — owned by the engine.

The api's final report sums them (``server.total_cost`` for eval-side cost +
the engine's reported ``adapter_cost``) but they are never conflated here.
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
    """Thread-safe, in-process eval-call budget enforcer.

    Args:
        max_evals: Maximum number of evaluation calls allowed. ``None`` means
            unlimited eval calls — valid for proposer-cost-only runs, where the
            run is instead bounded by the engine's ``max_token_cost`` cap.
    """

    max_evals: int | None = None

    _used: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

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
        return self.max_evals is not None and self._used >= self.max_evals

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {"exhausted": self.exhausted}
        if self.max_evals is not None:
            result["max_evals"] = self.max_evals
            result["used"] = self._used
            result["remaining_evals"] = self.remaining
        return result
