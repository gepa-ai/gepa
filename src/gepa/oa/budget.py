"""In-process eval-budget enforcement.

The :class:`BudgetTracker` is the **eval ledger**: it counts (and caps) calls
to the eval function and nothing else. It lives inside the optimize_anything
process and wraps the real eval function. Engines — including external
black-box ones — can only evaluate through the eval server, so they cannot
modify the counter.

The cap is a strict ceiling, enforced with a reservation: callers
:meth:`BudgetTracker.reserve` the evals they are about to run — one call, or a
whole stage in bulk — and :meth:`BudgetTracker.commit` each one after it
finishes. The gate and the allocation are a single step under the lock, so an
unaffordable request is refused before anything is spent and concurrent
callers can never jointly exceed the cap.

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
    """Thread-safe eval-call budget: reserve before running, commit after.

    Args:
        max_evals: Maximum number of evaluation calls allowed. ``None`` means
            unlimited eval calls — valid for proposer-cost-only runs, where the
            run is instead bounded by the engine's ``max_token_cost`` cap.
    """

    max_evals: int | None = None

    _used: int = field(default=0, init=False, repr=False)
    _reserved: int = field(default=0, init=False, repr=False)
    _refused: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def reserve(self, n: int = 1) -> None:
        """Atomically set aside ``n`` eval calls, or raise :class:`BudgetExhausted`.

        Nothing has been spent when this raises. A successful reservation must
        be followed by exactly ``n`` :meth:`commit` calls, success or failure.
        """
        if n < 0:
            raise ValueError(f"reserve() needs a non-negative count, got {n}")
        with self._lock:
            if self.max_evals is not None and self._used + self._reserved + n > self.max_evals:
                self._refused += 1
                available = max(0, self.max_evals - self._used - self._reserved)
                raise BudgetExhausted(
                    f"Eval budget exhausted: {n} requested, {available} of {self.max_evals} available "
                    f"({self._used} used, {self._reserved} reserved)"
                )
            self._reserved += n

    def commit(self, score: float) -> None:
        """Book one reserved eval call as spent. Failed attempts commit too (as 0.0)."""
        with self._lock:
            if self._reserved <= 0:
                raise RuntimeError("commit() without a matching reserve()")
            self._reserved -= 1
            self._used += 1
            self._log.append({"eval": self._used, "score": score, "time": time.time()})

    @property
    def used(self) -> int:
        return self._used

    @property
    def reserved(self) -> int:
        """Evals reserved but not yet committed (in flight)."""
        return self._reserved

    @property
    def refused(self) -> int:
        """Number of reservations refused so far — the stop signal for in-process engines."""
        return self._refused

    @property
    def remaining(self) -> int | None:
        """Evals still available to reserve (``None`` when unlimited)."""
        if self.max_evals is None:
            return None
        return max(0, self.max_evals - self._used - self._reserved)

    @property
    def exhausted(self) -> bool:
        return self.max_evals is not None and self.remaining == 0

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {"exhausted": self.exhausted}
        if self.max_evals is not None:
            result["max_evals"] = self.max_evals
            result["used"] = self._used
            result["reserved"] = self._reserved
            result["remaining_evals"] = self.remaining
        return result
