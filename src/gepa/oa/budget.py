"""In-process eval-budget enforcement.

The :class:`BudgetTracker` is the **eval ledger**: it counts (and caps) calls
to the eval function and nothing else. It lives inside the optimize_anything
process and wraps the real eval function. Engines — including external
black-box ones — can only evaluate through the eval server, so they cannot
modify the counter.

The cap gates *new* work only: :meth:`BudgetTracker.check` is called before an
eval starts and raises :class:`BudgetExhausted` once the budget is used up,
while :meth:`BudgetTracker.record` books work that already happened and never
raises. A grouped call that crosses the cap is therefore recorded in full — the
user has already paid for every pair in it.

An in-process engine that installs its own iteration-boundary stopper (GEPA
core's ``MaxMetricCallsStopper``) may set ``enforce=False``: the ledger keeps
counting and ``status()`` keeps reporting the cap, but ``check()`` stops
raising, so the engine finishes the iteration in flight instead of aborting
mid-valset and discarding the candidate under evaluation. External engines
have no such stopper and keep the default hard enforcement.

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
        enforce: Whether :meth:`check` raises once the cap is reached. An
            engine that stops itself at iteration boundaries (the in-process
            GEPA engine, via core's ``MaxMetricCallsStopper``) sets this to
            ``False`` so the ledger only counts. ``used``/``status()`` are
            unaffected.
    """

    max_evals: int | None = None
    enforce: bool = True

    _used: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def record(self, score: float) -> None:
        """Book one completed eval call.

        Never raises: the call already happened and the user already paid for
        it, so it is counted even past the cap. Gate new work with
        :meth:`check` *before* evaluating.
        """
        with self._lock:
            self._used += 1
            self._log.append({"eval": self._used, "score": score, "time": time.time()})

    def check(self, needed: int = 1) -> None:
        """Raise BudgetExhausted unless ``needed`` more eval calls fit in the budget.

        A no-op when the budget is unlimited or ``enforce`` is ``False``.
        """
        if not self.enforce or self.max_evals is None:
            return
        remaining = self.max_evals - self._used
        if remaining <= 0:
            raise BudgetExhausted(f"Eval budget exhausted: {self._used}/{self.max_evals} used")
        if remaining < needed:
            raise BudgetExhausted(f"Not enough budget to evaluate all examples: {remaining} remaining, {needed} needed")

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
