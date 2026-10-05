# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Unit tests for the eval-budget reservation ledger (:class:`gepa.oa.budget.BudgetTracker`).

Contract: ``reserve(n)`` atomically sets aside ``n`` evals or raises before
anything is spent; ``commit(score)`` books one reserved eval as used. The cap
is therefore a strict ceiling under any concurrency (issue #448).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gepa.oa.budget import BudgetExhausted, BudgetTracker


def test_reserve_then_commit_moves_evals_from_reserved_to_used():
    budget = BudgetTracker(max_evals=3)
    budget.reserve(2)
    assert (budget.used, budget.reserved, budget.remaining) == (0, 2, 1)
    budget.commit(0.5)
    budget.commit(0.0)  # a failed attempt commits too
    assert (budget.used, budget.reserved, budget.remaining) == (2, 0, 1)
    assert not budget.exhausted


def test_unaffordable_reservation_is_refused_before_any_spend():
    budget = BudgetTracker(max_evals=3)
    budget.reserve(2)
    with pytest.raises(BudgetExhausted, match="2 requested, 1 of 3 available"):
        budget.reserve(2)  # bulk request larger than what is left
    assert (budget.used, budget.reserved, budget.refused) == (0, 2, 1)
    budget.reserve(1)  # the single eval that does fit
    assert budget.exhausted
    with pytest.raises(BudgetExhausted):
        budget.reserve(1)
    assert budget.refused == 2


def test_in_flight_reservations_count_against_the_cap():
    budget = BudgetTracker(max_evals=2)
    budget.reserve(2)
    assert budget.used == 0
    assert budget.exhausted  # nothing committed yet, but nothing left to hand out
    with pytest.raises(BudgetExhausted):
        budget.reserve(1)


def test_commit_without_reservation_is_a_programming_error():
    budget = BudgetTracker(max_evals=2)
    with pytest.raises(RuntimeError, match="without a matching reserve"):
        budget.commit(1.0)


def test_unlimited_budget_never_refuses():
    budget = BudgetTracker(max_evals=None)
    budget.reserve(1000)
    for _ in range(1000):
        budget.commit(0.0)
    assert budget.used == 1000
    assert budget.remaining is None
    assert budget.status() == {"exhausted": False}


def test_status_reports_used_reserved_and_remaining():
    budget = BudgetTracker(max_evals=4)
    budget.reserve(3)
    budget.commit(1.0)
    assert budget.status() == {"exhausted": False, "max_evals": 4, "used": 1, "reserved": 2, "remaining_evals": 1}


def test_concurrent_reservations_never_exceed_the_cap():
    """N callers racing for the last eval: exactly one wins. The gate and the
    allocation are one step under the lock, so there is no check-then-act
    window to overshoot through."""
    n = 16
    budget = BudgetTracker(max_evals=5)
    budget.reserve(4)
    start = threading.Barrier(n)
    outcomes: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        start.wait(timeout=5)
        try:
            budget.reserve(1)
            ok = True
        except BudgetExhausted:
            ok = False
        with lock:
            outcomes.append(ok)

    with ThreadPoolExecutor(max_workers=n) as pool:
        for _ in range(n):
            pool.submit(attempt)

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == n - 1
    assert budget.reserved == 5
    assert budget.refused == n - 1
