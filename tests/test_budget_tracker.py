# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Unit tests for the eval-budget ledger (:class:`gepa.oa.budget.BudgetTracker`).

Contract: ``check()`` gates new work and raises at the cap; ``record()`` books
work that already happened and never raises, so a call that crosses the cap
is still counted. ``enforce=False`` turns the tracker into a pure ledger for
engines that stop themselves at iteration boundaries (issue #448).
"""

import pytest

from gepa.oa.budget import BudgetExhausted, BudgetTracker


def test_check_raises_at_cap_and_record_never_does():
    budget = BudgetTracker(max_evals=2)
    budget.check()
    budget.record(0.5)
    budget.check()
    budget.record(0.5)
    with pytest.raises(BudgetExhausted, match="2/2 used"):
        budget.check()
    # Work already done is booked even past the cap.
    budget.record(1.0)
    assert budget.used == 3
    assert budget.remaining == 0
    assert budget.exhausted


def test_check_needed_requires_room_for_the_whole_group():
    budget = BudgetTracker(max_evals=3)
    budget.check(needed=3)
    with pytest.raises(BudgetExhausted, match="3 remaining, 4 needed"):
        budget.check(needed=4)
    budget.record(0.0)
    with pytest.raises(BudgetExhausted, match="2 remaining, 3 needed"):
        budget.check(needed=3)


def test_unlimited_budget_never_raises():
    budget = BudgetTracker(max_evals=None)
    for _ in range(5):
        budget.check(needed=100)
        budget.record(0.0)
    assert budget.used == 5
    assert budget.remaining is None
    assert budget.status() == {"exhausted": False}


def test_enforce_false_keeps_the_ledger_but_stops_gating():
    budget = BudgetTracker(max_evals=1, enforce=False)
    budget.record(0.0)
    budget.record(0.0)
    budget.check()
    budget.check(needed=10)
    assert budget.used == 2
    assert budget.exhausted
    assert budget.status() == {"exhausted": True, "max_evals": 1, "used": 2, "remaining_evals": 0}
