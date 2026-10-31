"""Elapsed-time stopping must not depend on system clock corrections."""

from unittest.mock import Mock

import pytest

from gepa.core.state import GEPAState
from gepa.utils import stop_condition


@pytest.mark.parametrize("wall_clock_jump,elapsed,expected", [(3600.0, 5.0, False), (-3600.0, 11.0, True)])
def test_timeout_ignores_wall_clock_adjustments(monkeypatch, wall_clock_jump, elapsed, expected):
    clocks = {"wall": 1_000_000.0, "elapsed": 100.0}
    monkeypatch.setattr(stop_condition.time, "time", lambda: clocks["wall"])
    monkeypatch.setattr(stop_condition.time, "monotonic", lambda: clocks["elapsed"])
    stopper = stop_condition.TimeoutStopCondition(10.0)
    clocks["wall"] += wall_clock_jump
    clocks["elapsed"] += elapsed
    assert stopper(Mock(spec=GEPAState)) is expected


@pytest.mark.parametrize("elapsed,expected", [(0.0, False), (9.0, False), (10.0, False), (10.5, True)])
def test_timeout_preserves_elapsed_time_boundary(monkeypatch, elapsed, expected):
    clock = [100.0]
    monkeypatch.setattr(stop_condition.time, "time", lambda: clock[0])
    monkeypatch.setattr(stop_condition.time, "monotonic", lambda: clock[0])
    stopper = stop_condition.TimeoutStopCondition(10.0)
    clock[0] += elapsed
    assert stopper(Mock(spec=GEPAState)) is expected
