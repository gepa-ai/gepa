# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the budget-floor warning and post-run summary added in #375.

These tests don't actually run optimization — they unit-test the helper
functions and assert that they emit ``GEPABudgetWarning`` under the right
conditions. End-to-end integration through ``gepa.optimize`` is covered by
the existing suite; we only need to verify the new pre/post-flight checks
fire as expected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pytest

from gepa import GEPABudgetWarning
from gepa.api import (
    _MIN_RECOMMENDED_PROPOSALS,
    _budget_floor,
    _safe_loader_len,
    _warn_if_budget_under_floor,
    _warn_if_too_few_accepted,
)


class _FakeLoader:
    """Stand-in for a DataLoader that just exposes ``__len__``."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


class _StreamingLoader:
    """A loader with no ``__len__`` — should be tolerated silently."""

    def __iter__(self):  # pragma: no cover - never iterated in tests
        yield


@dataclass
class _FakeResult:
    """Minimal stand-in for GEPAResult exposing the fields the summary reads."""

    num_candidates: int
    total_metric_calls: int | None


# ---------------------------------------------------------------------------
# _budget_floor
# ---------------------------------------------------------------------------


class TestBudgetFloor:
    def test_default_minibatch_uses_3(self):
        # 10-example valset, default minibatch=3, 3 proposals:
        # 10 (baseline) + 3 * (3 + 10) = 49
        assert _budget_floor(valset_len=10, reflection_minibatch_size=None) == 49

    def test_explicit_minibatch(self):
        # 10-example valset, minibatch=5, 3 proposals:
        # 10 + 3 * (5 + 10) = 55
        assert _budget_floor(valset_len=10, reflection_minibatch_size=5) == 55

    def test_scales_with_min_proposals(self):
        assert _budget_floor(10, 3, min_proposals=10) == 10 + 10 * (3 + 10)


# ---------------------------------------------------------------------------
# _safe_loader_len
# ---------------------------------------------------------------------------


class TestSafeLoaderLen:
    def test_returns_length_when_available(self):
        assert _safe_loader_len(_FakeLoader(42)) == 42

    def test_returns_none_for_streaming_loader(self):
        assert _safe_loader_len(_StreamingLoader()) is None

    def test_returns_none_for_object_without_len(self):
        assert _safe_loader_len(object()) is None


# ---------------------------------------------------------------------------
# _warn_if_budget_under_floor
# ---------------------------------------------------------------------------


class TestPreflightWarning:
    def test_warns_when_below_floor(self):
        # Reproduce the reported incident: valset=6, mb defaulted to 3,
        # budget=8. Floor would be 6 + 3*(3+6) = 33.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_budget_under_floor(
                max_metric_calls=8,
                val_loader=_FakeLoader(6),
                reflection_minibatch_size=None,
            )
        budget_warnings = [w for w in caught if issubclass(w.category, GEPABudgetWarning)]
        assert len(budget_warnings) == 1
        msg = str(budget_warnings[0].message)
        assert "max_metric_calls=8" in msg
        assert "33" in msg  # recommended floor for this configuration
        assert "https://gepa-ai.github.io/gepa/guides/budget/" in msg

    def test_silent_when_at_or_above_floor(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_budget_under_floor(
                max_metric_calls=100,
                val_loader=_FakeLoader(6),
                reflection_minibatch_size=None,
            )
        assert not any(issubclass(w.category, GEPABudgetWarning) for w in caught)

    def test_silent_when_max_metric_calls_is_none(self):
        # User opted into stop_callbacks instead — don't pretend we know better.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_budget_under_floor(
                max_metric_calls=None,
                val_loader=_FakeLoader(6),
                reflection_minibatch_size=None,
            )
        assert not any(issubclass(w.category, GEPABudgetWarning) for w in caught)

    def test_silent_when_loader_length_unknown(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_budget_under_floor(
                max_metric_calls=1,
                val_loader=_StreamingLoader(),
                reflection_minibatch_size=None,
            )
        assert not any(issubclass(w.category, GEPABudgetWarning) for w in caught)

    def test_silent_when_valset_is_empty(self):
        # Avoid divide-by-zero shaped recommendations on a 0-length loader.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_budget_under_floor(
                max_metric_calls=1,
                val_loader=_FakeLoader(0),
                reflection_minibatch_size=None,
            )
        assert not any(issubclass(w.category, GEPABudgetWarning) for w in caught)


# ---------------------------------------------------------------------------
# _warn_if_too_few_accepted
# ---------------------------------------------------------------------------


class TestPostRunSummary:
    def test_warns_when_only_baseline(self):
        # num_candidates == 1 means GEPA accepted zero proposals.
        result: Any = _FakeResult(num_candidates=1, total_metric_calls=8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_too_few_accepted(result, logger=None)
        budget_warnings = [w for w in caught if issubclass(w.category, GEPABudgetWarning)]
        assert len(budget_warnings) == 1
        assert "0 accepted proposal" in str(budget_warnings[0].message)

    def test_warns_at_the_incident_config(self):
        # Reproduce the X-thread incident: baseline + 1 accepted (num_candidates=2).
        result: Any = _FakeResult(num_candidates=2, total_metric_calls=8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_too_few_accepted(result, logger=None)
        budget_warnings = [w for w in caught if issubclass(w.category, GEPABudgetWarning)]
        assert len(budget_warnings) == 1
        assert "1 accepted proposal" in str(budget_warnings[0].message)
        assert "under-budgeted" in str(budget_warnings[0].message)

    def test_silent_when_enough_proposals_accepted(self):
        result: Any = _FakeResult(
            num_candidates=1 + _MIN_RECOMMENDED_PROPOSALS, total_metric_calls=200
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_too_few_accepted(result, logger=None)
        assert not any(issubclass(w.category, GEPABudgetWarning) for w in caught)

    def test_calls_logger_with_summary_line(self):
        calls: list[str] = []

        class FakeLogger:
            def log(self, msg: str) -> None:
                calls.append(msg)

        result: Any = _FakeResult(num_candidates=5, total_metric_calls=150)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _warn_if_too_few_accepted(result, logger=FakeLogger())
        assert len(calls) == 1
        assert "GEPA finished:" in calls[0]
        assert "4 proposal(s) accepted" in calls[0]
        assert "150 metric call" in calls[0]

    def test_logger_exception_does_not_break_summary(self):
        class BrokenLogger:
            def log(self, msg: str) -> None:
                raise RuntimeError("logger blew up")

        result: Any = _FakeResult(num_candidates=5, total_metric_calls=150)
        # Should not raise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _warn_if_too_few_accepted(result, logger=BrokenLogger())


# ---------------------------------------------------------------------------
# Public re-export
# ---------------------------------------------------------------------------


def test_gepa_budget_warning_is_exported():
    """Users should be able to silence the warning via ``gepa.GEPABudgetWarning``."""
    import gepa

    assert gepa.GEPABudgetWarning is GEPABudgetWarning
    assert issubclass(GEPABudgetWarning, UserWarning)


def test_warning_can_be_silenced(monkeypatch):
    """Confirm the warning respects the standard ``warnings`` filter mechanism."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", GEPABudgetWarning)
        with pytest.raises(GEPABudgetWarning):
            _warn_if_budget_under_floor(
                max_metric_calls=1,
                val_loader=_FakeLoader(6),
                reflection_minibatch_size=None,
            )
    # And the inverse: can be globally suppressed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", GEPABudgetWarning)
        _warn_if_budget_under_floor(
            max_metric_calls=1,
            val_loader=_FakeLoader(6),
            reflection_minibatch_size=None,
        )
    assert not any(issubclass(w.category, GEPABudgetWarning) for w in caught)
