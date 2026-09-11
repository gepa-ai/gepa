"""Tests for the low-budget warning surfaced by ``optimize`` (see issue #375).

``max_metric_calls`` is the only knob that controls how many proposals GEPA attempts, and an
under-budgeted run stops after one or two proposals while still returning an improved candidate.
These tests cover the heuristic that warns loudly in that case.
"""

import re
import warnings
from unittest.mock import Mock

import pytest

from gepa import optimize
from gepa.api import _warn_if_budget_too_low


def _catch(max_metric_calls, valset_size, minibatch_size, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_budget_too_low(max_metric_calls, valset_size, minibatch_size, **kwargs)
    return [w for w in caught if issubclass(w.category, UserWarning)]


def test_warns_when_budget_below_floor():
    # valset=10, minibatch=3 -> floor = 10 + 3*(3+10) = 49. Budget of 8 is far below.
    warned = _catch(8, 10, 3)
    assert len(warned) == 1
    msg = str(warned[0].message)
    assert "max_metric_calls=8" in msg
    assert "at least 49" in msg


def test_no_warning_when_budget_at_or_above_floor():
    # floor = 49; exactly at the floor and above must stay silent.
    assert _catch(49, 10, 3) == []
    assert _catch(150, 10, 3) == []


def test_reports_supported_proposal_count():
    # valset=6, minibatch=2, budget=8 -> per_proposal = 8, baseline = 6.
    # supported = (8 - 6) // 8 = 0 proposals.
    warned = _catch(8, 6, 2)
    assert len(warned) == 1
    assert "only ~0 proposal" in str(warned[0].message)


def test_min_proposals_parameter_moves_the_floor():
    # floor = valset + min_proposals*(minibatch+valset) = 10 + 1*13 = 23.
    assert _catch(30, 10, 3, min_proposals=1) == []
    assert len(_catch(22, 10, 3, min_proposals=1)) == 1


def test_silent_when_sizes_unknown():
    # A streaming loader with no stable length or a custom sampler without a minibatch size
    # should not produce a spurious warning.
    assert _catch(1, 0, 3) == []
    assert _catch(1, 10, 0) == []


def test_optimize_emits_warning_for_low_budget():
    """End-to-end: the warning is wired into ``optimize`` and fires for a tiny budget."""
    mock_data = [{"input": "my_input", "answer": "my_answer", "additional_context": {"context": "ctx"}}]

    task_lm = Mock(return_value="test response")

    def mock_reflection_lm(prompt):
        return "```\nimproved instructions\n```"

    with pytest.warns(UserWarning, match=re.compile(r"max_metric_calls=2 is likely too low")):
        optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )
