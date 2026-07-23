"""Regression tests for dict (multi-component) seed candidates.

``optimize_anything(seed_candidate=...)`` and :class:`Task` accept a
``{component: text}`` dict as a v0.1.x parity feature. Only the ``gepa`` engine
co-optimizes the components; every other engine treats the seed as one text via
:func:`gepa.oa.task.seed_as_text`. These tests pin that a dict seed flows
through the shared plumbing (the eval server's tracked best) and the text-only
coercion without regressing to the ``str``-only assumptions.
"""

from __future__ import annotations

import pytest

from gepa.oa.budget import BudgetTracker
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task, seed_as_text


def test_seed_as_text_none_and_str() -> None:
    assert seed_as_text(None) == ""
    assert seed_as_text("hello") == "hello"


def test_seed_as_text_rejects_dict() -> None:
    # Text-only engines cannot honor a multi-component dict seed (they optimize
    # one string), so a dict is refused loudly rather than silently flattened.
    with pytest.raises(TypeError, match="engine='gepa'"):
        seed_as_text({"a": "foo", "b": "bar"})


def test_eval_server_tracks_dict_seed_as_best_candidate() -> None:
    """A dict seed is a legal candidate; the server's best starts as the dict."""
    seed = {"system": "sys prompt", "user": "user prompt"}
    task = Task(name="dict-seed", seed_candidate=seed)
    server = EvalServer(task, lambda c: (0.0, {}), BudgetTracker(max_evals=1))
    assert server.best_candidate == seed
