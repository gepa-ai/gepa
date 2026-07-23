"""Regression tests for dict (multi-component) seed candidates.

``optimize_anything(seed_candidate=...)`` and :class:`Task` accept a
``{component: text}`` dict as a v0.1.x parity feature. Only the ``gepa`` engine
co-optimizes the components; every other engine treats the seed as one text via
:func:`gepa.oa.task.seed_as_text`. These tests pin that a dict seed flows
through the shared plumbing (the eval server's tracked best) and the text-only
coercion without regressing to the ``str``-only assumptions.
"""

from __future__ import annotations

from gepa.oa.budget import BudgetTracker
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task, seed_as_text


def test_seed_as_text_none_str_and_dict() -> None:
    assert seed_as_text(None) == ""
    assert seed_as_text("hello") == "hello"
    # A multi-component dict is flattened to one text (component texts joined),
    # so text-only engines can seed a single-text run instead of crashing.
    assert seed_as_text({"a": "foo", "b": "bar"}) == "foo\n\nbar"


def test_eval_server_tracks_dict_seed_as_best_candidate() -> None:
    """A dict seed is a legal candidate; the server's best starts as the dict."""
    seed = {"system": "sys prompt", "user": "user prompt"}
    task = Task(name="dict-seed", seed_candidate=seed)
    server = EvalServer(task, lambda c: (0.0, {}), BudgetTracker(max_evals=1))
    assert server.best_candidate == seed
