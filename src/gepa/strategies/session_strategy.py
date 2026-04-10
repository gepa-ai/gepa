# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Built-in session strategies.

A ``SessionStrategy`` has one hook:

- ``select(ctx)`` — pick the session to use for the next mutation.

Bookkeeping (storing accepted candidates, tracking scores) is handled
by ``SessionManager.observe()`` — strategies only decide *which*
session to use.
"""

from __future__ import annotations

import random as random_module

from gepa.core.session import Session
from gepa.core.session_manager import SessionContext, SessionStrategy


def _most_recent(ctx: SessionContext) -> Session | None:
    """Return the most recently added session, or ``None`` if store is empty."""
    if not ctx.sessions:
        return None
    # dict preserves insertion order in Python 3.7+
    last_entry = list(ctx.sessions.values())[-1]
    return last_entry.session


class AlwaysFork(SessionStrategy):
    """Fork the most recent session -- copy history, diverge after.

    Creates an independent copy of the most recent session.  The parent
    stays in the store untouched.  The LLM sees the full conversation
    history and can build on what was tried before.
    """

    def select(self, ctx: SessionContext) -> Session:
        parent = _most_recent(ctx)
        return parent.fork() if parent is not None else ctx.create()


class AlwaysReset(SessionStrategy):
    """Reset from the most recent session -- no history.

    Creates a new session with the same backend config but no conversation
    history.  Equivalent to stateless reflection.  Each mutation starts
    with a clean slate.
    """

    def select(self, ctx: SessionContext) -> Session:
        parent = _most_recent(ctx)
        return parent.reset() if parent is not None else ctx.create()


class RandomStrategy(SessionStrategy):
    """Randomly fork or reset from the most recent session.

    Parameters
    ----------
    fork_probability:
        Probability of forking vs resetting.  Default 0.5.
    """

    def __init__(self, fork_probability: float = 0.5) -> None:
        self._fork_prob = fork_probability

    def select(self, ctx: SessionContext) -> Session:
        parent = _most_recent(ctx)
        if parent is None:
            return ctx.create()
        if random_module.random() < self._fork_prob:
            return parent.fork()
        return parent.reset()


class RoundRobin(SessionStrategy):
    """Alternate between fork and reset.  Starts with fork."""

    def __init__(self) -> None:
        self._counter = 0

    def select(self, ctx: SessionContext) -> Session:
        parent = _most_recent(ctx)
        if parent is None:
            return ctx.create()
        self._counter += 1
        if self._counter % 2 == 1:
            return parent.fork()
        return parent.reset()


class ParentLinked(SessionStrategy):
    """Fork from the parent candidate's session -- lineage-aware.

    The session store doubles as a candidate → session map: each accepted
    candidate is bound to the session that produced it.  When mutating
    candidate P, fork ``ctx.sessions[P]`` so the child's session descends
    from its parent's lineage.

    Re-visiting the same parent forks its session independently, so
    sibling mutations don't interfere.  The candidate tree and session
    tree stay isomorphic.
    """

    def select(self, ctx: SessionContext) -> Session:
        if ctx.parent_candidate_idx is not None:
            entry = ctx.sessions.get(ctx.parent_candidate_idx)
            if entry is not None:
                return entry.session.fork()
        # Fallback: no parent bound yet (root iteration or unknown parent)
        parent = _most_recent(ctx)
        return parent.fork() if parent is not None else ctx.create()


class BestScore(SessionStrategy):
    """Fork from the session with the highest validation score.

    Picks the session whose candidate scored best on the val set and
    forks it.  Falls back to the most recent session if no scores are
    available yet (first iteration).
    """

    def select(self, ctx: SessionContext) -> Session:
        scored = [(k, e) for k, e in ctx.sessions.items() if e.val_score is not None]
        if scored:
            _, best_entry = max(scored, key=lambda x: x[1].val_score)  # type: ignore[arg-type]
            return best_entry.session.fork()
        # No scores yet — fall back to most recent
        parent = _most_recent(ctx)
        return parent.fork() if parent is not None else ctx.create()
