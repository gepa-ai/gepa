# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Built-in session strategies.

A ``SessionStrategy`` has two hooks:

- ``select(ctx)`` — pick the session to use for the next mutation.
- ``observe(outcome)`` — return dict updates to merge into the manager's
  session store after the mutation completes.

Strategies are pure: all state changes flow through ``observe``'s return
value.  Default strategies bind newly-accepted candidates by index, so
the store doubles as a candidate → session map over time.
"""

from __future__ import annotations

import random as random_module
from collections.abc import Mapping

from gepa.core.session import Session
from gepa.core.session_manager import SessionContext, SessionEntry, SessionOutcome


def _most_recent(ctx: SessionContext) -> Session | None:
    """Return the most recently added session, or ``None`` if store is empty."""
    if not ctx.sessions:
        return None
    # dict preserves insertion order in Python 3.7+
    last_entry = list(ctx.sessions.values())[-1]
    return last_entry.session


def _bind_on_accept(outcome: SessionOutcome) -> Mapping[int | str, SessionEntry]:
    """Default observe: bind accepted candidates to their session."""
    if not outcome.accepted or outcome.candidate_idx is None:
        return {}
    return {outcome.candidate_idx: SessionEntry(outcome.session, outcome.val_score)}


class AlwaysFork:
    """Fork the most recent session -- copy history, diverge after.

    Creates an independent copy of the most recent session.  The parent
    stays in the store untouched.  The LLM sees the full conversation
    history and can build on what was tried before.
    """

    def select(self, ctx: SessionContext) -> Session:
        parent = _most_recent(ctx)
        return parent.fork() if parent is not None else ctx.create()

    def observe(self, outcome: SessionOutcome) -> Mapping[int | str, SessionEntry]:
        return _bind_on_accept(outcome)


class AlwaysReset:
    """Reset from the most recent session -- no history.

    Creates a new session with the same backend config but no conversation
    history.  Equivalent to stateless reflection.  Each mutation starts
    with a clean slate.
    """

    def select(self, ctx: SessionContext) -> Session:
        parent = _most_recent(ctx)
        return parent.reset() if parent is not None else ctx.create()

    def observe(self, outcome: SessionOutcome) -> Mapping[int | str, SessionEntry]:
        return _bind_on_accept(outcome)


class RandomStrategy:
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

    def observe(self, outcome: SessionOutcome) -> Mapping[int | str, SessionEntry]:
        return _bind_on_accept(outcome)


class RoundRobin:
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

    def observe(self, outcome: SessionOutcome) -> Mapping[int | str, SessionEntry]:
        return _bind_on_accept(outcome)


class ParentLinked:
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

    def observe(self, outcome: SessionOutcome) -> Mapping[int | str, SessionEntry]:
        return _bind_on_accept(outcome)
