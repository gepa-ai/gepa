# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Built-in session strategies for controlling session lifecycle."""

from __future__ import annotations

import random as random_module
from collections.abc import Callable

from gepa.core.session import Session


class AlwaysFork:
    """Fork the most recent session -- copy history, diverge after.

    Creates an independent copy of the most recent session.  The original
    stays in the pool untouched.  The LLM sees the full conversation
    history and can build on what was tried before.
    """

    def select(
        self,
        sessions: list[Session],
        create: Callable[[], Session],
    ) -> Session:
        return sessions[-1].fork() if sessions else create()


class AlwaysReset:
    """Reset from the most recent session -- no history.

    Creates a new session with the same backend config but no conversation
    history.  Equivalent to current GEPA behavior (stateless reflection LM).
    Each mutation starts with a clean slate.
    """

    def select(
        self,
        sessions: list[Session],
        create: Callable[[], Session],
    ) -> Session:
        return sessions[-1].reset() if sessions else create()


class RandomStrategy:
    """Randomly fork or reset a random session from the pool.

    Each iteration, picks a random session and randomly decides whether
    to fork it (keep history) or reset it (clean slate).

    Parameters
    ----------
    fork_probability:
        Probability of forking vs resetting.  Default 0.5.
    """

    def __init__(self, fork_probability: float = 0.5) -> None:
        self._fork_prob = fork_probability

    def select(
        self,
        sessions: list[Session],
        create: Callable[[], Session],
    ) -> Session:
        if not sessions:
            return create()
        session = random_module.choice(sessions)
        if random_module.random() < self._fork_prob:
            return session.fork()
        return session.reset()


class RoundRobin:
    """Alternate between fork and reset.

    Cycles through fork and reset on each call.  Starts with fork.
    """

    def __init__(self) -> None:
        self._counter = 0

    def select(
        self,
        sessions: list[Session],
        create: Callable[[], Session],
    ) -> Session:
        if not sessions:
            return create()
        session = sessions[-1]
        self._counter += 1
        if self._counter % 2 == 1:
            return session.fork()
        return session.reset()
