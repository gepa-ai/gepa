# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""SessionManager — orchestrates sessions across optimization iterations.

Holds a keyed store of sessions, delegates to a pluggable ``SessionStrategy``
for selection and bookkeeping, and bridges sessions into GEPA's
``LanguageModel`` protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from gepa.core.session import Session

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionRecord:
    """A session plus strategy-tracked metadata.

    Attributes
    ----------
    session:
        The session instance bound to this candidate.
    val_score:
        Aggregate validation score of the candidate produced in this
        session.  ``None`` before observation or when the strategy
        doesn't track scores.
    created_at:
        The ``SessionManager`` selection counter at the time this entry
        was created (i.e. which ``select()`` call produced it).
    """

    session: Session
    val_score: float | None = None
    created_at: int | None = None


@dataclass(frozen=True)
class SessionContext:
    """Snapshot passed to ``SessionStrategy.select`` before a mutation.

    Rebuilt every call — only ``sessions`` persists across iterations.

    Attributes
    ----------
    parent_candidate_idx:
        Which candidate the proposer is about to mutate.  ``None`` at the
        root iteration (no parent yet).
    iteration:
        Current iteration counter.
    sessions:
        Read-only view of the store — all accepted candidates and their
        sessions so far.
    create:
        Factory for brand-new sessions (used when the store is empty).
    """

    parent_candidate_idx: int | None
    iteration: int
    sessions: Mapping[int | str, SessionRecord]
    create: Callable[[], Session]


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStrategy(Protocol):
    """Policy for picking a session before each mutation.

    One hook:

    - ``select(ctx)`` — before each mutation, return the session to use.

    Bookkeeping (storing accepted candidates, tracking scores) is handled
    by ``SessionManager.observe()`` — strategies only decide *which*
    session to use.
    """

    def select(self, ctx: SessionContext) -> Session: ...


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_session_strategy(strategy: str | SessionStrategy) -> SessionStrategy:
    """Convert a strategy name to a ``SessionStrategy`` instance.

    Accepts a string (``"fork"``, ``"reset"``, ``"random"``, ``"round_robin"``,
    ``"parent_linked"``, ``"best_score"``) or an already-instantiated ``SessionStrategy``.
    """
    if not isinstance(strategy, str):
        return strategy
    from gepa.strategies.session_strategy import (
        AlwaysFork,
        AlwaysReset,
        BestScore,
        ParentLinked,
        RandomStrategy,
        RoundRobin,
    )

    strategies: dict[str, type] = {
        "fork": AlwaysFork,
        "reset": AlwaysReset,
        "random": RandomStrategy,
        "round_robin": RoundRobin,
        "parent_linked": ParentLinked,
        "best_score": BestScore,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown session_strategy: {strategy!r}. Supported: {list(strategies)}")
    return strategies[strategy]()


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages a keyed store of sessions with a pluggable strategy.

    The manager holds a ``dict[int | str, SessionRecord]``.  Accepted
    candidates are stored by index; rejected candidates are not stored.

    Lifecycle per iteration:

    1. ``select(parent_candidate_idx)`` — build ``SessionContext`` and ask
       the strategy which session to use.  The returned session becomes
       ``current_session``.
    2. The proposer runs the mutation using the current session.
    3. ``observe(candidate_idx, accepted, val_score)`` — store accepted
       candidates in the session map.

    Example
    -------
    ::

        from gepa.strategies.session_strategy import AlwaysFork

        create = lambda: LLMSession(system_prompt="...", api_call=llm)
        manager = SessionManager(create=create, strategy=AlwaysFork())

        session = manager.select(parent_candidate_idx=3)
        response = session.send("Improve the code...")
        manager.observe(candidate_idx=7, accepted=True, val_score=0.87)
    """

    def __init__(
        self,
        create: Callable[[], Session],
        strategy: SessionStrategy | None = None,
    ) -> None:
        self._create = create
        if strategy is None:
            from gepa.strategies.session_strategy import AlwaysFork

            strategy = AlwaysFork()
        self._strategy: SessionStrategy = strategy
        self._sessions: dict[int | str, SessionRecord] = {}
        self._iteration: int = 0
        self._current: Session | None = None

    @property
    def sessions(self) -> Mapping[int | str, SessionRecord]:
        """Read-only view of the session store."""
        return dict(self._sessions)

    @property
    def iteration(self) -> int:
        """The iteration counter, incremented on each ``select()`` call."""
        return self._iteration

    def select(self, parent_candidate_idx: int | None = None) -> Session:
        """Pick the session for the next mutation using the configured strategy."""
        ctx = SessionContext(
            parent_candidate_idx=parent_candidate_idx,
            iteration=self._iteration,
            sessions=dict(self._sessions),
            create=self._create,
        )
        self._current = self._strategy.select(ctx)
        self._iteration += 1
        return self._current

    def observe(
        self,
        candidate_idx: int | None,
        accepted: bool,
        val_score: float | None = None,
    ) -> None:
        """Record a mutation outcome.  Accepted candidates are stored in the session map."""
        if self._current is None:
            return
        if accepted and candidate_idx is not None:
            self._sessions[candidate_idx] = SessionRecord(
                session=self._current,
                val_score=val_score,
                created_at=self._iteration - 1,
            )

    def current_session(self) -> Session:
        """Return the active session (set by the last ``select()`` call)."""
        if self._current is None:
            self._current = self._create()
        return self._current


# ---------------------------------------------------------------------------
# LanguageModel bridge
# ---------------------------------------------------------------------------


def make_session_lm(
    session: Session | Callable[[], Session],
    fallback: Callable | None = None,
) -> Callable[[str | list[dict[str, Any]]], str]:
    """Wrap a Session (or session provider) as a ``LanguageModel`` callable.

    The returned callable satisfies the ``LanguageModel`` protocol used by
    ``ReflectiveMutationProposer``::

        class LanguageModel(Protocol):
            def __call__(self, prompt: str | list[dict[str, Any]]) -> str: ...

    For multimodal prompts (``list[dict]`` with image content), the session
    is bypassed and the ``fallback`` LM is called directly, since sessions
    only handle text.

    Parameters
    ----------
    session:
        Either a ``Session`` instance (fixed) or a callable that returns the
        current session (dynamic — e.g. ``manager.current_session``).
    fallback:
        Optional LM callable for multimodal prompts that sessions can't handle.
    """

    def lm(prompt: str | list[dict[str, Any]]) -> str:
        sess = session() if callable(session) and not isinstance(session, Session) else session
        if isinstance(prompt, str):
            return sess.send(prompt)
        # Multimodal prompt (list of dicts) — bypass session if fallback is available
        if fallback is not None:
            return fallback(prompt)
        # No fallback — extract text content and send through session
        content = prompt[-1]["content"] if prompt else ""
        return sess.send(content)

    return lm
