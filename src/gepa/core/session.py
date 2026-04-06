# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Session(Protocol):
    """Forkable interaction context for LLMs and coding agents.

    A session tracks conversation state and supports three operations:

    - ``send(content)``: send a message and get a response -- history grows.
    - ``fork()``: create a **new** session with history **copied**.
    - ``reset()``: create a **new** session with **no history** but same config.

    Both ``fork()`` and ``reset()`` leave the original untouched.

    +-----------------+--------------+----------+----------------+
    | Operation       | Creates new? | History  | Mutates orig?  |
    +-----------------+--------------+----------+----------------+
    | send(content)   | No           | Grows    | Yes            |
    | fork()          | Yes          | Copied   | No             |
    | reset()         | Yes          | Empty    | No             |
    +-----------------+--------------+----------+----------------+
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this session."""
        ...

    def send(self, content: str, **kwargs: Any) -> str:
        """Send a message and get a response.  History grows."""
        ...

    def fork(self) -> Session:
        """Create a new session with history copied from this one.

        Original is untouched.  The fork diverges independently.
        """
        ...

    def reset(self) -> Session:
        """Create a new session with no history but same backend config.

        Original is untouched.  Use when starting fresh exploration
        without conversation baggage.
        """
        ...

    @property
    def history(self) -> list[dict[str, Any]]:
        """Message history (read-only view)."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class LLMSession(Session):
    """Session backed by an in-memory message list -- works with any LLM API.

    ``send()`` appends a user message, calls the LLM, and appends the response.
    ``fork()`` deep-copies the message list so the child can diverge.
    ``reset()`` creates a new session with the same system prompt and API but
    no message history.

    Parameters
    ----------
    system_prompt:
        The system-level instruction prepended to every API call.
    api_call:
        A callable ``(messages: list[dict]) -> str`` that sends messages to an
        LLM and returns the assistant's text response.  This keeps the session
        backend-agnostic (works with litellm, openai, anthropic, etc.).
    session_id:
        Optional explicit id.  Auto-generated (uuid4) when omitted.
    messages:
        Optional pre-existing message history to resume from.
    """

    def __init__(
        self,
        system_prompt: str,
        api_call: Any,
        *,
        session_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._system_prompt = system_prompt
        self._api_call = api_call
        self._messages: list[dict[str, Any]] = list(messages) if messages else []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def send(self, content: str, **kwargs: Any) -> str:
        self._messages.append({"role": "user", "content": content})
        full_messages = [{"role": "system", "content": self._system_prompt}, *self._messages]
        response = self._api_call(full_messages, **kwargs)
        self._messages.append({"role": "assistant", "content": response})
        return response

    def fork(self) -> LLMSession:
        return LLMSession(
            system_prompt=self._system_prompt,
            api_call=self._api_call,
            messages=copy.deepcopy(self._messages),
        )

    def reset(self) -> LLMSession:
        return LLMSession(
            system_prompt=self._system_prompt,
            api_call=self._api_call,
        )


class NullSession(Session):
    """No-op session for backward compatibility.

    All operations are safe to call but do nothing meaningful.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id or "null"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def history(self) -> list[dict[str, Any]]:
        return []

    def send(self, content: str, **kwargs: Any) -> str:
        return ""

    def fork(self) -> NullSession:
        return NullSession(session_id=uuid.uuid4().hex[:12])

    def reset(self) -> NullSession:
        return NullSession(session_id=uuid.uuid4().hex[:12])


# ---------------------------------------------------------------------------
# Session strategy -- data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionEntry:
    """A session plus strategy-tracked metadata.

    ``val_score`` is the aggregate validation score of the candidate that
    was produced in this session.  ``None`` means unscored (e.g. before the
    session has been observed, or when the strategy doesn't track scores).
    """

    session: Session
    val_score: float | None = None


@dataclass(frozen=True)
class SessionContext:
    """Context handed to ``SessionStrategy.select`` before a mutation.

    Carries everything a strategy needs to decide which session to use:
    the parent candidate (for lineage-aware strategies), the current
    iteration, the read-only view of known sessions, and a factory for
    brand-new sessions.
    """

    parent_candidate_idx: int | None
    iteration: int
    sessions: Mapping[int | str, SessionEntry]
    create: Callable[[], Session]


@dataclass(frozen=True)
class SessionOutcome:
    """Outcome of a mutation, handed to ``SessionStrategy.observe``.

    Strategies use this to update their bookkeeping — typically binding a
    newly-accepted candidate to the session that produced it.
    """

    candidate_idx: int | None
    accepted: bool
    session: Session
    val_score: float | None = None


# ---------------------------------------------------------------------------
# Session strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStrategy(Protocol):
    """Policy for picking and updating sessions across iterations.

    Two hooks:

    - ``select(ctx)`` — before each mutation, return the session to use.
    - ``observe(outcome)`` — after each mutation, return a dict of
      ``{key: SessionEntry}`` updates to merge into the manager's state.
      Return an empty mapping to skip updates.

    Strategies are pure: they never mutate shared state directly.  All
    updates flow through the return value of ``observe``.
    """

    def select(self, ctx: SessionContext) -> Session: ...

    def observe(self, outcome: SessionOutcome) -> Mapping[int | str, SessionEntry]: ...


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_session_strategy(strategy: str | SessionStrategy) -> SessionStrategy:
    """Convert a strategy name to a ``SessionStrategy`` instance.

    Accepts a string (``"fork"``, ``"reset"``, ``"random"``, ``"round_robin"``,
    ``"parent_linked"``) or an already-instantiated ``SessionStrategy``.
    """
    if not isinstance(strategy, str):
        return strategy
    from gepa.strategies.session_strategy import (
        AlwaysFork,
        AlwaysReset,
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
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown session_strategy: {strategy!r}. Supported: {list(strategies)}")
    return strategies[strategy]()


# ---------------------------------------------------------------------------
# Session manager -- keyed store + strategy
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages a keyed store of sessions with a pluggable strategy.

    The manager holds a ``dict[int | str, SessionEntry]``.  Strategies
    decide how keys are assigned (candidate index, iteration number,
    ``"global"``, etc.) via the ``observe`` return value.  The manager
    never inspects keys — it only merges updates.

    Lifecycle per iteration:

    1. ``select(parent_candidate_idx)`` — build ``SessionContext`` and ask
       the strategy which session to use.  The returned session becomes
       ``current_session``.
    2. The engine runs the mutation using the current session.
    3. ``observe(candidate_idx, accepted, val_score)`` — build
       ``SessionOutcome`` and merge the strategy's updates into the store.

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
        self._sessions: dict[int | str, SessionEntry] = {}
        self._iteration: int = 0
        self._current: Session | None = None

    @property
    def sessions(self) -> Mapping[int | str, SessionEntry]:
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
        """Notify the strategy of a mutation outcome and merge its updates."""
        if self._current is None:
            return
        outcome = SessionOutcome(
            candidate_idx=candidate_idx,
            accepted=accepted,
            session=self._current,
            val_score=val_score,
        )
        updates = self._strategy.observe(outcome)
        if updates:
            self._sessions.update(updates)

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
