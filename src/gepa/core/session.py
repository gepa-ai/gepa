# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
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
    | fork(label)     | Yes          | Copied   | No             |
    | reset(label)    | Yes          | Empty    | No             |
    +-----------------+--------------+----------+----------------+
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this session."""
        ...

    def send(self, content: str, **kwargs: Any) -> str:
        """Send a message and get a response.  History grows."""
        ...

    def fork(self, label: str = "") -> Session:
        """Create a new session with history copied from this one.

        Original is untouched.  The fork diverges independently.
        """
        ...

    def reset(self, label: str = "") -> Session:
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


class MessageListSession:
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

    # -- Protocol properties --------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    # -- Protocol methods -----------------------------------------------------

    def send(self, content: str, **kwargs: Any) -> str:
        self._messages.append({"role": "user", "content": content})
        full_messages = [{"role": "system", "content": self._system_prompt}, *self._messages]
        response = self._api_call(full_messages, **kwargs)
        self._messages.append({"role": "assistant", "content": response})
        return response

    def fork(self, label: str = "") -> MessageListSession:
        new_id = f"{self._session_id}_fork_{label or uuid.uuid4().hex[:6]}"
        return MessageListSession(
            system_prompt=self._system_prompt,
            api_call=self._api_call,
            session_id=new_id,
            messages=copy.deepcopy(self._messages),
        )

    def reset(self, label: str = "") -> MessageListSession:
        new_id = f"{self._session_id}_reset_{label or uuid.uuid4().hex[:6]}"
        return MessageListSession(
            system_prompt=self._system_prompt,
            api_call=self._api_call,
            session_id=new_id,
        )


class NullSession:
    """No-op session for text-mode backward compatibility.

    All operations are safe to call but do nothing meaningful.
    Used when a proposer does not need session management (e.g. the existing
    ``ReflectiveMutationProposer``).
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

    def fork(self, label: str = "") -> NullSession:
        return NullSession(session_id=f"{self._session_id}_fork_{label or 'null'}")

    def reset(self, label: str = "") -> NullSession:
        return NullSession(session_id=f"{self._session_id}_reset_{label or 'null'}")


# ---------------------------------------------------------------------------
# Session strategy -- governs session selection
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStrategy(Protocol):
    """Policy that picks which session to use for the next mutation.

    At each iteration, the engine picks a parent candidate to mutate.
    The strategy then decides: fork an existing session from the pool,
    or reset one to start fresh.

    The strategy sees the pool of existing sessions and a factory for
    bootstrapping the very first session (when the pool is empty).

    Two built-in primitives a strategy can use:

    - **fork**: call ``session.fork()`` on a pooled session (copy history)
    - **reset**: call ``session.reset()`` on a pooled session (no history)

    Built-in strategies use one primitive each.  Custom strategies can
    mix them: random, round-robin, LLM-decided, explore-exploit, etc.
    """

    def select(
        self,
        sessions: list[Session],
        create: Callable[[], Session],
    ) -> Session:
        """Return the session to use for the next mutation.

        Parameters
        ----------
        sessions:
            The pool of existing sessions (may be empty).
        create:
            Factory that creates the first session when the pool is empty.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Session manager -- pool + strategy
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages a pool of sessions with a pluggable strategy.

    At each iteration:

    1. The engine calls ``select()`` -- the strategy picks a session
       from the pool (fork) or resets one (fresh start).
    2. The returned session is used for the mutation (via ``send()``).
    3. The session is automatically added to the pool.

    Example
    -------
    ::

        create = lambda: MessageListSession(system_prompt="...", api_call=llm)
        manager = SessionManager(create=create, strategy=AlwaysFork())

        session = manager.select()
        response = session.send("Improve the code...")
    """

    def __init__(
        self,
        create: Callable[[], Session],
        strategy: SessionStrategy | None = None,
    ) -> None:
        self._create = create
        self._strategy: SessionStrategy = strategy or AlwaysFork()
        self._sessions: list[Session] = []
        self._current: Session | None = None

    @property
    def sessions(self) -> list[Session]:
        """The pool of sessions."""
        return list(self._sessions)

    def select(self) -> Session:
        """Pick the session for the next mutation using the configured strategy."""
        self._current = self._strategy.select(self._sessions, self._create)
        if self._current not in self._sessions:
            self._sessions.append(self._current)
        return self._current

    def current_session(self) -> Session:
        """Return the active session (set by the last ``select()`` call)."""
        if self._current is None:
            self._current = self._create()
            self._sessions.append(self._current)
        return self._current


# ---------------------------------------------------------------------------
# LanguageModel bridge
# ---------------------------------------------------------------------------


def make_session_lm(
    session: Session | Callable[[], Session],
) -> Callable[[str | list[dict[str, Any]]], str]:
    """Wrap a Session (or session provider) as a ``LanguageModel`` callable.

    The returned callable satisfies the ``LanguageModel`` protocol used by
    ``ReflectiveMutationProposer``::

        class LanguageModel(Protocol):
            def __call__(self, prompt: str | list[dict[str, Any]]) -> str: ...

    This bridges sessions into the existing reflection pipeline -- the proposer
    calls ``lm(prompt)`` and the session handles statefulness transparently.

    Parameters
    ----------
    session:
        Either a ``Session`` instance (fixed) or a callable that returns the
        current session (dynamic -- e.g. ``manager.current_session``).

    Example
    -------
    ::

        # Fixed session:
        lm = make_session_lm(session)

        # Dynamic via SessionManager:
        manager = SessionManager(create=factory, strategy=AlwaysFork())
        lm = make_session_lm(manager.current_session)
    """

    def lm(prompt: str | list[dict[str, Any]]) -> str:
        sess = session() if callable(session) and not isinstance(session, Session) else session
        if isinstance(prompt, str):
            content = prompt
        else:
            content = prompt[-1]["content"] if prompt else ""
        return sess.send(content)

    return lm
