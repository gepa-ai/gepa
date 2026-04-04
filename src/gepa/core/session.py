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

    A session tracks conversation state and supports branching:
    - ``send()``: interact with the underlying LLM/agent
    - ``fork()``: create an independent copy (explore different mutation directions)
    - ``reset()``: return to initial state (start fresh)

    **Text-mode LLMs**: ``MessageListSession`` wraps a system prompt + message
    list.  ``fork()`` deep-copies messages, ``reset()`` clears to system prompt.

    **Coding agents** (Claude Code, Codex): concrete implementations in
    ``src/gepa/core/sessions/`` wrap CLI subprocesses.  ``fork()`` creates a
    new session from a checkpoint, ``reset()`` starts a fresh session.
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this session."""
        ...

    def send(self, content: str, **kwargs: Any) -> str:
        """Send a message and get a response."""
        ...

    def fork(self, label: str = "") -> Session:
        """Create an independent copy of this session's state.

        The forked session shares history up to this point but diverges after.
        """
        ...

    def reset(self) -> None:
        """Return to initial state (system prompt only, no history)."""
        ...

    @property
    def history(self) -> list[dict[str, Any]]:
        """Message history (read-only view)."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class MessageListSession:
    """Session backed by an in-memory message list — works with any LLM API.

    ``fork()`` deep-copies the message list so the child can diverge.
    ``reset()`` clears back to the system prompt only.
    ``send()`` appends a user message, calls the LLM, and appends the response.

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

    def reset(self) -> None:
        self._messages.clear()


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

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Session strategy — governs the session tree
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStrategy(Protocol):
    """Policy that governs how the session tree evolves.

    A strategy decides — given the parent candidate and all available sessions —
    which session to use for the next mutation.  It uses the session primitives
    (``fork``, ``reset``, ``send``) internally but the *policy* logic is what
    makes it a strategy.

    Examples of strategies:

    - **Random**: randomly continue, fork, or start fresh each iteration.
    - **Greedy**: always continue the best-scoring session.
    - **Explore-exploit**: fork top K, reset bottom K, continue the rest.
    - **LLM-decided**: ask an LLM whether to continue or try fresh.
    - **Population**: maintain N active sessions, prune worst performers.
    """

    def select(
        self,
        parent_candidate_idx: int,
        sessions: dict[int, Session],
        factory: Callable[[], Session],
    ) -> Session:
        """Return the session to use for the next mutation.

        Parameters
        ----------
        parent_candidate_idx:
            Index of the candidate selected as the mutation parent.
        sessions:
            The current ``candidate_idx → Session`` registry (snapshots).
        factory:
            Creates a fresh session with no history.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class AlwaysContinueStrategy:
    """Reuse the parent's session — history keeps growing.

    The agent/LLM sees the full conversation history from prior iterations.
    Good when continuity helps (e.g. the LLM can learn from past mistakes).
    """

    def select(
        self,
        parent_candidate_idx: int,
        sessions: dict[int, Session],
        factory: Callable[[], Session],
    ) -> Session:
        return sessions.get(parent_candidate_idx) or factory()


class AlwaysFreshStrategy:
    """Start a fresh session every time — no conversation history.

    The agent only sees the candidate's code/text state, not prior conversation.
    Good for independent exploration from any point in the candidate tree.
    """

    def select(
        self,
        parent_candidate_idx: int,
        sessions: dict[int, Session],
        factory: Callable[[], Session],
    ) -> Session:
        return factory()


class AlwaysForkStrategy:
    """Deep-copy the parent's session — history preserved, diverges after.

    Like ``continue`` but creates an independent copy so the parent's session
    is not modified.  Good when you want to try a different direction while
    preserving the parent's context.
    """

    def select(
        self,
        parent_candidate_idx: int,
        sessions: dict[int, Session],
        factory: Callable[[], Session],
    ) -> Session:
        parent = sessions.get(parent_candidate_idx)
        return parent.fork() if parent else factory()


# ---------------------------------------------------------------------------
# Session manager — registry + strategy
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages the session tree with a pluggable strategy.

    Combines two concerns:

    1. **Registry** — ``candidate_idx → Session`` mapping (the tree).
    2. **Strategy** — a ``SessionStrategy`` that governs how sessions are
       selected, forked, or created for each mutation.

    Example
    -------
    ::

        factory = lambda: MessageListSession(system_prompt="...", api_call=llm)
        manager = SessionManager(
            session_factory=factory,
            strategy=AlwaysFreshStrategy(),
        )

        # When proposer selects parent candidate 2:
        session = manager.select(parent_candidate_idx=2)
        response = session.send("Improve the code...")

        # After candidate is accepted:
        manager.register(candidate_idx=5)
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        strategy: SessionStrategy | None = None,
    ) -> None:
        self._factory = session_factory
        self._strategy: SessionStrategy = strategy or AlwaysContinueStrategy()
        self._sessions: dict[int, Session] = {}
        self._current: Session | None = None

    @property
    def sessions(self) -> dict[int, Session]:
        """Read-only view of the candidate → session registry."""
        return dict(self._sessions)

    def register(self, candidate_idx: int, session: Session | None = None) -> None:
        """Snapshot a session for a candidate.

        If *session* is ``None``, snapshots (forks) the current session.
        The stored session is always a fork so the original can keep growing.
        """
        sess = session or self._current
        if sess is not None:
            self._sessions[candidate_idx] = sess.fork(label=f"c{candidate_idx}")

    def select(self, parent_candidate_idx: int) -> Session:
        """Pick the session for the next mutation using the configured strategy.

        Returns the selected session and sets it as ``current_session()``.
        """
        self._current = self._strategy.select(
            parent_candidate_idx, self._sessions, self._factory
        )
        return self._current

    def current_session(self) -> Session:
        """Return the active session (set by the last ``select()`` call)."""
        if self._current is None:
            self._current = self._factory()
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

    This bridges sessions into the existing reflection pipeline — the proposer
    calls ``lm(prompt)`` and the session handles statefulness transparently.

    Parameters
    ----------
    session:
        Either a ``Session`` instance (fixed) or a callable that returns the
        current session (dynamic — e.g. ``manager.current_session``).

    Example
    -------
    ::

        # Fixed session:
        lm = make_session_lm(session)

        # Dynamic via SessionManager:
        manager = SessionManager(factory, strategy="branch")
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
