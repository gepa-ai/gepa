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

    A session tracks conversation state and supports two operations:

    - ``resume(content)``: continue the conversation -- history grows.
    - ``fork()``: create a **new** session with history **copied** from this
      one.  Original is untouched.  The fork diverges independently.

    A third way to obtain a session -- ``create()`` -- lives on the
    ``SessionManager``, not on Session itself.  A session can clone itself
    (``fork``), but creating one from nothing is the manager's job.

    +-----------------+--------------+----------+----------------+
    | Operation       | Creates new? | History  | Mutates orig?  |
    +-----------------+--------------+----------+----------------+
    | resume(content) | No           | Grows    | Yes            |
    | fork(label)     | Yes          | Copied   | No             |
    +-----------------+--------------+----------+----------------+
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this session."""
        ...

    def resume(self, content: str, **kwargs: Any) -> str:
        """Send a message and get a response.  History grows."""
        ...

    def fork(self, label: str = "") -> Session:
        """Create a new session with history copied from this one.

        Original is untouched.  The fork diverges independently.
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

    ``fork()`` deep-copies the message list so the child can diverge.
    ``resume()`` appends a user message, calls the LLM, and appends the response.

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

    def resume(self, content: str, **kwargs: Any) -> str:
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

    def resume(self, content: str, **kwargs: Any) -> str:
        return ""

    def fork(self, label: str = "") -> NullSession:
        return NullSession(session_id=f"{self._session_id}_fork_{label or 'null'}")


# ---------------------------------------------------------------------------
# Session strategy -- governs the session tree
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStrategy(Protocol):
    """Policy that governs how the session tree evolves.

    A strategy makes two decisions each iteration:

    1. **select()** -- given the parent candidate and all available checkpoints,
       which session should the next mutation use?
    2. **checkpoint()** -- after a mutation, should we save the session state?

    The strategy uses three session primitives internally:

    - ``resume``: return the live session (history keeps growing)
    - ``fork``: create a copy from a checkpoint (sees past context, diverges)
    - ``create``: fresh session via factory (no history)

    Examples of strategies:

    - **AlwaysResume**: reuse the live session across iterations.
    - **AlwaysFork**: fork from the parent's checkpoint every time.
    - **AlwaysCreate**: fresh session every time (stateless, like current GEPA).
    - **ExploreExploit**: fork top-k performers, create fresh for the rest.
    - **LLMDecided**: ask an LLM whether to resume, fork, or create.
    """

    def select(
        self,
        parent_candidate_idx: int,
        current: Session | None,
        checkpoints: dict[int, Session],
        create: Callable[[], Session],
    ) -> Session:
        """Return the session to use for the next mutation.

        Parameters
        ----------
        parent_candidate_idx:
            Index of the candidate selected as the mutation parent.
        current:
            The live session from the previous iteration (if any).
        checkpoints:
            Immutable ``candidate_idx -> Session`` snapshots.
        create:
            Creates a brand-new session with no history.
        """
        ...

    def checkpoint(
        self,
        candidate_idx: int,
        session: Session,
        checkpoints: dict[int, Session],
        accepted: bool,
    ) -> Session | None:
        """Decide whether to save a checkpoint after a mutation.

        Parameters
        ----------
        candidate_idx:
            Index of the candidate just produced.
        session:
            The live session that produced the candidate.
        checkpoints:
            The current checkpoint registry.
        accepted:
            Whether the candidate was accepted by the selection step.

        Returns
        -------
        A forked session to store as checkpoint, or ``None`` to skip.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class AlwaysResume:
    """Reuse the live session -- history keeps growing.

    The agent/LLM sees the full conversation history from prior iterations.
    Good when continuity helps (e.g. the LLM can learn from past mistakes).
    """

    def select(
        self,
        parent_candidate_idx: int,
        current: Session | None,
        checkpoints: dict[int, Session],
        create: Callable[[], Session],
    ) -> Session:
        return current or create()

    def checkpoint(
        self,
        candidate_idx: int,
        session: Session,
        checkpoints: dict[int, Session],
        accepted: bool,
    ) -> Session | None:
        return session.fork(label=f"c{candidate_idx}")


class AlwaysFork:
    """Fork from the parent's checkpoint -- history copied, diverges after.

    Uses ``parent.fork()`` to create an independent copy with full conversation
    history.  Good when you want to try a different direction while keeping
    the context of what was tried before.
    """

    def select(
        self,
        parent_candidate_idx: int,
        current: Session | None,
        checkpoints: dict[int, Session],
        create: Callable[[], Session],
    ) -> Session:
        parent = checkpoints.get(parent_candidate_idx)
        return parent.fork() if parent else create()

    def checkpoint(
        self,
        candidate_idx: int,
        session: Session,
        checkpoints: dict[int, Session],
        accepted: bool,
    ) -> Session | None:
        return session.fork(label=f"c{candidate_idx}")


class AlwaysCreate:
    """Fresh session every time -- no history.

    Creates a new session with no conversation history for each mutation.
    Equivalent to current GEPA behavior (stateless reflection LM).
    Good as the default backward-compatible mode.
    """

    def select(
        self,
        parent_candidate_idx: int,
        current: Session | None,
        checkpoints: dict[int, Session],
        create: Callable[[], Session],
    ) -> Session:
        return create()

    def checkpoint(
        self,
        candidate_idx: int,
        session: Session,
        checkpoints: dict[int, Session],
        accepted: bool,
    ) -> Session | None:
        return session.fork(label=f"c{candidate_idx}")


# ---------------------------------------------------------------------------
# Session manager -- registry + strategy
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages the session tree with a pluggable strategy.

    Combines two concerns:

    1. **Registry** -- ``candidate_idx -> Session`` checkpoint mapping.
    2. **Strategy** -- a ``SessionStrategy`` that governs how sessions are
       selected and checkpointed for each mutation.

    Checkpoints are immutable snapshots (created via ``fork()``).  The live
    session can keep growing without corrupting stored snapshots.

    Example
    -------
    ::

        create = lambda: MessageListSession(system_prompt="...", api_call=llm)
        manager = SessionManager(create=create, strategy=AlwaysFork())

        # When proposer selects parent candidate 2:
        session = manager.select(parent_candidate_idx=2)
        response = session.resume("Improve the code...")

        # After candidate is evaluated:
        manager.checkpoint(candidate_idx=5, accepted=True)
    """

    def __init__(
        self,
        create: Callable[[], Session],
        strategy: SessionStrategy | None = None,
    ) -> None:
        self._create = create
        self._strategy: SessionStrategy = strategy or AlwaysResume()
        self._checkpoints: dict[int, Session] = {}
        self._current: Session | None = None

    @property
    def checkpoints(self) -> dict[int, Session]:
        """Read-only view of the candidate -> session checkpoint registry."""
        return dict(self._checkpoints)

    def select(self, parent_candidate_idx: int) -> Session:
        """Pick the session for the next mutation using the configured strategy.

        Returns the selected session and sets it as ``current_session()``.
        """
        self._current = self._strategy.select(parent_candidate_idx, self._current, self._checkpoints, self._create)
        return self._current

    def checkpoint(self, candidate_idx: int, accepted: bool) -> None:
        """Ask the strategy whether to save a checkpoint for this candidate.

        The strategy returns a forked session to store, or ``None`` to skip.
        """
        if self._current is None:
            return
        result = self._strategy.checkpoint(candidate_idx, self._current, self._checkpoints, accepted)
        if result is not None:
            self._checkpoints[candidate_idx] = result

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
        return sess.resume(content)

    return lm
