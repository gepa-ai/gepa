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
# LanguageModel bridge
# ---------------------------------------------------------------------------


def make_session_lm(session: Session) -> Callable[[str | list[dict[str, Any]]], str]:
    """Wrap a Session as a ``LanguageModel`` callable.

    The returned callable satisfies the ``LanguageModel`` protocol used by
    ``ReflectiveMutationProposer``::

        class LanguageModel(Protocol):
            def __call__(self, prompt: str | list[dict[str, Any]]) -> str: ...

    This bridges sessions into the existing reflection pipeline — the proposer
    calls ``lm(prompt)`` and the session handles statefulness (message history,
    fork/reset/continue) transparently.

    Parameters
    ----------
    session:
        The session to route calls through.  Each ``lm(prompt)`` call maps to
        ``session.send(content)``.

    Example
    -------
    ::

        session = MessageListSession(system_prompt="...", api_call=my_llm)
        lm = make_session_lm(session)

        # Use as a regular LanguageModel in the proposer:
        proposer = ReflectiveMutationProposer(reflection_lm=lm, ...)
    """

    def lm(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            content = prompt
        else:
            # Extract the last user message from an OpenAI-style message list
            content = prompt[-1]["content"] if prompt else ""
        return session.send(content)

    return lm
