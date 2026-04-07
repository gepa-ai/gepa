# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Session — forkable conversation primitive for LLMs and coding agents."""

from __future__ import annotations

import copy
import uuid
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
