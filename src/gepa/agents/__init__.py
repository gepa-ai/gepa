# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Agent abstractions for GEPA — coding agents backed by CLI tools.

The main entry point is :class:`CodingAgent`, a thin factory that creates
:class:`~gepa.core.session.Session` instances for a given CLI harness
(``"claude_code"`` or ``"opencode"``) and model.

Example::

    from gepa.agents import CodingAgent

    agent = CodingAgent(harness="claude_code", model="sonnet")
    optimize(..., reflection_lm=agent)

Passing a ``CodingAgent`` as ``reflection_lm`` auto-wires the session
machinery — no need to call ``create_session()`` or ``make_session_lm()``
manually.  For advanced use cases (custom session pool), build it yourself::

    from gepa.core.session_manager import SessionManager

    manager = SessionManager(create=agent.create_session)
"""

from __future__ import annotations

import warnings
from typing import ClassVar

from gepa.core.session import Session


class CodingAgent:
    """Factory that creates Session instances for a given CLI harness and model.

    Parameters
    ----------
    harness:
        Which CLI backend to use: ``"claude_code"`` or ``"opencode"``.
    model:
        Model name passed through to the CLI.  ``"claude_code"`` accepts
        native aliases like ``"sonnet"`` / ``"opus"``.  ``"opencode"``
        requires a full provider/model string
        (e.g. ``"openrouter/anthropic/claude-sonnet-4.6"``).
    system_prompt:
        System-level instruction for the agent session.
    timeout:
        Timeout in seconds for each CLI call.  Default 300 (5 min).
    max_budget_usd:
        Per-session budget cap.  Only supported by ``claude_code``.
    extra_flags:
        Additional CLI flags passed to every invocation.
    """

    _HARNESSES: ClassVar[set[str]] = {"claude_code", "opencode"}

    def __init__(
        self,
        harness: str,
        model: str,
        *,
        system_prompt: str = "",
        timeout: int = 300,
        max_budget_usd: float | None = None,
        extra_flags: list[str] | None = None,
    ) -> None:
        if harness not in self._HARNESSES:
            raise ValueError(f"Unknown harness: {harness!r}. Supported: {sorted(self._HARNESSES)}")

        if max_budget_usd is not None and harness != "claude_code":
            warnings.warn(
                f"max_budget_usd is not supported by harness {harness!r}, ignoring",
                stacklevel=2,
            )

        self._harness = harness
        self._model = model
        self._system_prompt = system_prompt
        self._timeout = timeout
        self._max_budget_usd = max_budget_usd
        self._extra_flags = extra_flags

    @property
    def harness(self) -> str:
        return self._harness

    @property
    def model(self) -> str:
        return self._model

    def create_session(self) -> Session:
        """Create a new session for this agent's harness and model."""
        if self._harness == "claude_code":
            from gepa.agents.claude_code import ClaudeCodeSession

            return ClaudeCodeSession(
                model=self._model,
                system_prompt=self._system_prompt,
                timeout=self._timeout,
                max_budget_usd=self._max_budget_usd,
                extra_flags=self._extra_flags,
            )
        elif self._harness == "opencode":
            from gepa.agents.opencode import OpenCodeSession

            return OpenCodeSession(
                model=self._model,
                system_prompt=self._system_prompt,
                timeout=self._timeout,
                extra_flags=self._extra_flags,
            )
        # Unreachable due to __init__ validation, but satisfies type checkers
        raise ValueError(f"Unknown harness: {self._harness!r}")  # pragma: no cover


