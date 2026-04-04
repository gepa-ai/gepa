# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""OpenCodeSession — Session backed by the OpenCode CLI."""

from __future__ import annotations

import json
import subprocess
import uuid  # used for fallback session_id
from typing import Any


class OpenCodeSession:
    """Session backed by OpenCode CLI.

    Uses ``opencode run`` for non-interactive mode.  History is managed by
    OpenCode internally via ``--session``.  ``fork()`` uses OpenCode's
    native ``--fork``.  ``reset()`` creates a fresh process.

    Parameters
    ----------
    model:
        Model name (e.g. ``"sonnet"``, ``"gpt-4"``, etc.).
    system_prompt:
        System prompt passed via ``--system-prompt``.
    session_id:
        Existing session ID to resume.  ``None`` starts a new session.
    timeout:
        Timeout in seconds for each CLI call.  Default 300 (5 min).
    extra_flags:
        Additional CLI flags to pass to every ``opencode`` invocation.
    """

    def __init__(
        self,
        model: str = "sonnet",
        *,
        system_prompt: str = "",
        session_id: str | None = None,
        timeout: int = 300,
        extra_flags: list[str] | None = None,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._session_id = session_id
        self._timeout = timeout
        self._extra_flags = extra_flags or []
        self._total_cost_usd = 0.0
        self._last_send_cost: float | None = None

    @property
    def session_id(self) -> str:
        return self._session_id or "uninitialized"

    @property
    def history(self) -> list[dict[str, Any]]:
        # OpenCode manages history internally
        return []

    @property
    def last_send_cost(self) -> float | None:
        """Cost of the most recent ``send()`` call, or ``None`` if unknown."""
        return self._last_send_cost

    @property
    def total_cost(self) -> float:
        """Cumulative cost across all ``send()`` calls in this session."""
        return self._total_cost_usd

    def _base_cmd(self) -> list[str]:
        cmd = [
            "opencode",
            "run",
            "--model", self._model,
            "--format", "json",
        ]
        if self._system_prompt:
            cmd += ["--system-prompt", self._system_prompt]
        cmd += self._extra_flags
        return cmd

    def send(self, content: str, **kwargs: Any) -> str:
        cmd = self._base_cmd() + [content]
        if self._session_id:
            cmd += ["--session", self._session_id]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode failed (exit {result.returncode}): {result.stderr.strip()}")

        output = json.loads(result.stdout)
        self._session_id = output.get("session_id", self._session_id or uuid.uuid4().hex[:12])
        cost = output.get("cost_usd", 0.0)
        self._last_send_cost = cost
        self._total_cost_usd += cost
        return output.get("result", "")

    def fork(self) -> OpenCodeSession:
        if self._session_id is None:
            raise RuntimeError("Cannot fork a session that hasn't been used yet.")

        cmd = self._base_cmd() + [
            "",
            "--session", self._session_id,
            "--fork",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode fork failed (exit {result.returncode}): {result.stderr.strip()}")

        output = json.loads(result.stdout)
        return OpenCodeSession(
            model=self._model,
            system_prompt=self._system_prompt,
            session_id=output.get("session_id"),
            timeout=self._timeout,
            extra_flags=self._extra_flags,
        )

    def reset(self) -> OpenCodeSession:
        return OpenCodeSession(
            model=self._model,
            system_prompt=self._system_prompt,
            timeout=self._timeout,
            extra_flags=self._extra_flags,
        )
