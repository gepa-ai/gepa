# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""OpenCodeSession — Session backed by the OpenCode CLI."""

from __future__ import annotations

import json
import subprocess
import uuid
from typing import Any


class OpenCodeSession:
    """Session backed by OpenCode CLI.

    Uses ``opencode run --format json`` for non-interactive mode.  The CLI
    emits newline-delimited JSON (NDJSON) events::

        {"type": "step_start",  "sessionID": "...", ...}
        {"type": "text",        "part": {"text": "..."}, ...}
        {"type": "step_finish", "part": {"cost": 0.01, "tokens": {...}}, ...}

    History is managed by OpenCode internally via ``--session``.
    ``fork()`` uses ``--fork``.  ``reset()`` creates a fresh session.

    Parameters
    ----------
    model:
        Full provider/model string passed to ``--model``
        (e.g. ``"openrouter/anthropic/claude-sonnet-4.6"``).  See
        ``opencode models`` for the list supported by your install.
    system_prompt:
        System prompt passed via ``--system``.
    session_id:
        Existing session ID to resume.  ``None`` starts a new session.
    timeout:
        Timeout in seconds for each CLI call.  Default 300 (5 min).
    extra_flags:
        Additional CLI flags to pass to every ``opencode`` invocation.
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-sonnet-4.6",
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
            "--model",
            self._model,
            "--format",
            "json",
        ]
        if self._system_prompt:
            cmd += ["--system", self._system_prompt]
        cmd += self._extra_flags
        return cmd

    @staticmethod
    def _parse_ndjson(raw: str) -> tuple[str, str, float]:
        """Parse NDJSON events from ``opencode run --format json``.

        Returns ``(session_id, result_text, cost)``.
        """
        session_id = ""
        text_parts: list[str] = []
        cost = 0.0

        for line in raw.strip().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = event.get("type", "")

            if not session_id:
                session_id = event.get("sessionID", "")

            if event_type == "text":
                part = event.get("part", {})
                text_parts.append(part.get("text", ""))
            elif event_type == "step_finish":
                part = event.get("part", {})
                cost += part.get("cost", 0.0)
            elif event_type == "error":
                error = event.get("error", {})
                msg = error.get("data", {}).get("message", "unknown error")
                raise RuntimeError(f"OpenCode error: {msg}")

        return session_id, "".join(text_parts), cost

    def send(self, content: str, **kwargs: Any) -> str:
        cmd = self._base_cmd() + [content]
        if self._session_id:
            cmd += ["--session", self._session_id]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)

        if result.returncode != 0:
            if result.stdout.strip():
                try:
                    self._parse_ndjson(result.stdout)
                except RuntimeError:
                    raise
                except Exception:
                    pass
            raise RuntimeError(f"OpenCode failed (exit {result.returncode}): {result.stderr.strip()}")

        session_id, text, cost = self._parse_ndjson(result.stdout)
        self._session_id = session_id or self._session_id or uuid.uuid4().hex[:12]
        self._last_send_cost = cost
        self._total_cost_usd += cost
        return text

    def fork(self) -> OpenCodeSession:
        if self._session_id is None:
            raise RuntimeError("Cannot fork a session that hasn't been used yet.")

        cmd = self._base_cmd() + [
            "Continue.",
            "--session",
            self._session_id,
            "--fork",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode fork failed (exit {result.returncode}): {result.stderr.strip()}")

        session_id, _, _ = self._parse_ndjson(result.stdout)
        return OpenCodeSession(
            model=self._model,
            system_prompt=self._system_prompt,
            session_id=session_id or None,
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
