# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""ClaudeCodeSession — Session backed by the Claude Code CLI."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class ClaudeCodeSession:
    """Session backed by Claude Code CLI.

    Uses ``claude -p`` for non-interactive mode.  History is managed by
    Claude Code internally via ``--resume``.  ``fork()`` uses Claude Code's
    native ``--fork-session``.  ``reset()`` creates a fresh process.

    Parameters
    ----------
    model:
        Model alias (e.g. ``"sonnet"``, ``"opus"``) or full model name.
    system_prompt:
        System prompt passed via ``--system-prompt``.
    session_id:
        Existing session ID to resume.  ``None`` starts a new session.
    timeout:
        Timeout in seconds for each CLI call.  Default 300 (5 min).
    max_budget_usd:
        Optional per-session budget cap passed via ``--max-budget-usd``.
    extra_flags:
        Additional CLI flags to pass to every ``claude`` invocation.
    """

    def __init__(
        self,
        model: str = "sonnet",
        *,
        system_prompt: str = "",
        session_id: str | None = None,
        timeout: int = 300,
        max_budget_usd: float | None = None,
        extra_flags: list[str] | None = None,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._session_id = session_id
        self._timeout = timeout
        self._max_budget_usd = max_budget_usd
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
            "claude",
            "-p",
            "--model",
            self._model,
            "--output-format",
            "json",
        ]
        if self._system_prompt:
            cmd += ["--system-prompt", self._system_prompt]
        if self._max_budget_usd is not None:
            cmd += ["--max-budget-usd", str(self._max_budget_usd)]
        cmd += self._extra_flags
        return cmd

    @staticmethod
    def _extract_result(raw: str) -> dict[str, Any]:
        """Parse Claude Code JSON output — handles both list and dict formats.

        ``claude -p --output-format json`` returns a JSON **array** where the
        last element is the result object containing ``result``, ``session_id``,
        ``total_cost_usd``, ``is_error``, etc.
        """
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed[-1]  # type: ignore[no-any-return]
        return parsed  # type: ignore[no-any-return]

    def send(self, content: str, **kwargs: Any) -> str:
        cmd = self._base_cmd() + [content]
        if self._session_id:
            cmd += ["--resume", self._session_id]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)

        if result.returncode != 0:
            raise RuntimeError(f"Claude Code failed (exit {result.returncode}): {result.stderr.strip()}")

        output = self._extract_result(result.stdout)

        if output.get("is_error"):
            raise RuntimeError(f"Claude Code error: {output.get('result', 'unknown error')}")

        self._session_id = output["session_id"]
        cost = output.get("total_cost_usd") or output.get("cost_usd", 0.0)
        self._last_send_cost = cost
        self._total_cost_usd += cost
        return output["result"]

    def fork(self) -> ClaudeCodeSession:
        if self._session_id is None:
            raise RuntimeError("Cannot fork a session that hasn't been used yet.")

        cmd = self._base_cmd() + [
            "Continue.",
            "--resume",
            self._session_id,
            "--fork-session",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)

        if result.returncode != 0:
            raise RuntimeError(f"Claude Code fork failed (exit {result.returncode}): {result.stderr.strip()}")

        output = self._extract_result(result.stdout)
        return ClaudeCodeSession(
            model=self._model,
            system_prompt=self._system_prompt,
            session_id=output["session_id"],
            timeout=self._timeout,
            max_budget_usd=self._max_budget_usd,
            extra_flags=self._extra_flags,
        )

    def reset(self) -> ClaudeCodeSession:
        return ClaudeCodeSession(
            model=self._model,
            system_prompt=self._system_prompt,
            timeout=self._timeout,
            max_budget_usd=self._max_budget_usd,
            extra_flags=self._extra_flags,
        )
