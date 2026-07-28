"""Agent-harness abstraction for optimize_anything's subprocess engines.

Every OA engine that drives a coding agent consumes the same signals from a
finished run: cost, success/failure, token usage, the agent's file outputs
in the work dir, and (as a diagnostic) a transcript. This module names that
contract so the agent runtime becomes a config knob
(``OptimizeAnythingConfig.harness``) instead of a hardcoded ``claude
--print`` argv. Implementations: ``claude_code.ClaudeCodeHarness`` (the
existing CLI path) and ``omnigent.OmnigentHarness`` (any Omnigent-wrapped
backend).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentRunSpec:
    """One agent invocation: ``prompt`` executed with tools in ``work_dir``.

    ``session_id``/``resume`` continue a prior session (autoresearch's loop):
    claude-code mints a uuid the caller passes back, omnigent returns the
    conversation id from the first run. ``effort`` is ignored when
    ``max_thinking_tokens`` is set (documented mutex). ``max_budget_usd`` is
    a hard cap where the harness supports one; cost is always *reported* so
    engines can enforce budgets across iterations either way.
    """

    prompt: str
    work_dir: Path
    model: str
    session_id: str | None = None
    resume: bool = False
    effort: str | None = None
    max_thinking_tokens: int | None = None
    max_budget_usd: float | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    system_prompt: str | None = None


@dataclass
class AgentRunResult:
    """Outcome of one agent run.

    Engines must treat ``is_error`` as "no proposal", never as "agent
    declined to change anything". ``native_session_id`` is the wrapped
    agent's own session id when the harness exposes one (omnigent
    ``external_session_id``). ``raw`` is the harness's own result record
    (claude JSON envelope / omnigent session snapshot) for log files.
    """

    text: str | None = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    is_error: bool = False
    error: str | None = None
    session_id: str | None = None
    native_session_id: str | None = None
    duration_ms: float | None = None
    num_turns: int | None = None
    returncode: int = 0
    stderr: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class AgentHarness(abc.ABC):
    """Runtime that executes one agent session per :meth:`run` call."""

    name: str = "agent"

    @abc.abstractmethod
    def preflight(self, engine_name: str) -> None:
        """Fail fast with an actionable message when the harness cannot run."""

    @abc.abstractmethod
    def run(self, spec: AgentRunSpec) -> AgentRunResult:
        """Execute one agent session synchronously.

        Agent-level failures are reported via ``is_error``, not raised.
        """

    def export_transcript(self, session_id: str, work_dir: Path, dst_dir: Path) -> None:
        """Best-effort diagnostic: mirror session transcript(s) into ``dst_dir``."""

    def close(self) -> None:
        """Release long-lived resources (servers, runners)."""
