"""Agent-harness abstraction for optimize_anything's subprocess engines.

Every OA engine that drives a coding agent (autoresearch, meta_harness, the
gepa Claude-Code proposer) consumes exactly three signals from a finished
agent run — cost (budget enforcement), success/failure, and token usage —
plus the agent's file outputs in the work dir and, as a diagnostic artifact,
a transcript. Today each engine builds a ``claude --print`` argv by hand;
this module names that contract so the agent runtime becomes a config knob
(``OptimizeAnythingConfig.harness``) instead of a hardcoded CLI.

Implementations:

- :class:`~gepa.oa.harness.claude_code.ClaudeCodeHarness` — the current
  ``claude --print --output-format json`` subprocess path, verbatim.
- :class:`~gepa.oa.harness.omnigent.OmnigentHarness` — any agent backend
  Omnigent wraps (claude-sdk, codex, goose, qwen, kimi, hermes, ...) via the
  Omnigent server + runner and the ``omnigent_client`` SDK.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentRunSpec:
    """One agent invocation: a prompt executed with tools in ``work_dir``.

    Attributes:
        prompt: The task prompt for the agent session.
        work_dir: Working directory the agent runs in; candidate files and
            engine state live here, and sandboxes are rooted here.
        model: Backend model identifier (harness-specific format).
        session_id: Harness-scoped session identifier. For claude-code the
            caller mints a uuid and passes it on every call; for omnigent
            leave ``None`` on the first call and pass back
            :attr:`AgentRunResult.session_id` with ``resume=True`` to
            continue the same conversation.
        resume: Continue the session named by ``session_id`` instead of
            starting a fresh one (autoresearch's iteration loop).
        effort: Reasoning-effort level (``"low"``..``"max"``), or ``None``
            for the backend default. Mutually exclusive with
            ``max_thinking_tokens`` — implementations ignore ``effort``
            when a thinking budget is set.
        max_thinking_tokens: Fixed thinking-token budget, or ``None``.
        max_budget_usd: Hard per-session spend cap when the harness can
            enforce one (claude-code: ``--max-budget-usd``). Harnesses that
            cannot enforce pre-emptively still *report* cost so the engine's
            cross-iteration budget loop works; see the implementation's
            docstring.
        extra_env: Extra environment variables for the agent process
            (e.g. ``GEPA_OMNI_WORK_DIR``).
        timeout_seconds: Kill the run after this long. ``None`` = no limit.
        system_prompt: Optional system-prompt override where the harness
            supports one (omnigent agent spec ``prompt``); claude-code
            ignores it (the CLI's default system prompt is part of the
            harness).
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
    """What an engine needs to know about a finished agent run.

    Attributes:
        text: Final assistant text, or ``None`` when the run produced none.
            Engines generally read candidate artifacts from the work dir,
            not from this field.
        cost_usd: Harness-reported USD cost of the session (cumulative for
            resumed sessions where noted by the implementation).
        tokens_in / tokens_out: Token usage, 0 when unreported.
        is_error: True when the run failed (nonzero exit, harness error
            envelope, failed session status, unparseable output). Engines
            must treat an ``is_error`` run as "no proposal", never as "agent
            declined to change anything".
        error: Human-readable failure description when ``is_error``.
        session_id: Harness-scoped session id, for ``resume`` and
            :meth:`AgentHarness.export_transcript`.
        native_session_id: The wrapped agent's own session id when the
            harness exposes one (omnigent ``external_session_id``); ``None``
            otherwise.
        duration_ms / num_turns: Diagnostics when reported, else ``None``.
        returncode: Subprocess exit code, or a synthesized 0/1 for
            non-subprocess harnesses.
        stderr: Captured stderr tail for logging, "" when not applicable.
        raw: The harness's own result record (claude JSON envelope /
            omnigent session snapshot), for engine log files.
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

    #: Registry / display name, e.g. ``"claude-code"``.
    name: str = "agent"

    @abc.abstractmethod
    def preflight(self, engine_name: str) -> None:
        """Fail fast (with an actionable message) when the harness cannot run.

        Called once at engine start, before any budget is spent.
        """

    @abc.abstractmethod
    def run(self, spec: AgentRunSpec) -> AgentRunResult:
        """Execute one agent session synchronously and return its outcome.

        Must not raise for agent-level failures — report them via
        :attr:`AgentRunResult.is_error` so engines uniformly distinguish
        "failed" from "ran and proposed nothing". May raise for harness
        misconfiguration (preflight-class errors).
        """

    def export_transcript(self, session_id: str, work_dir: Path, dst_dir: Path) -> None:
        """Best-effort: mirror the session transcript(s) into ``dst_dir``.

        A diagnostic artifact, not a correctness signal (see
        ``claude_utils.copy_session_transcript``). Default: no-op.
        """

    def close(self) -> None:
        """Release long-lived resources (servers, runners). Default: no-op."""
