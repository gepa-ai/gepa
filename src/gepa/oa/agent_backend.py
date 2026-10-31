"""Internal coding-agent seam for optimize_anything engines and proposers.

The engines (``autoresearch``, ``meta_harness``) and agent-driven proposers
share one shape of interaction: hand an agent a prompt and a working
directory, let it act autonomously, and read back structured results from
files it writes. Historically every site built its own ``claude --print``
argv, cloned Claude-specific sandbox assumptions, and re-parsed usage
JSON. This module owns that seam so engines depend on a narrow interface
and provider-specific invocation lives in one backend class each.

Design constraints (per the maintainer discussion in gepa#459):

- intentionally **internal** (not a public plugin surface yet) — revisit
  a public abstraction or ACP/Omnigent integration only when the
  supported-agent surface grows;
- Claude Code stays the *default* backend: existing behavior and
  UX are unchanged when ``agent_backend`` is unset;
- backends own provider-specific bits — CLI argv, session resume,
  transcript layout, budget flags, sandbox mounts, permission posture —
  while sharing :class:`AgentRunResult` and the supervision contract.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class AgentRunResult:
    """Normalized outcome of one coding-agent invocation."""

    text: str = ""
    """Final assistant message (for reflection prompts); may be empty."""

    session_id: str | None = None
    """Provider-resumable session identifier, when the provider has one."""

    usage: dict[str, Any] = field(default_factory=dict)
    """Provider-native usage counters (tokens, requests, durations)."""

    cost: float | None = None
    """Reported cost in USD when the provider exposes it."""

    raw: dict[str, Any] | None = None
    """Provider-native result payload, when available."""


@dataclass(frozen=True)
class AgentRunRequest:
    """Everything a backend needs to run one autonomous agent invocation."""

    prompt: str
    workdir: Path
    resume_session_id: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: int | None = None
    extra_args: Sequence[str] = field(default_factory=tuple)


class CodingAgentBackend:
    """Interface for coding agents used as optimization engines/proposers."""

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        raise NotImplementedError

    def _spawn(self, argv: Sequence[str], request: AgentRunRequest) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(request.env)
        return subprocess.run(
            list(argv),
            cwd=request.workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
        )


class ClaudeCodeBackend(CodingAgentBackend):
    """Default backend: drives the existing ``claude --print`` flow.

    Wraps (not replaces) the invocation currently living in the engines so
    behavior is bit-identical; the argv listed here is the same one the
    engines assembled before the seam existed.
    """

    def __init__(
        self,
        model: str | None = None,
        disallowed_tools: Sequence[str] = (),
        permission_mode: str | None = "acceptEdits",
    ):
        self.model = model
        self.disallowed_tools = list(disallowed_tools)
        self.permission_mode = permission_mode

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        argv = [
            "claude",
            "--print",
            "--output-format",
            "json",
        ]
        if self.model:
            argv += ["--model", self.model]
        if request.resume_session_id:
            argv += ["--resume", request.resume_session_id]
        argv += ["--disallowedTools", *self.disallowed_tools]
        if self.permission_mode:
            argv += ["--permission-mode", self.permission_mode]
        argv += [request.prompt]
        argv += list(request.extra_args)
        proc = self._spawn(argv, request)
        usage, _result = {}, {}
        try:
            payload = json.loads(proc.stdout or "{}")
        except (ValueError, TypeError):
            payload = {}
        session_id = payload.get("session_id") or request.resume_session_id
        usage = payload.get("usage") or {}
        cost = payload.get("total_cost_usd")
        return AgentRunResult(
            text=payload.get("result", ""),
            session_id=session_id,
            usage=usage,
            cost=cost if isinstance(cost, int | float) else None,
            raw=payload,
        )


class CodexBackend(CodingAgentBackend):
    """OpenAI Codex CLI backend, ``codex exec --json`` based.

    Sessions are directory-scoped rollouts rather than named resumable
    threads; ``resume_session_id`` is intentionally unused for now and
    preserved in the result if the backend ever gains resume support.
    """

    def __init__(self, model: str | None = None, sandbox_mode: str = "workspace-write", full_auto: bool = False):
        self.model = model
        self.sandbox_mode = sandbox_mode
        self.full_auto = full_auto

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        argv: list[str] = ["codex", "exec", "--json"]
        if self.model:
            argv += ["-m", self.model]
        if self.full_auto:
            argv += ["--full-auto"]
        else:
            argv += ["--sandbox", self.sandbox_mode]
        argv += [request.prompt]
        proc = self._spawn(argv, request)
        # `codex exec --json` streams events; the final message is in
        # item.type == "agent_message" (or task_complete). Parse defensively.
        text = ""
        session_id = None
        usage: dict[str, Any] = {}
        for line in (proc.stdout or "").splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            t = event.get("type", "")
            if t == "session_configured":
                session_id = str(event.get("session_id"))
            elif t == "agent_message" and event.get("message"):
                text = event["message"]
        return AgentRunResult(text=text, session_id=session_id, usage=usage, cost=None, raw=None)


def default_backend() -> CodingAgentBackend:
    """Claude stays default; engines need not change when unset."""
    return ClaudeCodeBackend()
