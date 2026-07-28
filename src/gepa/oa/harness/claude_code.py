"""Claude Code CLI harness — the existing ``claude --print`` path, named.

Unifies the subprocess invocation the OA engines built inline (autoresearch,
meta_harness, ClaudeCodeAgentProposer) so those call sites collapse to
``harness.run(spec)``. Sandboxing and permission posture come from
:mod:`gepa.oa.sandbox` unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from gepa.oa.engines.claude_utils import copy_session_transcript
from gepa.oa.harness.base import AgentHarness, AgentRunResult, AgentRunSpec
from gepa.oa.sandbox import (
    DENY_WEB_TOOLS,
    bwrap_prefix,
    claude_permission_args,
    preflight_claude_engine,
)


class ClaudeCodeHarness(AgentHarness):
    """Run agent sessions through the ``claude`` CLI in print mode."""

    name = "claude-code"

    def __init__(self, *, sandbox: bool = True) -> None:
        self.sandbox = bool(sandbox)

    def preflight(self, engine_name: str) -> None:
        preflight_claude_engine(engine_name, sandbox=self.sandbox)

    def run(self, spec: AgentRunSpec) -> AgentRunResult:
        session_id = spec.session_id or str(uuid.uuid4())
        cmd: list[str] = bwrap_prefix(spec.work_dir) if self.sandbox else []
        cmd += [
            "claude",
            "--print",
            spec.prompt,
            "--output-format",
            "json",
            "--model",
            spec.model,
        ]
        if spec.resume:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--session-id", session_id])
        cmd.append(DENY_WEB_TOOLS)
        cmd.extend(claude_permission_args(spec.work_dir, sandboxed=self.sandbox))
        # A fixed thinking budget replaces effort-based adaptive thinking.
        if spec.max_thinking_tokens is None and spec.effort is not None:
            cmd.extend(["--effort", spec.effort])
        if spec.max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", f"{max(0.01, spec.max_budget_usd):.4f}"])

        env = {**os.environ, **spec.extra_env}
        env.pop("CLAUDECODE", None)
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        if spec.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(spec.max_thinking_tokens)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(spec.work_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return AgentRunResult(
                is_error=True,
                error=f"claude timed out after {spec.timeout_seconds}s",
                session_id=session_id,
                returncode=124,
                stderr=(e.stderr or "")[-4000:] if isinstance(e.stderr, str) else "",
            )
        return self.parse_result(proc, session_id)

    @staticmethod
    def parse_result(proc: subprocess.CompletedProcess[str], session_id: str) -> AgentRunResult:
        payload: dict[str, Any] = {}
        parse_error: str | None = None
        stdout = (proc.stdout or "").strip()
        if stdout:
            try:
                payload = json.loads(stdout)
            except (json.JSONDecodeError, ValueError) as e:
                parse_error = f"{type(e).__name__}: {e}"

        try:
            cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        usage = payload.get("usage") or {}
        try:
            tokens_in = int(usage.get("input_tokens", 0) or 0)
            tokens_out = int(usage.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            tokens_in = tokens_out = 0

        # A non-zero exit, unparseable/empty output, or is_error in the
        # envelope all mean "the run failed", not "no changes proposed".
        is_error_payload = bool(payload.get("is_error"))
        empty_payload = not payload and not stdout
        is_error = bool(proc.returncode != 0 or parse_error or is_error_payload or empty_payload)
        error = None
        if is_error:
            error = (
                f"returncode={proc.returncode}"
                + (f" parse_error={parse_error}" if parse_error else "")
                + (" is_error=true" if is_error_payload else "")
                + (" empty_output=true" if empty_payload else "")
            )

        return AgentRunResult(
            text=payload.get("result"),
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            is_error=is_error,
            error=error,
            session_id=payload.get("session_id") or session_id,
            duration_ms=payload.get("duration_ms"),
            num_turns=payload.get("num_turns"),
            returncode=proc.returncode,
            stderr=(proc.stderr or "")[-4000:],
            raw=payload,
        )

    def export_transcript(self, session_id: str, work_dir: Path, dst_dir: Path) -> None:
        copy_session_transcript(work_dir, session_id, dst_dir)
