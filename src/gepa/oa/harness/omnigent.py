"""Omnigent harness — run OA agent sessions on any Omnigent-wrapped backend.

Omnigent (https://omnigent.ai) wraps coding agents — Claude Code
(``claude-sdk``), Codex, Goose, Qwen, Kimi, Hermes, and more — behind one
server/runner runtime with its own OS sandboxing. Driving OA through it
makes the backend a config value, delegates jailing to Omnigent's sandbox,
and reads cost/status/usage from Omnigent's uniform session record.

One server + runner pair per harness instance (or attach via
``server_url``); one Omnigent session per :meth:`run`; ``resume=True``
continues the same conversation. EXPERIMENTAL — not yet run end-to-end; see
``harness/README.md`` for the known gaps and the upstream ask-list.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from gepa.oa.engines.claude_utils import copy_session_transcript
from gepa.oa.harness.base import AgentHarness, AgentRunResult, AgentRunSpec

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

# Backends whose file/shell tools run through an Omnigent-managed OS
# environment (mirrors Omnigent's _OS_ENV_HARNESSES).
OS_ENV_BACKENDS = frozenset({"claude-sdk", "codex", "pi", "qwen", "goose", "kimi"})

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent executing one optimization step for GEPA's "
    "optimize_anything. Follow the instructions in the prompt exactly; write "
    "all outputs as files in the working directory."
)

HEALTH_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 0.25


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class OmnigentHarness(AgentHarness):
    """Agent harness backed by an Omnigent server + runner.

    ``backend`` is any Omnigent harness id (``"claude-sdk"``, ``"codex"``,
    ``"goose"``, ...). ``sandbox=True`` uses Omnigent's platform-default
    sandbox, ``sandbox_type`` picks an explicit type/provider, ``False``
    disables it — GEPA's own bwrap jail is never stacked on top (Omnigent's
    seccomp profile forbids nesting). Pass ``server_url`` + ``runner_id`` to
    attach to an external server instead of spawning one.
    """

    name = "omnigent"

    def __init__(
        self,
        *,
        backend: str = "claude-sdk",
        sandbox: bool = True,
        sandbox_type: str | None = None,
        server_url: str | None = None,
        runner_id: str | None = None,
        state_dir: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.sandbox = bool(sandbox) or sandbox_type is not None
        self.sandbox_type = sandbox_type
        self.state_dir = Path(state_dir) if state_dir else None
        self.server_url = server_url
        self.runner_id = runner_id
        self.managed = server_url is None
        self.server_proc: subprocess.Popen[bytes] | None = None
        self.runner_proc: subprocess.Popen[bytes] | None = None

    def preflight(self, engine_name: str) -> None:
        try:
            import omnigent_client  # noqa: F401

            if self.managed:
                import omnigent  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"[{engine_name}] harness=omnigent requires the omnigent packages: "
                "install per https://omnigent.ai/quickstart/install (or pass "
                "server_url= to attach to a running server)."
            ) from e

    def ensure_server(self, work_dir: Path) -> str:
        if self.server_url is not None:
            return self.server_url
        # Ported from Omnigent's full-server test infra; the env vars and
        # runner entrypoint are internal surfaces (ask: a supported
        # local-embedded mode).
        from omnigent.runner.identity import token_bound_runner_id

        state = self.state_dir or (work_dir / ".omnigent")
        state.mkdir(parents=True, exist_ok=True)
        (state / "artifacts").mkdir(exist_ok=True)
        port = find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        binding_token = secrets.token_urlsafe(32)
        runner_id = token_bound_runner_id(binding_token)

        self.server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omnigent.cli",
                "server",
                "--port",
                str(port),
                "--database-uri",
                f"sqlite:///{state / 'omnigent.db'}",
                "--artifact-location",
                str(state / "artifacts"),
            ],
            env={**os.environ, "OMNIGENT_RUNNER_TUNNEL_TOKEN": binding_token},
            stdout=(state / "server.log").open("wb"),
            stderr=subprocess.STDOUT,
        )
        self.runner_proc = subprocess.Popen(
            [sys.executable, "-m", "omnigent.runner._entry"],
            env={
                **os.environ,
                "OMNIGENT_RUNNER_ID": runner_id,
                "OMNIGENT_RUNNER_TUNNEL_BINDING_TOKEN": binding_token,
                "OMNIGENT_RUNNER_PARENT_PID": str(os.getpid()),
                "RUNNER_SERVER_URL": base_url,
            },
            stdout=(state / "runner.log").open("wb"),
            stderr=subprocess.STDOUT,
        )
        self.wait_ready(base_url, runner_id, state)
        self.server_url = base_url
        self.runner_id = runner_id
        return base_url

    @staticmethod
    def wait_ready(base_url: str, runner_id: str, state: Path) -> None:
        import httpx

        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                health = httpx.get(f"{base_url}/health", timeout=2)
                status = httpx.get(f"{base_url}/v1/runners/{runner_id}/status", timeout=2)
                if health.status_code == 200 and status.status_code == 200 and status.json().get("online") is True:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(POLL_INTERVAL_S)
        raise RuntimeError(
            f"omnigent server+runner not ready within {HEALTH_TIMEOUT_S}s; "
            f"see {state / 'server.log'} and {state / 'runner.log'}"
        )

    def close(self) -> None:
        for proc in (self.runner_proc, self.server_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.runner_proc = self.server_proc = None
        if self.managed:
            self.server_url = None

    def agent_config(self, spec: AgentRunSpec) -> dict[str, Any]:
        config: dict[str, Any] = {
            "spec_version": 1,
            "name": SAFE_NAME_RE.sub("-", f"gepa-oa-{self.backend}"),
            "prompt": spec.system_prompt or DEFAULT_SYSTEM_PROMPT,
            "executor": {"type": "omnigent", "model": spec.model, "config": {"harness": self.backend}},
        }
        if self.backend in OS_ENV_BACKENDS:
            os_env: dict[str, Any] = {"type": "caller_process"}
            # No sandbox key = Omnigent's platform default (bwrap/Seatbelt).
            if self.sandbox_type is not None:
                os_env["sandbox"] = {"type": self.sandbox_type}
            elif not self.sandbox:
                os_env["sandbox"] = {"type": "none"}
            config["os_env"] = os_env
        return config

    @staticmethod
    def build_bundle(config: dict[str, Any]) -> bytes:
        import yaml

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            payload = yaml.safe_dump(config).encode()
            info = tarfile.TarInfo("config.yaml")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        return buf.getvalue()

    def run(self, spec: AgentRunSpec) -> AgentRunResult:
        base_url = self.ensure_server(spec.work_dir)
        try:
            return asyncio.run(self.run_async(base_url, spec))
        except Exception as e:
            # Engines must see is_error, not a crash.
            return AgentRunResult(
                is_error=True,
                error=f"{type(e).__name__}: {e}",
                session_id=spec.session_id,
                returncode=1,
            )

    async def run_async(self, base_url: str, spec: AgentRunSpec) -> AgentRunResult:
        from omnigent_client import OmnigentClient
        from omnigent_client._sessions_chat import SessionsChat

        client = OmnigentClient(base_url)
        try:
            if spec.resume and spec.session_id:
                session = await client.sessions.get(spec.session_id)
            else:
                session = await client.sessions.create(
                    self.build_bundle(self.agent_config(spec)),
                    workspace=str(spec.work_dir),
                    reasoning_effort=spec.effort if spec.max_thinking_tokens is None else None,
                )
                assert self.runner_id is not None
                session = await client.sessions.bind_runner(session.id, runner_id=self.runner_id)
            chat = SessionsChat(namespace=client.sessions, files_uploader=None, files_getter=None, session=session)
            result = await asyncio.wait_for(chat.query(spec.prompt), timeout=spec.timeout_seconds)
            snap = await client.sessions.get(session.id)
            return self.result_from_snapshot(snap, text=getattr(result, "text", None))
        finally:
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()

    @staticmethod
    def result_from_snapshot(snap: Any, *, text: str | None) -> AgentRunResult:
        raw: dict[str, Any] = snap.model_dump(mode="json")
        error = raw.get("last_task_error")
        is_error = bool(error) or raw.get("status") == "failed"

        tokens_in = tokens_out = 0
        for usage in (raw.get("usage_by_model") or {}).values():
            tokens_in += int(usage.get("input_tokens") or 0)
            tokens_out += int(usage.get("output_tokens") or 0)

        return AgentRunResult(
            text=text,
            cost_usd=float(raw.get("total_cost_usd") or 0.0),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            is_error=is_error,
            error=json.dumps(error) if error else ("session status=failed" if is_error else None),
            session_id=raw.get("id"),
            native_session_id=raw.get("external_session_id"),
            returncode=1 if is_error else 0,
            raw=raw,
        )

    def export_transcript(self, session_id: str, work_dir: Path, dst_dir: Path) -> None:
        """Write ``<session_id>.jsonl`` (same shape as ``omni session export``),
        plus the wrapped agent's native transcript when discoverable."""
        if self.server_url is None:
            return
        try:
            import httpx

            dst_dir.mkdir(parents=True, exist_ok=True)
            with httpx.Client(base_url=self.server_url, timeout=30) as http:
                meta = http.get(
                    f"/v1/sessions/{session_id}",
                    params={"include_items": "false", "include_liveness": "false"},
                ).json()
                lines = [json.dumps({"record_type": "session_meta", **meta})]
                after: str | None = None
                while True:
                    params: dict[str, str] = {"limit": "500", "order": "asc"}
                    if after:
                        params["after"] = after
                    page = http.get(f"/v1/sessions/{session_id}/items", params=params).json()
                    data = page.get("data", [])
                    lines.extend(json.dumps({"record_type": "item", **item}) for item in data)
                    if not data or not page.get("has_more"):
                        break
                    after = data[-1].get("id")
                (dst_dir / f"{session_id}.jsonl").write_text("\n".join(lines) + "\n")

                # external_session_id addresses the claude-sdk backend's
                # native transcript under ~/.claude/projects/.
                native = meta.get("external_session_id")
                workspace = meta.get("workspace")
                if native and workspace and self.backend == "claude-sdk":
                    copy_session_transcript(Path(workspace), native, dst_dir)
        except Exception:
            pass
