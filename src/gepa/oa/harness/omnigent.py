"""Omnigent harness — run OA agent sessions on any Omnigent-wrapped backend.

Omnigent (https://omnigent.ai, ``pip install omnigent``) is a meta-harness
that wraps coding agents — Claude Code (``claude-sdk``), Codex, Goose, Qwen,
Kimi, Hermes, Pi, Cursor, Copilot, and generic ACP agents — behind one
server/runner runtime with its own OS sandboxing (bwrap/Seatbelt/remote
providers). Driving OA's agent sessions through it buys three things:

1. **Backend choice as config** — ``harness={"type": "omnigent", "backend":
   "codex"}`` runs the same OA engine on Codex instead of Claude Code, with
   no engine changes and one uniform result/record format.
2. **Omnigent's sandbox instead of ours** — the agent bundle's
   ``os_env.sandbox`` block delegates jailing to Omnigent (platform default
   bwrap/Seatbelt, or a named provider). GEPA's own bwrap jail is *never*
   stacked on top: Omnigent's bwrap seccomp profile denies nested
   ``CLONE_NEW*``, so exactly one of the two must own isolation.
3. **A removal path for OA's Claude-CLI plumbing** — cost/status/usage come
   from Omnigent's session record, transcripts from its items API plus (for
   backends that expose one) the wrapped agent's native transcript via
   ``external_session_id``.

Topology per harness instance: one ``omnigent server`` (isolated sqlite DB
under the run dir) + one runner, spawned lazily and shared by every
:meth:`run` call; each call creates one Omnigent session (agent bundle upload
→ ``bind_runner`` → ``SessionsChat.query``) and reads the session snapshot
back for cost/usage/status. ``resume=True`` sends the next prompt to the
same session, which replaces the claude CLI's ``--resume`` in autoresearch's
iteration loop. Alternatively pass ``server_url``/``runner_id`` to attach to
an externally managed server.

STATUS: experimental draft. Known gaps, each an ask on the Omnigent side or
a documented workaround here — see ``harness/README.md`` for the full list:

- Managed spawn uses Omnigent-internal surfaces (``omnigent.runner._entry``,
  ``token_bound_runner_id``, ``OMNIGENT_RUNNER_TUNNEL_TOKEN``): there is no
  public "give me a local server+runner" API yet.
- No pre-emptive per-session budget cap (``--max-budget-usd`` equivalent):
  cost is enforced post-hoc by the engines' cross-iteration loop.
- ``max_thinking_tokens`` and the WebFetch/WebSearch denylist have no
  session-create-time equivalent (Omnigent policies could express the
  latter; not wired here).
- Session ``status`` served after a server restart, and
  ``external_session_id`` for headless backends, depend on two upstream
  PRs (see README).
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

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

#: Backends whose file/shell tools run through an Omnigent-managed OS
#: environment; only these take an ``os_env`` block (mirrors Omnigent's
#: ``_OS_ENV_HARNESSES``).
_OS_ENV_BACKENDS = frozenset({"claude-sdk", "codex", "pi", "qwen", "goose", "kimi"})

_DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent executing one optimization step for GEPA's "
    "optimize_anything. Follow the instructions in the prompt exactly; write "
    "all outputs as files in the working directory."
)

_HEALTH_TIMEOUT_S = 60.0
_POLL_INTERVAL_S = 0.25


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class OmnigentHarness(AgentHarness):
    """Agent harness backed by an Omnigent server + runner.

    Args:
        backend: Omnigent harness id to run the agent on, e.g.
            ``"claude-sdk"`` (default), ``"codex"``, ``"goose"``, ``"qwen"``,
            ``"kimi"``, ``"hermes"``. Any id Omnigent's registry accepts.
        sandbox: Delegate OS jailing to Omnigent. ``True`` (default) leaves
            the bundle's ``os_env.sandbox`` unset so Omnigent applies its
            platform default (bwrap on Linux, Seatbelt on macOS); ``False``
            writes ``sandbox: {type: none}``. GEPA's own bwrap prefix is
            never applied around Omnigent processes.
        sandbox_type: Explicit Omnigent sandbox type/provider (e.g.
            ``"bwrap"``, ``"seatbelt"``, or an installed remote provider),
            overriding the platform default. Implies ``sandbox=True``.
        server_url: Attach to an already-running Omnigent server instead of
            spawning one. Requires ``runner_id`` of an online runner.
        runner_id: Runner to bind sessions to in attach mode.
        state_dir: Where the managed server keeps its sqlite DB, artifacts,
            and logs. Defaults to ``<work_dir>/.omnigent`` of the first run.
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
        self._state_dir = Path(state_dir) if state_dir else None
        self._server_url = server_url
        self._runner_id = runner_id
        self._managed = server_url is None
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._runner_proc: subprocess.Popen[bytes] | None = None

    # ── preflight / lifecycle ────────────────────────────────────────

    def preflight(self, engine_name: str) -> None:
        try:
            import omnigent_client  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"[{engine_name}] harness=omnigent requires the omnigent_client "
                "SDK (and, for the managed server mode, the omnigent package): "
                "install Omnigent per https://omnigent.ai/quickstart/install, "
                "or `uv pip install omnigent`."
            ) from e
        if self._managed:
            try:
                import omnigent  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    f"[{engine_name}] no server_url given, so the omnigent "
                    "package must be importable to spawn a local server+runner."
                ) from e

    def _ensure_server(self, work_dir: Path) -> str:
        """Spawn (once) or return the Omnigent server this harness talks to."""
        if self._server_url is not None:
            return self._server_url
        # Managed mode. NOTE: this ports the spawn recipe from Omnigent's own
        # full-server test infrastructure; the env vars and runner entrypoint
        # are internal surfaces (ask: a supported local-embedded mode).
        from omnigent.runner.identity import token_bound_runner_id

        state = self._state_dir or (work_dir / ".omnigent")
        state.mkdir(parents=True, exist_ok=True)
        (state / "artifacts").mkdir(exist_ok=True)
        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        binding_token = secrets.token_urlsafe(32)
        runner_id = token_bound_runner_id(binding_token)

        self._server_proc = subprocess.Popen(
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
        self._runner_proc = subprocess.Popen(
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
        self._wait_ready(base_url, runner_id, state)
        self._server_url = base_url
        self._runner_id = runner_id
        return base_url

    @staticmethod
    def _wait_ready(base_url: str, runner_id: str, state: Path) -> None:
        import httpx

        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                health = httpx.get(f"{base_url}/health", timeout=2)
                status = httpx.get(f"{base_url}/v1/runners/{runner_id}/status", timeout=2)
                if health.status_code == 200 and status.status_code == 200 and status.json().get("online") is True:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(_POLL_INTERVAL_S)
        raise RuntimeError(
            f"omnigent server+runner not ready within {_HEALTH_TIMEOUT_S}s; "
            f"see {state / 'server.log'} and {state / 'runner.log'}"
        )

    def close(self) -> None:
        for proc in (self._runner_proc, self._server_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._runner_proc = self._server_proc = None
        if self._managed:
            self._server_url = None

    # ── agent bundle ─────────────────────────────────────────────────

    def _agent_config(self, spec: AgentRunSpec) -> dict[str, Any]:
        """Omnigent ``spec_version: 1`` agent config for this run."""
        name = _SAFE_NAME_RE.sub("-", f"gepa-oa-{self.backend}")
        config: dict[str, Any] = {
            "spec_version": 1,
            "name": name,
            "prompt": spec.system_prompt or _DEFAULT_SYSTEM_PROMPT,
            "executor": {
                "type": "omnigent",
                "model": spec.model,
                "config": {"harness": self.backend},
            },
        }
        if self.backend in _OS_ENV_BACKENDS:
            os_env: dict[str, Any] = {"type": "caller_process"}
            if self.sandbox_type is not None:
                os_env["sandbox"] = {"type": self.sandbox_type}
            elif not self.sandbox:
                os_env["sandbox"] = {"type": "none"}
            # sandbox=True with no explicit type: omit the key so Omnigent
            # applies its platform default (bwrap on Linux, Seatbelt on mac).
            config["os_env"] = os_env
        return config

    @staticmethod
    def _bundle(config: dict[str, Any]) -> bytes:
        import yaml

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            payload = yaml.safe_dump(config).encode()
            info = tarfile.TarInfo("config.yaml")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        return buf.getvalue()

    # ── run ──────────────────────────────────────────────────────────

    def run(self, spec: AgentRunSpec) -> AgentRunResult:
        base_url = self._ensure_server(spec.work_dir)
        try:
            return asyncio.run(self._run_async(base_url, spec))
        except Exception as e:
            return AgentRunResult(
                is_error=True,
                error=f"{type(e).__name__}: {e}",
                session_id=spec.session_id,
                returncode=1,
            )

    async def _run_async(self, base_url: str, spec: AgentRunSpec) -> AgentRunResult:
        from omnigent_client import OmnigentClient
        from omnigent_client._sessions_chat import SessionsChat

        client = OmnigentClient(base_url)
        try:
            if spec.resume and spec.session_id:
                session = await client.sessions.get(spec.session_id)
            else:
                session = await client.sessions.create(
                    self._bundle(self._agent_config(spec)),
                    workspace=str(spec.work_dir),
                    reasoning_effort=spec.effort if spec.max_thinking_tokens is None else None,
                )
                assert self._runner_id is not None
                session = await client.sessions.bind_runner(session.id, runner_id=self._runner_id)
            chat = SessionsChat(
                namespace=client.sessions,
                files_uploader=None,
                files_getter=None,
                session=session,
            )
            query = chat.query(spec.prompt)
            if spec.timeout_seconds is not None:
                result = await asyncio.wait_for(query, timeout=spec.timeout_seconds)
            else:
                result = await query

            snap = await client.sessions.get(session.id)
            return self._result_from_snapshot(snap, text=getattr(result, "text", None))
        finally:
            aclose = getattr(client, "aclose", None) or getattr(client, "close", None)
            if aclose is not None:
                maybe = aclose()
                if asyncio.iscoroutine(maybe):
                    await maybe

    @staticmethod
    def _result_from_snapshot(snap: Any, *, text: str | None) -> AgentRunResult:
        status = getattr(snap, "status", None)
        is_error = status == "failed"
        error = None
        last_task_error = getattr(snap, "last_task_error", None)
        if last_task_error:
            is_error = True
            error = json.dumps(last_task_error) if isinstance(last_task_error, dict) else str(last_task_error)
        elif is_error:
            error = "omnigent session status=failed"

        tokens_in = tokens_out = 0
        usage_by_model = getattr(snap, "usage_by_model", None) or {}
        try:
            for usage in dict(usage_by_model).values():
                tokens_in += int(getattr(usage, "input_tokens", 0) or usage.get("input_tokens", 0) or 0)
                tokens_out += int(getattr(usage, "output_tokens", 0) or usage.get("output_tokens", 0) or 0)
        except (TypeError, AttributeError, ValueError):
            pass

        raw: dict[str, Any]
        dump = getattr(snap, "model_dump", None)
        raw = dump(mode="json") if callable(dump) else {"repr": repr(snap)}

        return AgentRunResult(
            text=text,
            cost_usd=float(getattr(snap, "total_cost_usd", 0.0) or 0.0),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            is_error=is_error,
            error=error,
            session_id=getattr(snap, "id", None),
            native_session_id=getattr(snap, "external_session_id", None),
            returncode=1 if is_error else 0,
            raw=raw,
        )

    # ── transcript export ────────────────────────────────────────────

    def export_transcript(self, session_id: str, work_dir: Path, dst_dir: Path) -> None:
        """Write ``<session_id>.jsonl`` (session_meta + items, the same shape
        as ``omni session export``) into ``dst_dir``; for backends whose
        native transcript is discoverable (claude-sdk + upstream PR), also
        mirror the wrapped agent's own transcript."""
        if self._server_url is None:
            return
        try:
            import httpx

            dst_dir.mkdir(parents=True, exist_ok=True)
            with httpx.Client(base_url=self._server_url, timeout=30) as http:
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
                    for item in data:
                        lines.append(json.dumps({"record_type": "item", **item}))
                    if not data or not page.get("has_more"):
                        break
                    after = data[-1].get("id")
                (dst_dir / f"{session_id}.jsonl").write_text("\n".join(lines) + "\n")

                # Native transcript, when the backend exposes one: claude-sdk
                # sessions record the Claude Code session uuid as
                # external_session_id (upstream PR), which addresses
                # ~/.claude/projects/<workspace-slug>/<uuid>.jsonl.
                native = meta.get("external_session_id")
                workspace = meta.get("workspace")
                if native and workspace and self.backend == "claude-sdk":
                    copy_session_transcript(Path(workspace), native, dst_dir)
        except Exception:
            pass
