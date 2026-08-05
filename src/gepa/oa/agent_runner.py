"""Agent subprocess runners shared by the long-running OA engines.

The public engines deliberately keep their evaluator and result contracts
independent from the agent CLI.  This module is the small boundary between
those concerns: a runner returns normalized output while the engine owns
candidate materialization, evaluation, and budget policy.

``PiAgentRunner`` supports both modes used by Omni:

* one-shot JSONL sessions for Meta-Harness iterations and GEPA proposers;
* one persistent RPC process for AutoResearch/Ralph continuations.

``CodexAgentRunner`` uses the Codex CLI's workspace-write sandbox. Meta-Harness
uses one ephemeral invocation per iteration; AutoResearch resumes one persisted
Codex thread for Ralph continuations.

Pi's tool allowlist is intentionally only a command-construction convenience.
Callers that request ``sandbox=True`` must provide an OS sandbox prefix.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ProgressCheck = Callable[[], str | None]
SandboxPrefix = Callable[[Path], list[str]]


@dataclass
class AgentRunResult:
    """Normalized result from an agent invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    session_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None  # ``None`` means token usage was not priced.
    cost_known: bool = True
    timed_out: bool = False
    completed: bool = False
    final_text: str = ""


class AgentRunner(Protocol):
    """Minimal runner contract consumed by agent-backed engines."""

    def run(
        self,
        prompt: str,
        *,
        work_dir: Path,
        timeout_seconds: float | None = None,
        progress_check: ProgressCheck | None = None,
        max_budget_usd: float | None = None,
    ) -> AgentRunResult: ...

    def close(self) -> None: ...


def _iter_json_lines(stdout: str) -> Iterable[dict[str, Any]]:
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            yield payload


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text_value(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "result", "message"):
            if key in value:
                text = _text_value(value[key])
                if text:
                    return text
    return ""


def _event_text(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", ""))
    if event_type in {"message_end", "assistant_message", "result", "agent_end"}:
        for key in ("message", "content", "text", "result", "data"):
            text = _text_value(event.get(key))
            if text:
                return text
    return ""


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_mappings(value: Any, *, inside_cost: bool = False) -> Iterable[tuple[dict[str, Any], bool]]:
    if not isinstance(value, dict):
        return
    yield value, inside_cost
    for key, child in value.items():
        if isinstance(child, dict):
            yield from _iter_mappings(child, inside_cost=inside_cost or key == "cost")
        elif isinstance(child, list):
            for item in child:
                yield from _iter_mappings(item, inside_cost=inside_cost or key == "cost")


def _merge_usage(events: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """Collect nested Pi usage/cost fields without assuming one event schema."""
    usage: dict[str, Any] = {}
    cost = 0.0
    token_aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens", "inputTokens", "input"),
        "output_tokens": ("output_tokens", "completion_tokens", "outputTokens", "output"),
        "cache_read_tokens": ("cache_read_tokens", "cacheReadTokens", "cache_read"),
        "cache_write_tokens": ("cache_write_tokens", "cacheWriteTokens", "cache_write"),
    }
    for event in events:
        for source, inside_cost in _iter_mappings(event):
            for canonical, aliases in token_aliases.items():
                value = next(
                    (_numeric(source.get(key)) for key in aliases if _numeric(source.get(key)) is not None), None
                )
                if value is not None:
                    usage[canonical] = value if canonical not in usage else usage[canonical] + value
            for key in ("cost_usd", "total_cost_usd", "totalCostUsd", "cost"):
                value = _numeric(source.get(key))
                if value is not None:
                    cost = max(cost, value)
            if inside_cost:
                value = _numeric(source.get("total"))
                if value is not None:
                    cost = max(cost, value)
    return usage, cost


def normalize_pi_output(stdout: str) -> tuple[dict[str, Any], float, str, bool]:
    """Return ``(usage, cost, final_text, completed)`` for Pi JSONL output."""
    events = list(_iter_json_lines(stdout))
    usage, cost = _merge_usage(events)
    final_text = ""
    for event in events:
        text = _event_text(event)
        if text:
            final_text = text
    completed = any(str(event.get("type", "")) == "agent_end" for event in events)
    return usage, cost, final_text, completed


_CODEX_TOKEN_ALIASES = {
    "input_tokens": ("input_tokens", "prompt_tokens", "inputTokens"),
    "output_tokens": ("output_tokens", "completion_tokens", "outputTokens"),
    "cache_read_tokens": ("cache_read_tokens", "cached_input_tokens", "cacheReadTokens"),
    "reasoning_output_tokens": ("reasoning_output_tokens", "reasoningOutputTokens"),
}


def _codex_usage(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return the last cumulative usage mapping emitted by Codex."""
    latest: dict[str, int] = {}
    for event in events:
        for source, _inside_cost in _iter_mappings(event):
            candidate: dict[str, int] = {}
            for canonical, aliases in _CODEX_TOKEN_ALIASES.items():
                value = next(
                    (_numeric(source.get(alias)) for alias in aliases if _numeric(source.get(alias)) is not None),
                    None,
                )
                if value is not None:
                    candidate[canonical] = int(value)
            if candidate:
                latest = candidate
    return latest


def _codex_session_id(events: Iterable[dict[str, Any]]) -> str | None:
    session_id: str | None = None
    for event in events:
        value = event.get("thread_id") or event.get("session_id")
        if value is None and isinstance(event.get("thread"), dict):
            value = event["thread"].get("id")
        if value:
            session_id = str(value)
    return session_id


def _codex_final_text(events: Iterable[dict[str, Any]]) -> str:
    final_text = ""
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type not in {"item.completed", "turn.completed", "assistant_message", "result", "agent_end"}:
            continue
        for key in ("text", "message", "content", "result", "item", "last_message"):
            text = _text_value(event.get(key))
            if text:
                final_text = text
    return final_text


def normalize_codex_output(
    stdout: str,
    *,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
) -> tuple[dict[str, Any], float | None, str, str | None, bool, bool]:
    """Normalize Codex JSONL output.

    Returns ``(usage, cost, final_text, session_id, completed, cost_known)``.
    Codex does not emit provider USD pricing, so ``cost_known`` is true only
    when both rates and both input/output token totals are available.
    """
    events = list(_iter_json_lines(stdout))
    usage = _codex_usage(events)
    session_id = _codex_session_id(events)
    final_text = _codex_final_text(events)
    completed = any(
        str(event.get("type", "")) in {"turn.completed", "agent_end"} for event in events
    )
    cost_known = (
        input_cost_per_million is not None
        and output_cost_per_million is not None
        and "input_tokens" in usage
        and "output_tokens" in usage
    )
    cost: float | None = None
    if cost_known:
        assert input_cost_per_million is not None
        assert output_cost_per_million is not None
        cost = (
            usage["input_tokens"] * input_cost_per_million
            + usage["output_tokens"] * output_cost_per_million
        ) / 1_000_000.0
    return usage, cost, final_text, session_id, completed, cost_known


def validate_codex_pricing(
    max_token_cost: float | None,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> None:
    """Validate Codex pricing before a backend process is launched."""
    if max_token_cost is not None and (
        isinstance(max_token_cost, bool)
        or not isinstance(max_token_cost, (int, float))
        or not math.isfinite(max_token_cost)
        or max_token_cost < 0
    ):
        raise ValueError("max_token_cost must be a finite non-negative number or None")
    for name, value in (
        ("input_cost_per_million", input_cost_per_million),
        ("output_cost_per_million", output_cost_per_million),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number or None")
    if (input_cost_per_million is None) != (output_cost_per_million is None):
        raise ValueError(
            "Codex pricing must provide both input_cost_per_million and "
            "output_cost_per_million, or neither"
        )
    if max_token_cost is not None and (input_cost_per_million is None or output_cost_per_million is None):
        raise ValueError(
            "Codex with max_token_cost requires both input_cost_per_million "
            "and output_cost_per_million"
        )


@dataclass
class _PersistentProcess:
    process: subprocess.Popen[str]
    events: queue.Queue[tuple[str, str | None]]
    stdout: list[str]
    stderr: list[str]
    reader_threads: list[threading.Thread]


class PiAgentRunner:
    """Run Pi in JSON or persistent RPC mode.

    Args:
        command: Pi executable or an argv prefix.
        model: Pi provider/model string. ``None`` leaves model selection to Pi.
        persistent: Keep one RPC process alive across :meth:`run` calls.
        tools: Comma-separated Pi tool allowlist.
        sandbox: Require and use ``sandbox_prefix`` for every process.
        sandbox_prefix: Function returning an OS-level sandbox argv prefix.
    """

    def __init__(
        self,
        command: str | list[str] = "pi",
        *,
        model: str | None = None,
        persistent: bool = False,
        tools: str = "read,grep,find,ls,bash,edit,write",
        sandbox: bool = False,
        sandbox_prefix: SandboxPrefix | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = [command] if isinstance(command, str) else list(command)
        self.model = model
        self.persistent = persistent
        self.tools = tools
        self.sandbox = sandbox
        self.sandbox_prefix = sandbox_prefix
        self.env = env
        self._persistent: _PersistentProcess | None = None
        self._session_id: str | None = None

    def build_command(self, work_dir: Path, *, rpc: bool, prompt: str | None = None) -> list[str]:
        if self.sandbox:
            if self.sandbox_prefix is None:
                raise RuntimeError("Pi sandboxing was requested without an OS sandbox prefix")
            command = self.sandbox_prefix(work_dir)
        else:
            command = []
        command += [*self.command, "--mode", "rpc" if rpc else "json"]
        command += [
            "--no-session",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-approve",
            "--tools",
            self.tools,
        ]
        if self.model:
            command += ["--model", self.model]
        if prompt is not None:
            command += ["--print", prompt]
        return command

    def run(
        self,
        prompt: str,
        *,
        work_dir: Path,
        timeout_seconds: float | None = None,
        progress_check: ProgressCheck | None = None,
        max_budget_usd: float | None = None,
    ) -> AgentRunResult:
        work_dir = Path(work_dir)
        if self.persistent:
            return self._run_rpc(
                prompt,
                work_dir=work_dir,
                timeout_seconds=timeout_seconds,
                progress_check=progress_check,
                max_budget_usd=max_budget_usd,
            )
        return self._run_json(
            prompt,
            work_dir=work_dir,
            timeout_seconds=timeout_seconds,
            progress_check=progress_check,
            max_budget_usd=max_budget_usd,
        )

    def _run_json(
        self,
        prompt: str,
        *,
        work_dir: Path,
        timeout_seconds: float | None,
        progress_check: ProgressCheck | None,
        max_budget_usd: float | None,
    ) -> AgentRunResult:
        command = self.build_command(work_dir, rpc=False, prompt=prompt)
        running = self._start_persistent(command, work_dir)
        proc = running.process
        if proc.stdin is not None:
            proc.stdin.close()
        started = time.monotonic()
        timed_out = False
        reason: str | None = None
        while proc.poll() is None:
            self._drain_queued(running)
            reason = progress_check() if progress_check is not None else None
            if reason:
                timed_out = True
                self._terminate(proc)
                break
            if max_budget_usd is not None:
                _usage, cost, _text, _completed = normalize_pi_output("".join(running.stdout))
                if cost > max_budget_usd:
                    timed_out = True
                    reason = f"PI_TOKEN_BUDGET: terminated Pi above the ${max_budget_usd:.6f} cap."
                    self._terminate(proc)
                    break
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                timed_out = True
                reason = f"PI_TIMEOUT: terminated Pi after {timeout_seconds:.1f}s."
                self._terminate(proc)
                break
            time.sleep(0.05)
        self._drain_queued(running)
        if proc.poll() is None:
            proc.wait()
        for thread in running.reader_threads:
            thread.join(timeout=1)
        self._drain_queued(running)
        stdout = "".join(running.stdout)
        stderr = "".join(running.stderr)
        if reason:
            stderr = (stderr or "") + f"\n{reason}\n"
        usage, cost, final_text, completed = normalize_pi_output(stdout or "")
        return AgentRunResult(
            command=tuple(command),
            returncode=proc.returncode if proc.returncode is not None else -signal.SIGTERM,
            stdout=stdout or "",
            stderr=stderr or "",
            session_id=self._session_id or str(uuid.uuid4()),
            usage=usage,
            cost_usd=cost,
            timed_out=timed_out,
            completed=completed and not timed_out,
            final_text=final_text,
        )

    def _run_rpc(
        self,
        prompt: str,
        *,
        work_dir: Path,
        timeout_seconds: float | None,
        progress_check: ProgressCheck | None,
        max_budget_usd: float | None,
    ) -> AgentRunResult:
        if self._persistent is None:
            command = self.build_command(work_dir, rpc=True)
            self._persistent = self._start_persistent(command, work_dir)
            self._session_id = str(uuid.uuid4())
        running = self._persistent
        assert running is not None
        process = running.process
        command = self.build_command(work_dir, rpc=True)
        stdout_start = len(running.stdout)
        stderr_start = len(running.stderr)
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps({"type": "prompt", "message": prompt}) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            return AgentRunResult(
                command=tuple(command),
                returncode=process.returncode if process.returncode is not None else 1,
                stderr=f"Pi RPC session could not receive prompt: {exc}",
                session_id=self._session_id,
            )

        started = time.monotonic()
        completed = False
        timed_out = False
        reason: str | None = None
        while not completed:
            reason = progress_check() if progress_check is not None else None
            if reason:
                timed_out = True
                break
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                timed_out = True
                reason = f"PI_TIMEOUT: terminated Pi RPC after {timeout_seconds:.1f}s."
                break
            try:
                stream, line = running.events.get(timeout=0.05)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if stream == "stdout":
                if line is not None:
                    running.stdout.append(line)
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        event = None
                    if isinstance(event, dict) and event.get("type") == "agent_end":
                        completed = True
                    if max_budget_usd is not None:
                        _usage, cost, _text, _complete = normalize_pi_output(line)
                        if cost > max_budget_usd:
                            timed_out = True
                            reason = f"PI_TOKEN_BUDGET: terminated Pi above the ${max_budget_usd:.6f} cap."
                            break
            elif line is not None:
                running.stderr.append(line)
        if timed_out:
            self._terminate(process)
            self._persistent = None
        # Keep the RPC process alive after agent_end.  Drain only already queued
        # output so the next prompt starts with a clean event boundary.
        while True:
            try:
                stream, line = running.events.get_nowait()
            except queue.Empty:
                break
            if stream == "stdout" and line is not None:
                running.stdout.append(line)
            elif line is not None:
                running.stderr.append(line)
        stdout = "".join(running.stdout[stdout_start:])
        stderr = "".join(running.stderr[stderr_start:])
        if reason:
            stderr += f"\n{reason}\n"
        usage, cost, final_text, parsed_completed = normalize_pi_output(stdout)
        return AgentRunResult(
            command=tuple(command),
            returncode=0 if completed and not timed_out else (process.returncode or -signal.SIGTERM),
            stdout=stdout,
            stderr=stderr,
            session_id=self._session_id,
            usage=usage,
            cost_usd=cost,
            timed_out=timed_out,
            completed=completed and parsed_completed and not timed_out,
            final_text=final_text,
        )

    @staticmethod
    def _start(command: list[str], work_dir: Path, *, stdin: Any) -> subprocess.Popen[str]:
        kwargs: dict[str, Any] = {
            "cwd": str(work_dir),
            "stdin": stdin,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        return subprocess.Popen(command, **kwargs)

    def _start_persistent(self, command: list[str], work_dir: Path) -> _PersistentProcess:
        kwargs: dict[str, Any] = {
            "cwd": str(work_dir),
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if self.env is not None:
            kwargs["env"] = self.env
        if os.name == "posix":
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            raise RuntimeError(f"failed to start Pi: {exc}") from exc
        events: queue.Queue[tuple[str, str | None]] = queue.Queue()
        stdout: list[str] = []
        stderr: list[str] = []

        def read_stream(stream: Any, name: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    events.put((name, line))
            finally:
                events.put((name, None))

        threads = [
            threading.Thread(target=read_stream, args=(process.stdout, "stdout"), daemon=True),
            threading.Thread(target=read_stream, args=(process.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()
        return _PersistentProcess(process, events, stdout, stderr, threads)

    @staticmethod
    def _drain_queued(running: _PersistentProcess) -> None:
        while True:
            try:
                stream, line = running.events.get_nowait()
            except queue.Empty:
                return
            if stream == "stdout" and line is not None:
                running.stdout.append(line)
            elif line is not None:
                running.stderr.append(line)

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        if os.name == "posix" and getattr(proc, "pid", None) is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def close(self) -> None:
        running = self._persistent
        self._persistent = None
        if running is not None:
            if running.process.stdin is not None:
                try:
                    running.process.stdin.close()
                except OSError:
                    pass
            if running.process.poll() is None:
                self._terminate(running.process)


class CodexAgentRunner:
    """Run Codex CLI agent turns in the workspace-write sandbox.

    ``persistent=True`` means that each call is a new ``codex exec resume``
    process attached to one persisted Codex thread. It does not keep a CLI
    process alive between calls. ``persistent=False`` uses an ephemeral thread
    for each call.
    """

    def __init__(
        self,
        command: str | list[str] = "codex",
        *,
        model: str | None = None,
        persistent: bool = False,
        sandbox: bool = True,
        env: dict[str, str] | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        if not sandbox:
            raise ValueError("CodexAgentRunner requires sandbox=True (workspace-write)")
        validate_codex_pricing(None, input_cost_per_million, output_cost_per_million)
        self.command = [command] if isinstance(command, str) else list(command)
        if not self.command:
            raise ValueError("CodexAgentRunner requires a non-empty command")
        self.model = model
        self.persistent = persistent
        self.env = env
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self._session_id: str | None = None

    def build_command(self, work_dir: Path, output_path: Path, prompt: str) -> list[str]:
        command = [*self.command, "exec"]
        resuming = self.persistent and self._session_id is not None
        if resuming:
            command.append("resume")
        else:
            if not self.persistent:
                command.append("--ephemeral")
        command.extend(
            [
                "--config",
                'approval_policy="never"',
                "--config",
                "sandbox_workspace_write.network_access=true",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--json",
                "--output-last-message",
                str(output_path),
            ]
        )
        if not resuming:
            command.extend(["--cd", str(work_dir), "--sandbox", "workspace-write"])
        if self.model:
            command.extend(["--model", self.model])
        if resuming:
            command.append(self._session_id or "")
        command.append(prompt)
        return command

    def run(
        self,
        prompt: str,
        *,
        work_dir: Path,
        timeout_seconds: float | None = None,
        progress_check: ProgressCheck | None = None,
        max_budget_usd: float | None = None,
    ) -> AgentRunResult:
        validate_codex_pricing(
            max_budget_usd,
            self.input_cost_per_million,
            self.output_cost_per_million,
        )
        work_dir = Path(work_dir)
        log_dir = work_dir / ".codex-runner"
        log_dir.mkdir(parents=True, exist_ok=True)
        output_path = log_dir / f"last-message-{uuid.uuid4().hex}.txt"
        command = self.build_command(work_dir, output_path, prompt)
        running = self._start_capture(command, work_dir)
        proc = running.process
        started = time.monotonic()
        timed_out = False
        reason: str | None = None

        while proc.poll() is None:
            self._drain_queued(running)
            reason = progress_check() if progress_check is not None else None
            stdout = "".join(running.stdout)
            _usage, cost, _text, _session, _completed, cost_known = normalize_codex_output(
                stdout,
                input_cost_per_million=self.input_cost_per_million,
                output_cost_per_million=self.output_cost_per_million,
            )
            if reason:
                timed_out = True
                self._terminate(proc)
                break
            if max_budget_usd is not None and cost_known and cost is not None and cost > max_budget_usd:
                timed_out = True
                reason = f"CODEX_TOKEN_BUDGET: terminated Codex above the ${max_budget_usd:.6f} cap."
                self._terminate(proc)
                break
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                timed_out = True
                reason = f"CODEX_TIMEOUT: terminated Codex after {timeout_seconds:.1f}s."
                self._terminate(proc)
                break
            time.sleep(0.05)

        self._drain_queued(running)
        if proc.poll() is None:
            proc.wait()
        for thread in running.reader_threads:
            thread.join(timeout=1)
        self._drain_queued(running)
        stdout = "".join(running.stdout)
        stderr = "".join(running.stderr)
        if reason:
            stderr += f"\n{reason}\n"
        usage, cost, final_text, session_id, completed, cost_known = normalize_codex_output(
            stdout,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
        )
        if output_path.exists():
            saved_text = output_path.read_text(encoding="utf-8")
            if saved_text.strip():
                final_text = saved_text.strip()
        if session_id is not None:
            self._session_id = session_id
        effective_session_id = session_id or self._session_id
        returncode = proc.returncode if proc.returncode is not None else -signal.SIGTERM
        budget_exceeded = False
        if max_budget_usd is not None and cost_known and cost is not None and cost > max_budget_usd and not timed_out:
            budget_exceeded = True
            stderr += f"\nCODEX_TOKEN_BUDGET: Codex exited above the ${max_budget_usd:.6f} cap.\n"
            returncode = 1
        if max_budget_usd is not None and not cost_known and not timed_out:
            stderr += "\nCODEX_USAGE_MISSING: cannot enforce the USD cap without Codex token usage.\n"
            returncode = 1
        if returncode == 0 and not completed and not timed_out:
            stderr += "\nCODEX_MALFORMED_OUTPUT: Codex exited without turn.completed.\n"
            returncode = 1
        if returncode != 0 and not timed_out and not budget_exceeded and "CODEX_" not in stderr:
            stderr += f"\nCODEX_PROCESS_ERROR: Codex exited with status {returncode}.\n"
        if returncode == 0 and effective_session_id is None:
            stderr += "\nCODEX_MISSING_SESSION_ID: Codex did not emit a resumable thread id.\n"
            returncode = 1
            completed = False
        (log_dir / f"{output_path.stem}.jsonl").write_text(stdout, encoding="utf-8")
        (log_dir / f"{output_path.stem}.stderr").write_text(stderr, encoding="utf-8")
        (log_dir / f"{output_path.stem}.command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        return AgentRunResult(
            command=tuple(command),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            session_id=effective_session_id,
            usage=usage,
            cost_usd=cost,
            cost_known=cost_known,
            timed_out=timed_out,
            completed=completed and not timed_out and returncode == 0,
            final_text=final_text,
        )

    def _start_capture(self, command: list[str], work_dir: Path) -> _PersistentProcess:
        kwargs: dict[str, Any] = {
            "cwd": str(work_dir),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if self.env is not None:
            process_env = dict(self.env)
        else:
            process_env = dict(os.environ)
        if process_env.get("TERM") in {None, "", "dumb"}:
            # Codex's non-interactive exec path still initializes a terminal
            # UI when TERM=dumb. Give the subprocess a normal terminal type;
            # this does not alter its workspace-write sandbox posture.
            process_env["TERM"] = "xterm-256color"
        kwargs["env"] = process_env
        if os.name == "posix":
            kwargs["start_new_session"] = True
        executable = shutil.which(self.command[0]) or self.command[0]
        command = [executable, *command[1:]] if command else ["codex"]
        try:
            process = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            raise RuntimeError(f"failed to start Codex: {exc}") from exc
        events: queue.Queue[tuple[str, str | None]] = queue.Queue()
        stdout: list[str] = []
        stderr: list[str] = []

        def read_stream(stream: Any, name: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    events.put((name, line))
            finally:
                events.put((name, None))

        threads = [
            threading.Thread(target=read_stream, args=(process.stdout, "stdout"), daemon=True),
            threading.Thread(target=read_stream, args=(process.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()
        return _PersistentProcess(process, events, stdout, stderr, threads)

    @staticmethod
    def _drain_queued(running: _PersistentProcess) -> None:
        while True:
            try:
                stream, line = running.events.get_nowait()
            except queue.Empty:
                return
            if stream == "stdout" and line is not None:
                running.stdout.append(line)
            elif line is not None:
                running.stderr.append(line)

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        PiAgentRunner._terminate(proc)

    def close(self) -> None:
        """Close runner state without deleting persisted Codex sessions."""
        return


__all__ = [
    "AgentRunResult",
    "AgentRunner",
    "CodexAgentRunner",
    "PiAgentRunner",
    "normalize_codex_output",
    "normalize_pi_output",
    "validate_codex_pricing",
]
