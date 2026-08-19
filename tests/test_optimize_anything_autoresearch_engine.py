import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engines.autoresearch import AutoResearchEngine, _RunningClaudeProcess, _terminate_claude_process
from gepa.oa.task import Task


@pytest.fixture(autouse=True)
def _skip_claude_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests mock the claude subprocess; skip the CLI/bwrap preflight."""
    monkeypatch.setattr("gepa.oa.engines.autoresearch.preflight_claude_engine", lambda *a, **k: None)


class _FakeServer:
    def __init__(self) -> None:
        self.budget = BudgetTracker(max_evals=10)
        self.url = "http://127.0.0.1:9"
        self.best_score = 0.0
        self.eval_log = []


class _FakePopen:
    """Stands in for subprocess.Popen: the engine polls until done, then communicates."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def poll(self) -> int:
        return self.returncode

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        del timeout
        return self._stdout, self._stderr

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _HangingFakePopen(_FakePopen):
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        super().__init__(-15, stdout, stderr)
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else self.returncode

    def terminate(self) -> None:
        self._running = False

    def kill(self) -> None:
        self._running = False


def _engine_with_no_eval_watchdog(seconds: float) -> AutoResearchEngine:
    return AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=False,
            engine_config={"ralph": False, "max_no_eval_seconds": seconds},
        )
    )


def _run_once(engine: AutoResearchEngine, work_dir: Path, budget: BudgetTracker) -> subprocess.CompletedProcess[str]:
    return engine._run_claude(
        work_dir=work_dir,
        session_id="test-session",
        prompt="test prompt",
        budget=budget,
        adapter_cost=0.0,
        resume=False,
        env=dict(os.environ),
    )


def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    real_sleep = time.sleep
    monkeypatch.setattr("gepa.oa.engines.autoresearch.time.sleep", lambda _: real_sleep(0.01))
    return real_sleep


def test_autoresearch_drains_large_stdout_and_stderr_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_size = 256 * 1024
    original_popen = subprocess.Popen
    child_command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            f"sys.stdout.write('o' * {output_size}); sys.stdout.flush(); "
            f"sys.stderr.write('e' * {output_size}); sys.stderr.flush()"
        ),
    ]

    def launch_child(_: list[str], **kwargs: object) -> subprocess.Popen[str]:
        return original_popen(child_command, **kwargs)

    _fast_sleep(monkeypatch)
    engine = _engine_with_no_eval_watchdog(1.0)
    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=launch_child):
        completed = _run_once(engine, tmp_path, BudgetTracker(max_evals=1))

    assert completed.returncode == 0, (len(completed.stdout), len(completed.stderr))
    assert completed.stdout == "o" * output_size
    assert completed.stderr == "e" * output_size


@pytest.mark.skipif(os.name != "posix", reason="process-group signalling is POSIX-specific")
def test_autoresearch_watchdog_terminates_posix_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grandchild_pid_file = tmp_path / "grandchild.pid"
    grandchild_alive_file = tmp_path / "grandchild.alive"
    original_popen = subprocess.Popen
    grandchild_command = [
        sys.executable,
        "-c",
        (
            "import pathlib, time; "
            "time.sleep(1); "
            f"pathlib.Path({str(grandchild_alive_file)!r}).write_text('alive'); "
            "time.sleep(30)"
        ),
    ]
    child_command = [
        sys.executable,
        "-c",
        (
            "import pathlib, subprocess, sys, time; "
            f"pid = subprocess.Popen({grandchild_command!r}).pid; "
            f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(pid)); "
            "time.sleep(30)"
        ),
    ]

    def launch_child(_: list[str], **kwargs: object) -> subprocess.Popen[str]:
        proc = original_popen(child_command, **kwargs)
        deadline = time.monotonic() + 2.0
        while not grandchild_pid_file.exists() and time.monotonic() < deadline:
            real_sleep(0.01)
        return proc

    real_sleep = _fast_sleep(monkeypatch)
    engine = _engine_with_no_eval_watchdog(0.1)
    try:
        started = time.monotonic()
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=launch_child):
            completed = _run_once(engine, tmp_path, BudgetTracker(max_evals=1))
        elapsed = time.monotonic() - started
        assert grandchild_pid_file.exists()
        assert completed.returncode != 0
        assert elapsed < 2.0
        real_sleep(1.1)
        assert not grandchild_alive_file.exists()
    finally:
        if grandchild_pid_file.exists():
            try:
                os.kill(int(grandchild_pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_autoresearch_no_eval_watchdog_preserves_reason_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine_with_no_eval_watchdog(0.0)
    _fast_sleep(monkeypatch)
    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", return_value=_HangingFakePopen()):
        completed = _run_once(engine, tmp_path, BudgetTracker(max_evals=1))

    assert "NO_EVAL_PROGRESS" in completed.stderr


def test_autoresearch_budget_watchdog_preserves_reason_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    budget = BudgetTracker(max_evals=1)
    budget.record(0.0)
    engine = _engine_with_no_eval_watchdog(60.0)
    _fast_sleep(monkeypatch)
    monkeypatch.setattr("gepa.oa.engines.autoresearch._BUDGET_EXHAUSTION_GRACE_SECONDS", 0.0)
    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", return_value=_HangingFakePopen()):
        completed = _run_once(engine, tmp_path, budget)

    assert "BUDGET_EXHAUSTED" in completed.stderr


def test_autoresearch_kills_unresponsive_process_without_sigkill_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    class _UnresponsiveFakePopen(_FakePopen):
        def __init__(self) -> None:
            super().__init__(-9, "")
            self.terminate_calls = 0
            self.kill_calls = 0

        def terminate(self) -> None:
            self.terminate_calls += 1

        def wait(self, timeout: float | None = None) -> int:
            if self.kill_calls == 0:
                raise subprocess.TimeoutExpired("claude", timeout)
            return self.returncode

        def kill(self) -> None:
            self.kill_calls += 1

    fake = _UnresponsiveFakePopen()
    monkeypatch.setattr("gepa.oa.engines.autoresearch.os.name", "nt")
    monkeypatch.delattr("gepa.oa.engines.autoresearch.signal.SIGKILL", raising=False)

    _terminate_claude_process(_RunningClaudeProcess(fake, StringIO(), StringIO()))

    assert fake.terminate_calls == 1
    assert fake.kill_calls == 1


def test_autoresearch_uses_direct_process_fallback_on_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine_with_no_eval_watchdog(0.0)
    fake = _HangingFakePopen()
    captured_kwargs: dict[str, object] = {}

    def capture_popen(_: list[str], **kwargs: object) -> _HangingFakePopen:
        captured_kwargs.update(kwargs)
        return fake

    _fast_sleep(monkeypatch)
    monkeypatch.setattr("gepa.oa.engines.autoresearch.os.name", "nt")
    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=capture_popen):
        _run_once(engine, tmp_path, BudgetTracker(max_evals=1))

    assert "start_new_session" not in captured_kwargs


def test_autoresearch_engine_ralph_resumes_with_remaining_budget(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", seed_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        cost = 0.2 if len(calls) == 1 else 0.0005
        return _FakePopen(0, json.dumps({"total_cost_usd": cost}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path), max_token_cost=1.0, engine_config={}
        )
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 2
    assert "--session-id" in calls[0]
    assert "--resume" not in calls[0]
    assert "--resume" in calls[1]
    assert calls[1][calls[1].index("--max-budget-usd") + 1] == "0.800000"
    assert result.best_candidate == "candidate"
    assert result.metadata["adapter_cost"] == 0.2005
    assert result.metadata["ralph_iterations"] == 2


def test_autoresearch_engine_can_disable_ralph(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", seed_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={"ralph": False}
        )
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 1
    assert "--session-id" in calls[0]
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_engine_string_false_disables_ralph(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", seed_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={"ralph": "false"}
        )
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 1
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_engine_ralph_respects_stop_at_score(tmp_path: Path) -> None:
    server = _FakeServer()
    server.best_score = 1.0
    task = Task(name="smoke", seed_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path), stop_at_score=1.0, engine_config={}
        )
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 1
    assert result.metadata["adapter_cost"] == 0.2
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_engine_counts_failed_resume_cost(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", seed_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        if len(calls) == 1:
            return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))
        return _FakePopen(1, json.dumps({"total_cost_usd": 0.1}), stderr="failed")

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={})
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 2
    assert result.metadata["adapter_cost"] == 0.30000000000000004
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_engine_materializes_optimize_anything_handoff(tmp_path: Path) -> None:
    server = _FakeServer()
    source = tmp_path / "source"
    source.mkdir()
    (source / "summary.json").write_text(json.dumps({"stage_idx": 0, "best_score": 0.7}))
    (source / "best_candidate.txt").write_text("prior-best")
    evals = source / "evals"
    evals.mkdir()
    (evals / "0.json").write_text(json.dumps({"score": 0.7, "candidate": "prior"}))
    task = Task(name="smoke", seed_candidate="seed")
    handoffs = [
        {
            "stage_idx": 0,
            "engine": "gepa",
            "best_score": 0.7,
            "num_evals": 1,
            "summary_path": str(source / "summary.json"),
            "best_candidate_path": str(source / "best_candidate.txt"),
            "eval_trace_dir": str(evals),
        }
    ]

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        del cmd
        work_dir = Path(str(kwargs["cwd"]))
        assert (work_dir / "handoff" / "index.json").exists()
        assert (work_dir / "handoff" / "stage_00_gepa" / "summary.json").exists()
        assert (work_dir / "handoff" / "stage_00_gepa" / "best_candidate.txt").read_text() == "prior-best"
        assert (work_dir / "handoff" / "stage_00_gepa" / "evals" / "0.json").exists()
        assert "Prior Optimizer Handoff" in (work_dir / "program.md").read_text()
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            run_dir=str(tmp_path / "run"),
            sandbox=False,
            engine_config={"ralph": False, "handoffs": handoffs},
        )
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert result.best_candidate == "candidate"
