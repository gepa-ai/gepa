import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engine import Result
from gepa.oa.engines.autoresearch import AutoResearchEngine
from gepa.oa.eval_server import EvalServer, EvaluationSessionClosedError
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
        self.best_candidate = "seed"
        self.eval_log = []

    def open_evaluation_session(self, _initial_candidate: str) -> str:
        return "test-session"

    def close_evaluation_session(self, _session_id: str, *, timeout: float) -> tuple[str, float]:
        del timeout
        return self.best_candidate, self.best_score

    def evaluation_session_aggregate(self, _session_id: str) -> tuple[str, float] | None:
        return None

    def wait_for_idle(self, _session_id: str, *, timeout: float) -> bool:
        del timeout
        return True


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
    assert result.best_candidate == "seed"
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


def test_autoresearch_waits_for_active_evaluation_and_ignores_tampered_best_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    returned = threading.Event()

    def evaluate(candidate: str) -> tuple[float, dict[str, object]]:
        assert candidate == "server-winner"
        started.set()
        assert release.wait(timeout=2.0)
        return 0.9, {}

    task = Task(name="race", seed_candidate="seed")
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=2), max_concurrency=1)
    server.start()
    session_ids: list[str] = []
    original_open = server.open_evaluation_session

    def open_evaluation_session(initial_candidate: str) -> str:
        session_id = original_open(initial_candidate)
        session_ids.append(session_id)
        return session_id

    monkeypatch.setattr(server, "open_evaluation_session", open_evaluation_session)

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        work_dir = Path(str(kwargs["cwd"]))
        (work_dir / "best_candidate.txt").write_text("tampered-workspace-file")
        threading.Thread(
            target=lambda: server.evaluate("server-winner", evaluation_session_id=session_ids[0]), daemon=True
        ).start()
        assert started.wait(timeout=2.0)
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={"ralph": False}
        )
    )
    outcome: dict[str, object] = {}

    def run() -> None:
        outcome["result"] = engine.run(task, server)
        returned.set()

    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            runner = threading.Thread(target=run)
            runner.start()
            returned_before_completion = returned.wait(timeout=0.1)
            release.set()
            runner.join(timeout=2.0)
    finally:
        release.set()
        server.stop()

    assert not returned_before_completion
    assert returned.is_set()
    result = outcome["result"]
    assert isinstance(result, Result)
    assert result.best_candidate == "server-winner"
    assert result.best_score == 0.9


def test_autoresearch_rejects_delayed_evaluation_after_session_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admit_late_request = threading.Event()
    late_request_done = threading.Event()
    session_ids: list[str] = []
    late_errors: list[BaseException] = []

    task = Task(name="late", seed_candidate="seed")
    server = EvalServer(task, lambda candidate: (0.9, {"candidate": candidate}), BudgetTracker(max_evals=2))
    server.start()
    original_open = server.open_evaluation_session

    def open_evaluation_session(initial_candidate: str) -> str:
        session_id = original_open(initial_candidate)
        session_ids.append(session_id)
        return session_id

    monkeypatch.setattr(server, "open_evaluation_session", open_evaluation_session)

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("tampered-workspace-file")

        def submit_late_request() -> None:
            assert admit_late_request.wait(timeout=2.0)
            try:
                server.evaluate("late-winner", evaluation_session_id=session_ids[0])
            except BaseException as e:
                late_errors.append(e)
            finally:
                late_request_done.set()

        threading.Thread(target=submit_late_request, daemon=True).start()
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={"ralph": False}
        )
    )
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
        admit_late_request.set()
        assert late_request_done.wait(timeout=2.0)
    finally:
        admit_late_request.set()
        server.stop()

    assert result.best_candidate == "seed"
    assert result.best_score == float("-inf")
    assert session_ids[0] in (tmp_path / "eval.sh").read_text()
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], EvaluationSessionClosedError)
    assert server.best_candidate == "seed"


def test_autoresearch_fails_closed_when_admitted_evaluation_does_not_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    session_ids: list[str] = []

    def evaluate(_candidate: str) -> tuple[float, dict[str, object]]:
        started.set()
        assert release.wait(timeout=2.0)
        return 0.9, {}

    task = Task(name="hung", seed_candidate="seed")
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=2), max_concurrency=1)
    server.start()
    original_open = server.open_evaluation_session

    def open_evaluation_session(initial_candidate: str) -> str:
        session_id = original_open(initial_candidate)
        session_ids.append(session_id)
        return session_id

    monkeypatch.setattr(server, "open_evaluation_session", open_evaluation_session)

    def fake_popen(_cmd: list[str], **_kwargs: object) -> _FakePopen:
        threading.Thread(
            target=lambda: server.evaluate("hung", evaluation_session_id=session_ids[0]), daemon=True
        ).start()
        assert started.wait(timeout=2.0)
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=False,
            run_dir=str(tmp_path),
            engine_config={"ralph": False, "max_no_eval_seconds": 0.05},
        )
    )
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            with pytest.raises(RuntimeError, match="did not drain"):
                engine.run(task, server)
    finally:
        release.set()
        server.stop()


def test_autoresearch_wait_keeps_shared_server_reusable(tmp_path: Path) -> None:
    task = Task(name="reuse", seed_candidate="seed")
    scores = {"first": 0.9, "second": 0.5}
    server = EvalServer(task, lambda candidate: (scores[candidate], {}), BudgetTracker(max_evals=3))
    server.start()
    invocations = 0

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        nonlocal invocations
        invocations += 1
        candidate = "first" if invocations == 1 else "second"
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        session_id = next(reversed(server._evaluation_sessions))
        server.evaluate(candidate, evaluation_session_id=session_id)
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    first = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path / "first"), engine_config={"ralph": False}
        )
    )
    second = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path / "second"), engine_config={"ralph": False}
        )
    )
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            first_result = first.run(task, server)
            second_result = second.run(task, server)
    finally:
        server.stop()

    assert first_result.best_candidate == "first"
    assert second_result.best_candidate == "second"
    assert second_result.best_score == 0.5


def test_autoresearch_single_task_tie_keeps_first_completed_candidate(tmp_path: Path) -> None:
    task = Task(name="tie", seed_candidate="seed")
    server = EvalServer(task, lambda _candidate: (0.5, {}), BudgetTracker(max_evals=3))
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        session_id = next(reversed(server._evaluation_sessions))
        server.evaluate("first", evaluation_session_id=session_id)
        server.evaluate("second", evaluation_session_id=session_id)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={"ralph": False}
        )
    )
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
    finally:
        server.stop()

    assert result.best_candidate == "first"
    assert result.best_score == 0.5


def test_autoresearch_does_not_resume_until_prior_evaluation_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[list[str]] = []

    def evaluate(candidate: str) -> tuple[float, dict[str, object]]:
        assert candidate == "first"
        started.set()
        assert release.wait(timeout=2.0)
        return 0.8, {}

    task = Task(name="resume", seed_candidate="seed")
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=3), max_concurrency=1)
    server.start()
    session_ids: list[str] = []
    original_open = server.open_evaluation_session

    def open_evaluation_session(initial_candidate: str) -> str:
        session_id = original_open(initial_candidate)
        session_ids.append(session_id)
        return session_id

    monkeypatch.setattr(server, "open_evaluation_session", open_evaluation_session)

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        if len(calls) == 1:
            threading.Thread(
                target=lambda: server.evaluate("first", evaluation_session_id=session_ids[0]), daemon=True
            ).start()
            assert started.wait(timeout=2.0)
            return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.0005}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={})
    )
    runner = threading.Thread(target=lambda: engine.run(task, server))
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            runner.start()
            assert not release.wait(timeout=0.1)
            assert len(calls) == 1
            release.set()
            runner.join(timeout=2.0)
    finally:
        release.set()
        server.stop()

    assert not runner.is_alive()
    assert len(calls) == 2
    assert "--resume" in calls[1]


def test_autoresearch_closes_each_ralph_iteration_before_opening_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(name="iteration-boundary", seed_candidate="seed")
    scores = {"first": 0.4, "second": 0.8}
    server = EvalServer(task, lambda candidate: (scores[candidate], {}), BudgetTracker(max_evals=3))
    server.start()
    session_ids: list[str] = []
    late_errors: list[BaseException] = []
    original_open = server.open_evaluation_session

    def open_evaluation_session(initial_candidate: str) -> str:
        session_id = original_open(initial_candidate)
        session_ids.append(session_id)
        return session_id

    monkeypatch.setattr(server, "open_evaluation_session", open_evaluation_session)
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        if len(calls) == 1:
            server.evaluate("first", evaluation_session_id=session_ids[0])
            return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

        assert len(session_ids) == 2
        work_dir = Path(str(kwargs["cwd"]))
        assert session_ids[1] in (work_dir / "eval-2.sh").read_text()
        assert str(work_dir / "eval-2.sh") in cmd[-1]
        with pytest.raises(EvaluationSessionClosedError) as raised:
            server.evaluate("first", evaluation_session_id=session_ids[0])
        late_errors.append(raised.value)
        server.evaluate("second", evaluation_session_id=session_ids[1])
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.0005}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={})
    )
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
    finally:
        server.stop()

    assert len(calls) == 2
    assert len(session_ids) == 2
    assert len(late_errors) == 1
    assert result.best_candidate == "second"
    assert result.best_score == 0.8


def test_autoresearch_closes_a_session_when_materialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(name="materialize-failure", seed_candidate="seed")
    server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=1))
    session_ids: list[str] = []
    original_open = server.open_evaluation_session

    def open_evaluation_session(initial_candidate: str) -> str:
        session_id = original_open(initial_candidate)
        session_ids.append(session_id)
        return session_id

    monkeypatch.setattr(server, "open_evaluation_session", open_evaluation_session)
    monkeypatch.setattr(
        "gepa.oa.engines.autoresearch._materialize_sandbox", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )
    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={"ralph": False})
    )

    with pytest.raises(OSError):
        engine.run(task, server)

    with pytest.raises(EvaluationSessionClosedError):
        server.evaluate("late", evaluation_session_id=session_ids[0])


def test_autoresearch_closes_a_session_when_claude_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(name="popen-failure", seed_candidate="seed")
    server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=1))
    server.start()
    session_ids: list[str] = []
    original_open = server.open_evaluation_session

    def open_evaluation_session(initial_candidate: str) -> str:
        session_id = original_open(initial_candidate)
        session_ids.append(session_id)
        return session_id

    monkeypatch.setattr(server, "open_evaluation_session", open_evaluation_session)
    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", sandbox=False, run_dir=str(tmp_path), engine_config={"ralph": False})
    )

    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=OSError("cannot start")):
            with pytest.raises(OSError, match="cannot start"):
                engine.run(task, server)

        with pytest.raises(EvaluationSessionClosedError):
            server.evaluate("late", evaluation_session_id=session_ids[0])
    finally:
        server.stop()


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

    assert result.best_candidate == "seed"
