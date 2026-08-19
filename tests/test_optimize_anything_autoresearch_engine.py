import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engine import Result
from gepa.oa.engines.autoresearch import AutoResearchEngine, _build_program_md
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task


@pytest.fixture(autouse=True)
def _skip_claude_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests mock the claude subprocess; skip the CLI/bwrap preflight."""
    monkeypatch.setattr("gepa.oa.engines.autoresearch.preflight_claude_engine", lambda *a, **k: None)


class _FakeServer:
    def __init__(self) -> None:
        self.budget = BudgetTracker(max_evals=10)
        self.url = "http://127.0.0.1:9"
        self.best_score = float("-inf")
        self.best_candidate = "seed"
        self.eval_log = []
        self.progress_log: list[dict[str, object]] = []
        self._candidate_registry: dict[str, int] = {}

    def drain_http(self, *, timeout: float, quiet: float) -> bool:
        del timeout, quiet
        return True

    def pause_http(self) -> None:
        return None

    def resume_http(self) -> None:
        return None

    def wait_idle(self, timeout: float | None = None) -> bool:
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


def _engine(tmp_path: Path, **engine_config: object) -> AutoResearchEngine:
    merged: dict[str, object] = {
        "ralph": False,
        "drain_quiet_seconds": 0.0,
        "drain_timeout_seconds": 5.0,
    }
    merged.update(engine_config)
    run_dir = str(merged.pop("run_dir", str(tmp_path)))
    stop_at_score = merged.pop("stop_at_score", None)
    max_token_cost = merged.pop("max_token_cost", None)
    config_kwargs: dict[str, object] = {
        "engine": "autoresearch",
        "sandbox": False,
        "run_dir": run_dir,
        "engine_config": merged,
    }
    if stop_at_score is not None:
        config_kwargs["stop_at_score"] = stop_at_score
    if max_token_cost is not None:
        config_kwargs["max_token_cost"] = max_token_cost
    return AutoResearchEngine(OptimizeAnythingConfig(**config_kwargs))  # type: ignore[arg-type]


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


def _http_evaluate(server: EvalServer, candidate: str) -> dict[str, object]:
    req = urllib.request.Request(
        f"{server.url}/evaluate",
        data=json.dumps({"candidate": candidate}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _http_evaluate_examples(server: EvalServer, candidate: str, example_ids: list[str] | None = None) -> None:
    body: dict[str, object] = {"candidate": candidate}
    if example_ids is not None:
        body["example_ids"] = example_ids
    req = urllib.request.Request(
        f"{server.url}/evaluate_examples",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=5)


def test_autoresearch_waits_for_inflight_eval_and_ignores_the_workspace_file(tmp_path: Path) -> None:
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

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("tampered-workspace-file")
        threading.Thread(target=lambda: server.evaluate("server-winner"), daemon=True).start()
        assert started.wait(timeout=2.0)
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
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
    assert result.best_candidate == "server-winner"
    assert result.best_score == 0.9


def test_autoresearch_waits_for_inflight_http_eval(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    returned = threading.Event()

    def evaluate(candidate: str) -> tuple[float, dict[str, object]]:
        if candidate == "http-winner":
            started.set()
            assert release.wait(timeout=2.0)
            return 0.9, {}
        return 0.0, {}

    task = Task(name="http-race", seed_candidate="seed")
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=2), max_concurrency=1)
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("tampered-workspace-file")
        threading.Thread(target=lambda: _http_evaluate(server, "http-winner"), daemon=True).start()
        assert started.wait(timeout=2.0)
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
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
    assert result.best_candidate == "http-winner"
    assert result.best_score == 0.9


def test_autoresearch_dataset_selects_full_pool_winner_not_per_example_spike(tmp_path: Path) -> None:
    def evaluate(candidate: str, example: object) -> tuple[float, dict[str, object]]:
        if candidate == "spiky":
            return (1.0 if str(example) == "a" else 0.0), {}
        if candidate == "steady":
            return 0.6, {}
        return 0.0, {}

    task = Task(name="dataset-select", seed_candidate="seed", train_set=["a", "b"])
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=10), max_concurrency=2)
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        _http_evaluate_examples(server, "spiky")
        _http_evaluate_examples(server, "steady")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
    finally:
        server.stop()

    assert result.best_candidate == "steady"
    assert result.best_score == pytest.approx(0.6)


def test_autoresearch_dataset_subset_eval_returns_seed(tmp_path: Path) -> None:
    def evaluate(_candidate: str, example: object) -> tuple[float, dict[str, object]]:
        return (1.0 if str(example) == "a" else 0.0), {}

    task = Task(name="subset-only", seed_candidate="seed", train_set=["a", "b"])
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=10), max_concurrency=2)
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        subset_id = next(iter(server._split_ids["train"]))
        _http_evaluate_examples(server, "spiky", example_ids=[subset_id])
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
    finally:
        server.stop()

    assert result.best_candidate == "seed"
    assert result.best_score == float("-inf")


def test_autoresearch_dataset_per_example_spike_does_not_stop_ralph(tmp_path: Path) -> None:
    task = Task(name="spike-continue", seed_candidate="seed", train_set=["a", "b"])
    server = EvalServer(task, lambda _candidate, _example: (1.0, {}), BudgetTracker(max_evals=10))
    server.start()
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        del kwargs
        calls.append(cmd)
        if len(calls) == 1:
            server.evaluate("spiky", "a")
            return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.0005}))

    engine = _engine(tmp_path, ralph=True, stop_at_score=1.0)
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
    finally:
        server.stop()

    assert len(calls) == 2
    assert result.best_candidate == "seed"
    assert result.best_score == float("-inf")


def test_autoresearch_test_set_only_uses_single_task_tracking(tmp_path: Path) -> None:
    def evaluate(candidate: str, example: object | None = None) -> tuple[float, dict[str, object]]:
        del example
        return (0.8 if candidate == "winner" else 0.0), {}

    task = Task(name="test-only", seed_candidate="seed", test_set=["held-out"])
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=4))
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        _http_evaluate_examples(server, "winner")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
    finally:
        server.stop()

    assert result.best_candidate == "winner"
    assert result.best_score == 0.8


def test_process_result_keeps_agent_file_when_winner_differs(tmp_path: Path) -> None:
    work = tmp_path / "run"
    work.mkdir()
    (work / "best_candidate.txt").write_text("agent-pick")
    engine = _engine(work)
    result = Result(
        best_candidate="seed",
        best_score=float("-inf"),
        metadata={"work_dir": str(work), "session_id": "unused"},
    )
    engine.process_result(result, work)
    assert (work / "best_candidate.txt").read_text() == "seed"
    assert (work / "agent_best_candidate.txt").read_text() == "agent-pick"


def test_run_preserves_agent_file_when_syncing_tracked_winner(tmp_path: Path) -> None:
    def evaluate(candidate: str) -> tuple[float, dict[str, object]]:
        return (0.9 if candidate == "server-winner" else 0.0), {}

    task = Task(name="preserve-agent", seed_candidate="seed")
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=2))
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("agent-pick")
        _http_evaluate(server, "server-winner")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
        engine.process_result(result, tmp_path)
    finally:
        server.stop()

    assert result.best_candidate == "server-winner"
    assert (tmp_path / "best_candidate.txt").read_text() == "server-winner"
    assert (tmp_path / "agent_best_candidate.txt").read_text() == "agent-pick"


def test_autoresearch_waits_for_inflight_http_dataset_eval(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    returned = threading.Event()

    def evaluate(candidate: str, example: object) -> tuple[float, dict[str, object]]:
        del example
        if candidate == "http-pool":
            started.set()
            assert release.wait(timeout=2.0)
            return 0.7, {}
        return 0.0, {}

    task = Task(name="dataset-http-race", seed_candidate="seed", train_set=["a", "b"])
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=10), max_concurrency=1)
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        threading.Thread(target=lambda: _http_evaluate_examples(server, "http-pool"), daemon=True).start()
        assert started.wait(timeout=2.0)
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
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
            runner.join(timeout=5.0)
    finally:
        release.set()
        server.stop()

    assert not returned_before_completion
    assert returned.is_set()
    result = outcome["result"]
    assert result.best_candidate == "http-pool"
    assert result.best_score == pytest.approx(0.7)


def test_autoresearch_drain_timeout_returns_completed_work(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def evaluate(candidate: str) -> tuple[float, dict[str, object]]:
        if candidate == "slow":
            started.set()
            assert release.wait(timeout=5.0)
            return 0.99, {}
        return 0.4, {}

    task = Task(name="drain-timeout", seed_candidate="seed")
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=4), max_concurrency=1)
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        server.evaluate("done")
        threading.Thread(target=lambda: server.evaluate("slow"), daemon=True).start()
        assert started.wait(timeout=2.0)
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path, drain_timeout_seconds=0.2, drain_quiet_seconds=0.0)
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            started_at = time.monotonic()
            result = engine.run(task, server)
            assert time.monotonic() - started_at < 2.0
    finally:
        release.set()
        server.stop()

    assert result.best_candidate == "done"
    assert result.best_score == 0.4
    assert result.metadata["drain_timed_out"] is True


def test_autoresearch_ignores_validate_checkpoint(tmp_path: Path) -> None:
    def evaluate(candidate: str, example: object) -> tuple[float, dict[str, object]]:
        if candidate == "val-spike":
            return (1.0 if str(example) == "v" else 0.0), {}
        if candidate == "steady":
            return 0.6, {}
        return 0.0, {}

    task = Task(name="validate-ignore", seed_candidate="seed", train_set=["a", "b"], val_set=["v"])
    server = EvalServer(task, evaluate, BudgetTracker(max_evals=20), max_concurrency=2)
    server.start()

    def fake_popen(_cmd: list[str], **kwargs: object) -> _FakePopen:
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("workspace-only")
        req = urllib.request.Request(
            f"{server.url}/validate",
            data=json.dumps({"candidate": "val-spike"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        _http_evaluate_examples(server, "steady")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = _engine(tmp_path)
    try:
        with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
            result = engine.run(task, server)
    finally:
        server.stop()

    assert result.best_candidate == "steady"
    assert result.best_score == pytest.approx(0.6)


def test_program_md_tells_agent_full_pool_eval_selects_the_winner() -> None:
    task = Task(name="t", seed_candidate="seed", train_set=["a", "b"])
    text = _build_program_md(
        task,
        BudgetTracker(max_evals=10),
        max_token_cost=None,
        perfect_score=None,
        handoffs=None,
    )
    assert "full-pool" in text
    assert "ignores `best_candidate.txt`" in text


def test_eval_sh_treats_http_409_as_failure(tmp_path: Path) -> None:
    from gepa.oa.engines.autoresearch import EVAL_SCRIPT_SINGLE

    script = tmp_path / "eval.sh"
    script.write_text(EVAL_SCRIPT_SINGLE.format(server_url="http://127.0.0.1:9"))
    assert 'HTTP_CODE" = "409"' in script.read_text()
    assert "EVAL_SERVER_PAUSED" in script.read_text()
