import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from gepa.omni.backends.claude_code import ClaudeCodeBackend
from gepa.omni.budget import BudgetTracker
from gepa.omni.config import OmniConfig
from gepa.omni.task import Task


class _FakeServer:
    def __init__(self) -> None:
        self.budget = BudgetTracker(max_evals=10, max_token_cost=1.0)
        self.url = "http://127.0.0.1:9"
        self.best_score = 0.0
        self.eval_log = []


def test_claude_code_backend_ralph_resumes_with_remaining_budget(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        cost = 0.2 if len(calls) == 1 else 0.0005
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"total_cost_usd": cost}), stderr="")

    backend = ClaudeCodeBackend(OmniConfig(backend="claude_code", run_dir=str(tmp_path), config={}))

    with patch("gepa.omni.backends.claude_code.subprocess.run", side_effect=fake_run):
        result = backend.run(task, server)

    assert len(calls) == 2
    assert "--session-id" in calls[0]
    assert "--resume" not in calls[0]
    assert "--resume" in calls[1]
    assert calls[1][calls[1].index("--max-budget-usd") + 1] == "0.800000"
    assert result.best_candidate == "candidate"
    assert result.metadata["adapter_cost"] == 0.2005
    assert result.metadata["ralph_iterations"] == 2


def test_claude_code_backend_can_disable_ralph(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"total_cost_usd": 0.2}), stderr="")

    backend = ClaudeCodeBackend(
        OmniConfig(backend="claude_code", run_dir=str(tmp_path), config={"ralph": False})
    )

    with patch("gepa.omni.backends.claude_code.subprocess.run", side_effect=fake_run):
        result = backend.run(task, server)

    assert len(calls) == 1
    assert "--session-id" in calls[0]
    assert result.metadata["ralph_iterations"] == 1


def test_claude_code_backend_string_false_disables_ralph(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"total_cost_usd": 0.2}), stderr="")

    backend = ClaudeCodeBackend(
        OmniConfig(backend="claude_code", run_dir=str(tmp_path), config={"ralph": "false"})
    )

    with patch("gepa.omni.backends.claude_code.subprocess.run", side_effect=fake_run):
        result = backend.run(task, server)

    assert len(calls) == 1
    assert result.metadata["ralph_iterations"] == 1


def test_claude_code_backend_ralph_respects_stop_at_score(tmp_path: Path) -> None:
    server = _FakeServer()
    server.best_score = 1.0
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"total_cost_usd": 0.2}), stderr="")

    backend = ClaudeCodeBackend(
        OmniConfig(backend="claude_code", run_dir=str(tmp_path), stop_at_score=1.0, config={})
    )

    with patch("gepa.omni.backends.claude_code.subprocess.run", side_effect=fake_run):
        result = backend.run(task, server)

    assert len(calls) == 1
    assert result.metadata["adapter_cost"] == 0.2
    assert result.metadata["ralph_iterations"] == 1


def test_claude_code_backend_counts_failed_resume_cost(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"total_cost_usd": 0.2}), stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps({"total_cost_usd": 0.1}), stderr="failed")

    backend = ClaudeCodeBackend(OmniConfig(backend="claude_code", run_dir=str(tmp_path), config={}))

    with patch("gepa.omni.backends.claude_code.subprocess.run", side_effect=fake_run):
        result = backend.run(task, server)

    assert len(calls) == 2
    assert result.metadata["adapter_cost"] == 0.30000000000000004
    assert result.metadata["ralph_iterations"] == 1


def test_claude_code_backend_materializes_omni_handoff(tmp_path: Path) -> None:
    server = _FakeServer()
    source = tmp_path / "source"
    source.mkdir()
    (source / "summary.json").write_text(json.dumps({"stage_idx": 0, "best_score": 0.7}))
    (source / "best_candidate.txt").write_text("prior-best")
    evals = source / "evals"
    evals.mkdir()
    (evals / "0.json").write_text(json.dumps({"score": 0.7, "candidate": "prior"}))
    task = Task(
        name="smoke",
        initial_candidate="seed",
        metadata={
            "omni_handoffs": [
                {
                    "stage_idx": 0,
                    "backend": "gepa",
                    "best_score": 0.7,
                    "num_evals": 1,
                    "summary_path": str(source / "summary.json"),
                    "best_candidate_path": str(source / "best_candidate.txt"),
                    "eval_trace_dir": str(evals),
                }
            ]
        },
    )

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del cmd
        work_dir = Path(str(kwargs["cwd"]))
        assert (work_dir / "handoff" / "index.json").exists()
        assert (work_dir / "handoff" / "stage_00_gepa" / "summary.json").exists()
        assert (work_dir / "handoff" / "stage_00_gepa" / "best_candidate.txt").read_text() == "prior-best"
        assert (work_dir / "handoff" / "stage_00_gepa" / "evals" / "0.json").exists()
        assert "Prior Optimizer Handoff" in (work_dir / "program.md").read_text()
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return subprocess.CompletedProcess([], 0, stdout=json.dumps({"total_cost_usd": 0.2}), stderr="")

    backend = ClaudeCodeBackend(
        OmniConfig(backend="claude_code", run_dir=str(tmp_path / "run"), config={"ralph": False})
    )

    with patch("gepa.omni.backends.claude_code.subprocess.run", side_effect=fake_run):
        result = backend.run(task, server)

    assert result.best_candidate == "candidate"
