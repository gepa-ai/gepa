import json
from pathlib import Path
from unittest.mock import patch

from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engines.autoresearch import AutoResearchEngine
from gepa.oa.task import Task


class _FakeServer:
    def __init__(self) -> None:
        self.budget = BudgetTracker(max_evals=10, max_token_cost=1.0)
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


def test_autoresearch_engine_ralph_resumes_with_remaining_budget(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        cost = 0.2 if len(calls) == 1 else 0.0005
        return _FakePopen(0, json.dumps({"total_cost_usd": cost}))

    engine = AutoResearchEngine(OptimizeAnythingConfig(engine="autoresearch", run_dir=str(tmp_path), engine_config={}))

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
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", run_dir=str(tmp_path), engine_config={"ralph": False})
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 1
    assert "--session-id" in calls[0]
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_engine_string_false_disables_ralph(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", run_dir=str(tmp_path), engine_config={"ralph": "false"})
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 1
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_engine_ralph_respects_stop_at_score(tmp_path: Path) -> None:
    server = _FakeServer()
    server.best_score = 1.0
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(engine="autoresearch", run_dir=str(tmp_path), stop_at_score=1.0, engine_config={})
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert len(calls) == 1
    assert result.metadata["adapter_cost"] == 0.2
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_engine_counts_failed_resume_cost(tmp_path: Path) -> None:
    server = _FakeServer()
    task = Task(name="smoke", initial_candidate="seed")
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        calls.append(cmd)
        Path(str(kwargs["cwd"]), "best_candidate.txt").write_text("candidate")
        if len(calls) == 1:
            return _FakePopen(0, json.dumps({"total_cost_usd": 0.2}))
        return _FakePopen(1, json.dumps({"total_cost_usd": 0.1}), stderr="failed")

    engine = AutoResearchEngine(OptimizeAnythingConfig(engine="autoresearch", run_dir=str(tmp_path), engine_config={}))

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
    task = Task(
        name="smoke",
        initial_candidate="seed",
        metadata={
            "optimize_anything_handoffs": [
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
        },
    )

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
        OptimizeAnythingConfig(engine="autoresearch", run_dir=str(tmp_path / "run"), engine_config={"ralph": False})
    )

    with patch("gepa.oa.engines.autoresearch.subprocess.Popen", side_effect=fake_popen):
        result = engine.run(task, server)

    assert result.best_candidate == "candidate"
