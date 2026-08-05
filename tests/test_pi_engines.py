from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from gepa.oa.agent_runner import AgentRunResult
from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engines.autoresearch import AutoResearchEngine
from gepa.oa.engines.meta_harness import _run_codex_proposer, _run_pi_proposer, _run_proposer
from gepa.oa.task import Task


class _FakePiRunner:
    instances: ClassVar[list[_FakePiRunner]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.closed = False
        self.session_id = f"session-{len(self.instances)}"
        self.instances.append(self)

    def run(self, prompt: str, *, work_dir: Path, **_kwargs: object) -> AgentRunResult:
        self.calls.append(prompt)
        if self.kwargs["persistent"]:
            (work_dir / "best_candidate.txt").write_text("pi candidate")
        event = {
            "type": "agent_end",
            "usage": {"input_tokens": 3, "output_tokens": 4},
            "cost_usd": 0.2 if len(self.calls) == 1 else 0.0005,
        }
        return AgentRunResult(
            command=("pi", "--mode", "rpc" if self.kwargs["persistent"] else "json"),
            returncode=0,
            stdout=json.dumps(event) + "\n",
            session_id=self.session_id,
            usage=event["usage"],
            cost_usd=event["cost_usd"],
            completed=True,
        )

    def close(self) -> None:
        self.closed = True


class _FakeCodexRunner:
    instances: ClassVar[list[_FakeCodexRunner]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.closed = False
        self.session_id = f"codex-thread-{len(self.instances)}"
        self.instances.append(self)

    def run(self, prompt: str, *, work_dir: Path, **_kwargs: object) -> AgentRunResult:
        self.calls.append(prompt)
        if self.kwargs["persistent"]:
            (work_dir / "best_candidate.txt").write_text("codex candidate")
        input_tokens = 100_000 if len(self.calls) == 1 else 250
        event = {
            "type": "turn.completed",
            "thread_id": self.session_id,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        return AgentRunResult(
            command=("codex", "exec"),
            returncode=0,
            stdout=json.dumps(event) + "\n",
            session_id=self.session_id,
            usage=event["usage"],
            cost_usd=0.2 if len(self.calls) == 1 else 0.0005,
            cost_known=True,
            completed=True,
        )

    def close(self) -> None:
        self.closed = True


class _UnknownCostCodexRunner:
    instances: ClassVar[list[_UnknownCostCodexRunner]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.closed = False
        self.session_id = f"unknown-cost-thread-{len(self.instances)}"
        self.instances.append(self)

    def run(self, prompt: str, *, work_dir: Path, **_kwargs: object) -> AgentRunResult:
        self.calls.append(prompt)
        if self.kwargs["persistent"]:
            (work_dir / "best_candidate.txt").write_text("codex candidate")
        event = {
            "type": "turn.completed",
            "thread_id": self.session_id,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return AgentRunResult(
            command=("codex", "exec"),
            returncode=0 if len(self.calls) == 1 else 1,
            stdout=json.dumps(event) + "\n",
            stderr="" if len(self.calls) == 1 else "simulated resume failure",
            session_id=self.session_id,
            usage=event["usage"],
            cost_usd=None,
            cost_known=False,
            completed=True,
        )

    def close(self) -> None:
        self.closed = True


class _FailingCodexRunner:
    instances: ClassVar[list[_FailingCodexRunner]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def run(self, _prompt: str, *, work_dir: Path, **_kwargs: object) -> AgentRunResult:
        diagnostics = work_dir / ".codex-runner"
        diagnostics.mkdir(parents=True, exist_ok=True)
        (diagnostics / "stderr.log").write_text("retained Codex diagnostics")
        return AgentRunResult(
            command=("codex", "exec"),
            returncode=1,
            stdout="{malformed jsonl\n",
            stderr="CODEX_TIMEOUT: simulated timeout",
            session_id=None,
            usage={},
            cost_usd=None,
            cost_known=False,
            timed_out=True,
            completed=False,
        )

    def close(self) -> None:
        self.closed = True


def test_autoresearch_pi_keeps_one_runner_for_ralph(monkeypatch, tmp_path: Path) -> None:
    _FakePiRunner.instances = []
    monkeypatch.setattr("gepa.oa.engines.autoresearch.PiAgentRunner", _FakePiRunner)
    monkeypatch.setattr("gepa.oa.engines.autoresearch.preflight_agent_engine", lambda *a, **k: None)

    class Server:
        budget = BudgetTracker(max_evals=10)
        url = "http://127.0.0.1:9"
        best_score = 0.0
        eval_log: ClassVar[list[object]] = []

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=False,
            run_dir=str(tmp_path),
            max_token_cost=1.0,
            engine_config={
                "agent_backend": "pi",
                "model": "provider/model",
                "ralph": True,
            },
        )
    )
    result = engine.run(Task(name="smoke", seed_candidate="seed"), Server())

    assert len(_FakePiRunner.instances) == 1
    runner = _FakePiRunner.instances[0]
    assert runner.kwargs["persistent"] is True
    assert len(runner.calls) == 2
    assert runner.calls[1].startswith("Continue iterating")
    assert result.metadata["agent_backend"] == "pi"
    assert result.metadata["ralph_iterations"] == 2
    assert runner.closed


def test_meta_harness_pi_starts_a_fresh_runner_per_iteration(tmp_path: Path, monkeypatch) -> None:
    _FakePiRunner.instances = []
    monkeypatch.setattr("gepa.oa.engines.meta_harness.PiAgentRunner", _FakePiRunner)
    monkeypatch.setattr("gepa.oa.engines.meta_harness.pi_sandbox_prefix", lambda path: [])
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    log_dir = tmp_path / "logs"

    first = _run_pi_proposer(
        work_dir=work_dir,
        iteration=1,
        model="provider/model",
        max_candidates=2,
        max_budget_usd=1.0,
        pending_path=work_dir / "state" / "pending_eval_iter1.json",
        log_dir=log_dir,
        sandbox=False,
        pi_command="pi",
    )
    second = _run_pi_proposer(
        work_dir=work_dir,
        iteration=2,
        model="provider/model",
        max_candidates=2,
        max_budget_usd=1.0,
        pending_path=work_dir / "state" / "pending_eval_iter2.json",
        log_dir=log_dir,
        sandbox=False,
        pi_command="pi",
    )

    assert first[0] == second[0] == 0
    assert len(_FakePiRunner.instances) == 2
    assert all(instance.kwargs["persistent"] is False for instance in _FakePiRunner.instances)
    assert _FakePiRunner.instances[0].session_id != _FakePiRunner.instances[1].session_id
    assert all(instance.closed for instance in _FakePiRunner.instances)
    assert "claude" not in (log_dir / "iter1_meta.json").read_text().lower()


def test_autoresearch_codex_reuses_one_thread_for_ralph(monkeypatch, tmp_path: Path) -> None:
    _FakeCodexRunner.instances = []
    monkeypatch.setattr("gepa.oa.engines.autoresearch.CodexAgentRunner", _FakeCodexRunner)
    monkeypatch.setattr("gepa.oa.engines.autoresearch.preflight_agent_engine", lambda *a, **k: None)

    class Server:
        budget = BudgetTracker(max_evals=10)
        url = "http://127.0.0.1:9"
        best_score = 0.0
        eval_log: ClassVar[list[object]] = []

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=False,
            run_dir=str(tmp_path),
            max_token_cost=1.0,
            engine_config={
                "agent_backend": "codex",
                "ralph": True,
                "codex_input_cost_per_million": 2.0,
                "codex_output_cost_per_million": 2.0,
            },
        )
    )
    result = engine.run(Task(name="smoke", seed_candidate="seed"), Server())

    assert len(_FakeCodexRunner.instances) == 1
    runner = _FakeCodexRunner.instances[0]
    assert runner.kwargs["persistent"] is True
    assert len(runner.calls) == 2
    assert runner.calls[1].startswith("Continue iterating")
    assert result.metadata["agent_backend"] == "codex"
    assert result.metadata["runner_session_id"] == runner.session_id
    assert result.metadata["adapter_cost_known"]
    assert runner.closed


def test_meta_harness_codex_starts_a_fresh_ephemeral_runner_per_iteration(
    tmp_path: Path, monkeypatch
) -> None:
    _FakeCodexRunner.instances = []
    monkeypatch.setattr("gepa.oa.engines.meta_harness.CodexAgentRunner", _FakeCodexRunner)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    log_dir = tmp_path / "logs"

    first = _run_codex_proposer(
        work_dir=work_dir,
        iteration=1,
        model=None,
        max_candidates=2,
        max_budget_usd=1.0,
        pending_path=work_dir / "state" / "pending_eval_iter1.json",
        log_dir=log_dir,
        sandbox=True,
        codex_command="codex",
        input_cost_per_million=2.0,
        output_cost_per_million=2.0,
    )
    second = _run_codex_proposer(
        work_dir=work_dir,
        iteration=2,
        model=None,
        max_candidates=2,
        max_budget_usd=1.0,
        pending_path=work_dir / "state" / "pending_eval_iter2.json",
        log_dir=log_dir,
        sandbox=True,
        codex_command="codex",
        input_cost_per_million=2.0,
        output_cost_per_million=2.0,
    )

    assert first[0] == second[0] == 0
    assert first[3] and second[3]
    assert len(_FakeCodexRunner.instances) == 2
    assert all(instance.kwargs["persistent"] is False for instance in _FakeCodexRunner.instances)
    assert _FakeCodexRunner.instances[0].session_id != _FakeCodexRunner.instances[1].session_id
    assert all(instance.closed for instance in _FakeCodexRunner.instances)
    metadata = json.loads((log_dir / "iter1_meta.json").read_text())
    assert metadata["agent_backend"] == "codex"
    assert metadata["cmd"] == ["codex", "exec"]
    assert metadata["cost_known"] is True


def test_codex_engine_requires_complete_pricing_when_capped() -> None:
    with pytest.raises(ValueError, match="requires both"):
        AutoResearchEngine(
            OptimizeAnythingConfig(
                engine="autoresearch",
                max_token_cost=1.0,
                sandbox=False,
                engine_config={"agent_backend": "codex"},
            )
        )


def test_autoresearch_uncapped_codex_preserves_unknown_cost_and_continues_ralph(
    monkeypatch, tmp_path: Path
) -> None:
    _UnknownCostCodexRunner.instances = []
    monkeypatch.setattr("gepa.oa.engines.autoresearch.CodexAgentRunner", _UnknownCostCodexRunner)
    monkeypatch.setattr("gepa.oa.engines.autoresearch.preflight_agent_engine", lambda *a, **k: None)

    class Server:
        budget = BudgetTracker(max_evals=10)
        url = "http://127.0.0.1:9"
        best_score = 0.0
        eval_log: ClassVar[list[object]] = []

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=False,
            run_dir=str(tmp_path),
            max_token_cost=None,
            engine_config={"agent_backend": "codex", "ralph": True},
        )
    )
    result = engine.run(Task(name="smoke", seed_candidate="seed"), Server())

    runner = _UnknownCostCodexRunner.instances[0]
    assert len(runner.calls) == 2
    assert result.metadata["adapter_cost"] is None
    assert result.metadata["adapter_cost_known"] is False
    assert result.metadata["adapter_cost_estimate_usd"] is None
    assert result.metadata["invocations"][0]["cost"] is None
    assert result.metadata["ralph_iterations"] == 1


def test_autoresearch_persists_codex_failure_artifacts_before_raising(monkeypatch, tmp_path: Path) -> None:
    _FailingCodexRunner.instances = []
    monkeypatch.setattr("gepa.oa.engines.autoresearch.CodexAgentRunner", _FailingCodexRunner)
    monkeypatch.setattr("gepa.oa.engines.autoresearch.preflight_agent_engine", lambda *a, **k: None)
    output_dir = tmp_path / "output"

    class Server:
        budget = BudgetTracker(max_evals=10)
        url = "http://127.0.0.1:9"
        best_score = 0.0
        eval_log: ClassVar[list[object]] = []

    Server.output_dir = output_dir

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=True,
            run_dir=str(tmp_path / "run"),
            max_token_cost=None,
            engine_config={"agent_backend": "codex", "ralph": False},
        )
    )
    with pytest.raises(RuntimeError, match="diagnostics="):
        engine.run(Task(name="smoke", seed_candidate="seed"), Server())

    retained = output_dir / "work" / ".codex-runner" / "stderr.log"
    assert retained.read_text() == "retained Codex diagnostics"
    assert _FailingCodexRunner.instances[0].closed


def test_meta_harness_uncapped_codex_logs_unknown_cost(tmp_path: Path, monkeypatch) -> None:
    _UnknownCostCodexRunner.instances = []
    monkeypatch.setattr("gepa.oa.engines.meta_harness.CodexAgentRunner", _UnknownCostCodexRunner)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    log_dir = tmp_path / "logs"

    exit_code, cost, session_id, cost_known = _run_codex_proposer(
        work_dir=work_dir,
        iteration=1,
        model=None,
        max_candidates=2,
        max_budget_usd=None,
        pending_path=work_dir / "state" / "pending_eval_iter1.json",
        log_dir=log_dir,
        sandbox=True,
        codex_command="codex",
        input_cost_per_million=None,
        output_cost_per_million=None,
    )

    assert exit_code == 0
    assert cost is None
    assert session_id == _UnknownCostCodexRunner.instances[0].session_id
    assert cost_known is False
    metadata = json.loads((log_dir / "iter1_meta.json").read_text())
    assert metadata["cost_usd"] is None
    assert metadata["cost_estimate_usd"] is None
    assert metadata["cost_known"] is False


def test_claude_command_omits_model_when_unset(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        "gepa.oa.engines.autoresearch._start_claude_process",
        lambda command, *_args, **_kwargs: captured.append(command)
        or SimpleNamespace(proc=SimpleNamespace(poll=lambda: 0, returncode=0)),
    )
    monkeypatch.setattr(
        "gepa.oa.engines.autoresearch._collect_claude_output",
        lambda _running: ('{"total_cost_usd": 0.1}', ""),
    )
    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=False,
            engine_config={"model": None, "ralph": False},
        )
    )

    engine._run_claude(
        work_dir=tmp_path,
        session_id="session",
        prompt="prompt",
        budget=BudgetTracker(max_evals=1),
        adapter_cost=0.0,
        resume=False,
        env={},
    )

    assert captured
    assert "--model" not in captured[0]


def test_meta_harness_claude_command_omits_model_when_unset(tmp_path: Path, monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"total_cost_usd": 0.1}), stderr="")

    monkeypatch.setattr("gepa.oa.engines.meta_harness.subprocess.run", fake_run)
    _run_proposer(
        work_dir=tmp_path,
        iteration=1,
        model=None,
        effort=None,
        max_candidates=1,
        max_budget_usd=None,
        pending_path=tmp_path / "state" / "pending.json",
        log_dir=tmp_path / "logs",
        sandbox=False,
    )

    assert captured
    assert "--model" not in captured[0]
