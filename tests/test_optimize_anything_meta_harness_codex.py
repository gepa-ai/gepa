"""Unit tests for meta_harness Codex CLI proposer support (mocked subprocess)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engines.meta_harness import (
    MetaHarnessConfig,
    MetaHarnessEngine,
    _build_claude_proposer_cmd,
    _build_codex_proposer_cmd,
    _parse_codex_proposer_result,
    _proposer_prompt,
)


def test_config_default_proposer_is_claude() -> None:
    cfg = MetaHarnessConfig()
    assert cfg.proposer == "claude"
    assert cfg.model == "claude-sonnet-4-6"


def test_config_codex_rewrites_claude_default_model() -> None:
    cfg = MetaHarnessConfig(proposer="codex")
    assert cfg.proposer == "codex"
    assert cfg.model == "gpt-5.6-terra"


def test_config_codex_keeps_explicit_model() -> None:
    cfg = MetaHarnessConfig(proposer="codex", model="o4-mini")
    assert cfg.model == "o4-mini"


def test_config_claude_code_alias() -> None:
    cfg = MetaHarnessConfig(proposer="claude-code")
    assert cfg.proposer == "claude"


def test_config_invalid_proposer() -> None:
    with pytest.raises(ValueError, match="claude.*codex"):
        MetaHarnessConfig(proposer="goose")


def test_config_codex_warns_on_claude_knobs() -> None:
    with pytest.warns(UserWarning, match="ignores"):
        MetaHarnessConfig(proposer="codex", effort="high", max_thinking_tokens=8000)


def test_engine_wires_proposer() -> None:
    engine = MetaHarnessEngine(
        OptimizeAnythingConfig(engine="meta_harness", engine_config={"proposer": "codex"}, sandbox=False)
    )
    assert engine.proposer == "codex"
    assert engine.model == "gpt-5.6-terra"


def test_build_codex_cmd_mac_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("gepa.oa.engines.meta_harness._is_macos", lambda: True)
    # Real bwrap_prefix returns [] on macOS; keep that contract.
    monkeypatch.setattr("gepa.oa.engines.meta_harness.bwrap_prefix", lambda *a, **k: [])
    cmd = _build_codex_proposer_cmd(work_dir=tmp_path, prompt="do the thing", model="gpt-5.6-terra", sandbox=True)
    assert cmd[0] == "codex"
    assert "bwrap" not in cmd
    assert cmd[:4] == ["codex", "exec", "--json", "--skip-git-repo-check"]
    assert "-C" in cmd and str(tmp_path.resolve()) in cmd
    assert "-m" in cmd and "gpt-5.6-terra" in cmd
    assert "--sandbox" in cmd and "workspace-write" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert cmd[-1] == "do the thing"


def test_build_codex_cmd_linux_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("gepa.oa.engines.meta_harness._is_macos", lambda: False)
    captured: dict[str, object] = {}

    def fake_bwrap(work_dir, **kwargs):
        captured["work_dir"] = work_dir
        captured["kwargs"] = kwargs
        return ["bwrap", "--bind", str(work_dir)]

    monkeypatch.setattr("gepa.oa.engines.meta_harness.bwrap_prefix", fake_bwrap)
    cmd = _build_codex_proposer_cmd(work_dir=tmp_path, prompt="propose", model="o4-mini", sandbox=True)
    assert cmd[:2] == ["bwrap", "--bind"]
    assert "codex" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "workspace-write" not in cmd
    assert captured["kwargs"] == {"agent": "codex"}


def test_build_codex_cmd_unsandboxed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("gepa.oa.engines.meta_harness._is_macos", lambda: True)
    cmd = _build_codex_proposer_cmd(work_dir=tmp_path, prompt="x", model="gpt-5.6-terra", sandbox=False)
    assert cmd[0] == "codex"
    assert "--sandbox" in cmd and "danger-full-access" in cmd


def test_build_claude_cmd_unchanged_shape(tmp_path: Path) -> None:
    cmd = _build_claude_proposer_cmd(
        work_dir=tmp_path,
        prompt="run iter",
        model="claude-sonnet-4-6",
        effort=None,
        max_budget_usd=1.5,
        session_id="sid-123",
        sandbox=False,
    )
    assert "claude" in cmd
    assert "--print" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--session-id" in cmd and "sid-123" in cmd
    assert "--max-budget-usd" in cmd and "1.5000" in cmd
    assert "codex" not in cmd


def test_proposer_prompt_points_at_agents_md_for_codex(tmp_path: Path) -> None:
    prompt = _proposer_prompt(
        work_dir=tmp_path,
        iteration=2,
        max_candidates=3,
        pending_path=tmp_path / "state" / "pending.json",
        proposer="codex",
    )
    assert "AGENTS.md" in prompt
    assert ".claude/skills" not in prompt


def test_proposer_prompt_points_at_skill_for_claude(tmp_path: Path) -> None:
    prompt = _proposer_prompt(
        work_dir=tmp_path,
        iteration=1,
        max_candidates=3,
        pending_path=tmp_path / "pending.json",
        proposer="claude",
    )
    assert ".claude/skills" in prompt


def test_parse_codex_jsonl_session_usage_cost() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-abc"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "cached_input_tokens": 0,
                        "output_tokens": 500_000,
                        "reasoning_output_tokens": 0,
                    },
                }
            ),
        ]
    )
    parsed = _parse_codex_proposer_result(stdout)
    assert parsed["thread_id"] == "thread-abc"
    assert parsed["usage"]["input_tokens"] == 1_000_000
    assert parsed["is_error"] is False


def test_parse_codex_jsonl_error_event() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t1"}),
            json.dumps({"type": "turn.failed", "error": "boom"}),
        ]
    )
    parsed = _parse_codex_proposer_result(stdout)
    assert parsed["is_error"] is True
    assert parsed["error"] == "boom"


def test_materialize_writes_agents_md_for_codex(tmp_path: Path) -> None:
    from gepa.oa.budget import BudgetTracker
    from gepa.oa.engines.meta_harness import _materialize_sandbox
    from gepa.oa.task import Task

    class _FakeServer:
        def iter_split(self, split: str):
            return iter(())

    task = Task(name="t", seed_candidate="hello", train_set=None, val_set=None, test_set=None)
    _materialize_sandbox(tmp_path, task, _FakeServer(), BudgetTracker(max_evals=10), proposer="codex")
    assert (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / ".claude" / "skills" / "gepa-optimize-anything-meta-harness" / "SKILL.md").exists()
    assert (tmp_path / "agents" / "baseline.txt").read_text() == "hello"


def test_materialize_writes_skill_for_claude(tmp_path: Path) -> None:
    from gepa.oa.budget import BudgetTracker
    from gepa.oa.engines.meta_harness import _materialize_sandbox
    from gepa.oa.task import Task

    class _FakeServer:
        def iter_split(self, split: str):
            return iter(())

    task = Task(name="t", seed_candidate="hello", train_set=None, val_set=None, test_set=None)
    _materialize_sandbox(tmp_path, task, _FakeServer(), BudgetTracker(max_evals=10), proposer="claude")
    assert (tmp_path / ".claude" / "skills" / "gepa-optimize-anything-meta-harness" / "SKILL.md").exists()
    assert not (tmp_path / "AGENTS.md").exists()


def test_run_proposer_codex_invokes_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gepa.oa.engines import meta_harness as mh

    monkeypatch.setattr(mh, "_is_macos", lambda: True)

    captured: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps({"type": "thread.started", "thread_id": "tid-1"}) + "\n"
        stdout += json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 50}})
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(mh.subprocess, "run", fake_run)
    log_dir = tmp_path / "sessions"
    pending = tmp_path / "pending.json"
    code, cost, sid = mh._run_proposer(
        work_dir=tmp_path,
        iteration=1,
        model="gpt-5.6-terra",
        effort=None,
        max_candidates=2,
        max_budget_usd=None,
        pending_path=pending,
        log_dir=log_dir,
        sandbox=False,
        proposer="codex",
    )
    assert code == 0
    assert sid == "tid-1"
    assert cost >= 0.0
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "codex"
    assert (log_dir / "iter1_stdout.jsonl").exists()
    assert (log_dir / "iter1_meta.json").exists()


def test_preflight_codex_requires_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from gepa.oa import sandbox as sb

    monkeypatch.setattr(sb.shutil, "which", lambda name: None if name == "codex" else "/bin/true")
    with pytest.raises(RuntimeError, match="Codex CLI"):
        sb.require_codex_cli("meta_harness")


def test_engine_run_calls_codex_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gepa.oa.budget import BudgetTracker
    from gepa.oa.engines import meta_harness as mh
    from gepa.oa.task import Task

    called: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        mh,
        "preflight_codex_engine",
        lambda name, *, sandbox: called.append((name, sandbox)),
    )
    monkeypatch.setattr(mh, "preflight_claude_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("claude")))

    # Short-circuit the loop: budget already exhausted.
    class _Server:
        budget = BudgetTracker(max_evals=0)
        best_candidate = "seed"
        best_score = 0.0

        def __init__(self) -> None:
            self.eval_log: list = []

        def iter_split(self, split: str):
            return iter(())

    engine = MetaHarnessEngine(
        OptimizeAnythingConfig(
            engine="meta_harness",
            engine_config={"proposer": "codex", "max_iterations": 1},
            sandbox=False,
            run_dir=str(tmp_path),
            max_evals=0,
        )
    )
    task = Task(name="t", seed_candidate="seed", train_set=None, val_set=None, test_set=None)
    # BudgetTracker(max_evals=0) → exhausted immediately; run still materializes + preflights.
    with patch.object(engine, "_best_score", return_value=None):
        result = engine.run(task, _Server())  # type: ignore[arg-type]
    assert called == [("meta_harness", False)]
    assert result.metadata["meta_harness"]["proposer"] == "codex"


def test_bwrap_prefix_codex_omits_claude_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Codex-only Linux host must not require ~/.claude* binds."""
    from gepa.oa import sandbox as sb

    home = tmp_path / "home"
    home.mkdir()
    (home / ".codex").mkdir()
    # Deliberately no ~/.claude or ~/.claude.json
    monkeypatch.setattr(sb, "_IS_MACOS", False)
    monkeypatch.setattr(sb.Path, "home", classmethod(lambda cls: home))

    cmd = sb.bwrap_prefix(tmp_path / "work", agent="codex")
    joined = " ".join(cmd)
    assert str(home / ".codex") in joined
    assert str(home / ".cache") in joined
    assert ".claude" not in joined


def test_bwrap_prefix_claude_includes_claude_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gepa.oa import sandbox as sb

    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").mkdir()
    (home / ".claude.json").write_text("{}")
    (home / ".local").mkdir()
    monkeypatch.setattr(sb, "_IS_MACOS", False)
    monkeypatch.setattr(sb.Path, "home", classmethod(lambda cls: home))

    cmd = sb.bwrap_prefix(tmp_path / "work", agent="claude")
    joined = " ".join(cmd)
    assert str(home / ".claude") in joined
    assert str(home / ".claude.json") in joined
    assert str(home / ".local") in joined
    assert ".codex" not in joined


def test_engine_rejects_max_token_cost_for_codex() -> None:
    with pytest.raises(ValueError, match="max_token_cost"):
        MetaHarnessEngine(
            OptimizeAnythingConfig(
                engine="meta_harness",
                engine_config={"proposer": "codex"},
                max_token_cost=1.0,
                sandbox=False,
            )
        )


def test_engine_allows_codex_without_max_token_cost() -> None:
    engine = MetaHarnessEngine(
        OptimizeAnythingConfig(
            engine="meta_harness",
            engine_config={"proposer": "codex"},
            max_token_cost=None,
            sandbox=False,
        )
    )
    assert engine.proposer == "codex"
    assert engine.max_token_cost is None
