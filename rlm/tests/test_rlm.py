"""Tests for RLM core functionality.

Tests are split into:
- Unit tests (mocked subprocess, fast)
- Integration tests (real claude -p calls, marked slow)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from unittest.mock import patch

import pytest

from rlm.core import RLMConfig, SpawnResult, _get_current_depth, _parse_stream_json, spawn, spawn_parallel


# --- Fixtures ---


@pytest.fixture
def config(tmp_path):
    budget_file = str(tmp_path / "budget.txt")
    return RLMConfig(
        max_depth=3,
        max_budget_usd=1.0,
        default_model="sonnet",
        timeout=30,
        budget_file=budget_file,
    )


def _make_stream_json(text: str, cost: float = 0.05, duration: int = 1000) -> str:
    """Build fake stream-json output."""
    init = json.dumps({"type": "system", "subtype": "init", "model": "claude-sonnet-4-6"})
    assistant = json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    })
    result = json.dumps({
        "type": "result",
        "result": text,
        "total_cost_usd": cost,
        "duration_ms": duration,
    })
    return f"{init}\n{assistant}\n{result}\n"


# --- Unit Tests ---


class TestParseStreamJson:
    def test_basic(self):
        text, cost, duration = _parse_stream_json(
            _make_stream_json("hello", cost=0.03, duration=500)
        )
        assert text == "hello"
        assert cost == 0.03
        assert duration == 500

    def test_empty(self):
        text, cost, duration = _parse_stream_json("")
        assert text == ""
        assert cost == 0.0

    def test_malformed_lines_skipped(self):
        raw = "not json\n" + _make_stream_json("ok")
        text, cost, _ = _parse_stream_json(raw)
        assert text == "ok"


class TestDepthControl:
    def test_depth_from_env(self):
        with patch.dict(os.environ, {"RLM_DEPTH": "2"}):
            assert _get_current_depth() == 2

    def test_depth_default_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _get_current_depth() == 0

    def test_max_depth_blocks_spawn(self, config):
        config.max_depth = 2
        with patch.dict(os.environ, {"RLM_DEPTH": "2"}):
            result = spawn("test", config=config)
            assert result.error is not None
            assert "Max depth" in result.error

    def test_depth_incremented_in_env(self, config):
        """Verify the subprocess receives RLM_DEPTH + 1."""
        with patch.dict(os.environ, {"RLM_DEPTH": "1"}):
            with patch("rlm.core.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0,
                    stdout=_make_stream_json("ok"),
                    stderr="",
                )
                spawn("test", config=config)
                call_env = mock_run.call_args.kwargs["env"]
                assert call_env["RLM_DEPTH"] == "2"


class TestBudgetControl:
    def test_budget_exhausted_blocks_spawn(self, config):
        # Write budget file showing we've spent the max
        with open(config.budget_file, "w") as f:
            f.write("1.5")
        result = spawn("test", config=config)
        assert result.error is not None
        assert "Budget exhausted" in result.error

    def test_budget_updated_after_spawn(self, config):
        with patch("rlm.core.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=_make_stream_json("ok", cost=0.05),
                stderr="",
            )
            spawn("test", config=config)
            spent = float(open(config.budget_file).read())
            assert spent == pytest.approx(0.05)

    def test_budget_accumulates(self, config):
        with patch("rlm.core.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=_make_stream_json("ok", cost=0.10),
                stderr="",
            )
            spawn("test", config=config)
            spawn("test", config=config)
            spent = float(open(config.budget_file).read())
            assert spent == pytest.approx(0.20)


class TestSpawn:
    def test_success(self, config):
        with patch("rlm.core.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=_make_stream_json("result text", cost=0.03, duration=800),
                stderr="",
            )
            result = spawn("test prompt", model="haiku", config=config)
            assert result.text == "result text"
            assert result.model == "haiku"
            assert result.cost_usd == 0.03
            assert result.duration_ms == 800
            assert result.error is None

    def test_nonzero_exit(self, config):
        with patch("rlm.core.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1,
                stdout="",
                stderr="something broke",
            )
            result = spawn("test", config=config)
            assert result.error is not None
            assert "something broke" in result.error

    def test_timeout(self, config):
        config.timeout = 1
        with patch("rlm.core.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=1)
            result = spawn("test", config=config)
            assert result.error is not None
            assert "Timeout" in result.error

    def test_cc_env_vars_stripped(self, config):
        with patch.dict(os.environ, {"CLAUDECODE": "1", "CLAUDE_CODE_SSE_PORT": "8080"}):
            with patch("rlm.core.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0,
                    stdout=_make_stream_json("ok"),
                    stderr="",
                )
                spawn("test", config=config)
                call_env = mock_run.call_args.kwargs["env"]
                assert "CLAUDECODE" not in call_env
                assert "CLAUDE_CODE_SSE_PORT" not in call_env

    def test_model_in_command(self, config):
        with patch("rlm.core.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=_make_stream_json("ok"),
                stderr="",
            )
            spawn("test", model="haiku", config=config)
            cmd = mock_run.call_args.args[0]
            assert "--model" in cmd
            idx = cmd.index("--model")
            assert cmd[idx + 1] == "haiku"


class TestSpawnParallel:
    def test_parallel_returns_ordered(self, config):
        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=_make_stream_json(f"result_{call_count}", cost=0.01),
                stderr="",
            )

        with patch("rlm.core.subprocess.run", side_effect=fake_run):
            results = spawn_parallel(
                [("prompt1", "haiku"), ("prompt2", "sonnet"), ("prompt3", None)],
                config=config,
            )
            assert len(results) == 3
            assert all(r.error is None for r in results)

    def test_parallel_budget_shared(self, config):
        config.max_budget_usd = 0.05  # tight budget

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=_make_stream_json("ok", cost=0.03),
                stderr="",
            )

        with patch("rlm.core.subprocess.run", side_effect=fake_run):
            results = spawn_parallel(
                [("p1", "haiku"), ("p2", "haiku"), ("p3", "haiku")],
                config=config,
                max_workers=1,  # sequential to make budget deterministic
            )
            # At least one should be budget-blocked
            errors = [r for r in results if r.error and "Budget" in r.error]
            successes = [r for r in results if r.error is None]
            assert len(successes) >= 1
            assert len(errors) >= 1


# --- Integration Tests (real claude -p) ---


@pytest.mark.slow
class TestIntegration:
    """These tests make real claude -p calls. Run with: pytest -m slow"""

    def test_spawn_real(self):
        config = RLMConfig(max_depth=2, timeout=30, default_model="sonnet")
        result = spawn("What is 2+2? Reply with just the number.", config=config)
        assert result.error is None
        assert "4" in result.text
        assert result.cost_usd > 0

    def test_spawn_parallel_real(self):
        config = RLMConfig(max_depth=2, timeout=30)
        results = spawn_parallel(
            [
                ("What is 2+2? Reply with just the number.", "sonnet"),
                ("What is 3+3? Reply with just the number.", "sonnet"),
            ],
            config=config,
        )
        assert len(results) == 2
        assert all(r.error is None for r in results)
        assert "4" in results[0].text
        assert "6" in results[1].text

    def test_depth_limit_real(self):
        config = RLMConfig(max_depth=1, timeout=30)
        with patch.dict(os.environ, {"RLM_DEPTH": "1"}):
            result = spawn("test", config=config)
            assert result.error is not None
            assert "Max depth" in result.error
