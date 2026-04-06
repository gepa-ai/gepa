# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.agents — CodingAgent, ClaudeCodeSession, and OpenCodeSession."""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from unittest.mock import MagicMock, patch

import pytest

from gepa.agents import CodingAgent
from gepa.agents.claude_code import ClaudeCodeSession
from gepa.agents.opencode import OpenCodeSession
from gepa.core.session import Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cc_result(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Build a fake subprocess.CompletedProcess with raw stdout string."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _claude_ok(
    result: str = "Hello world",
    session_id: str = "sess_abc123",
    cost: float = 0.01,
) -> str:
    """Typical successful Claude Code JSON output (single dict — legacy format)."""
    return json.dumps({"result": result, "session_id": session_id, "total_cost_usd": cost, "is_error": False})


def _claude_ok_list(
    result: str = "Hello world",
    session_id: str = "sess_abc123",
    cost: float = 0.01,
) -> str:
    """Claude Code JSON output as array (real CLI format)."""
    return json.dumps(
        [
            {"type": "system", "agents": ["general-purpose"]},
            {
                "type": "result",
                "result": result,
                "session_id": session_id,
                "total_cost_usd": cost,
                "is_error": False,
            },
        ]
    )


def _opencode_ndjson(
    text: str = "Hello world",
    session_id: str = "oc_abc123",
    cost: float = 0.005,
) -> str:
    """Typical successful OpenCode NDJSON output (multiple lines)."""
    events = [
        {"type": "step_start", "sessionID": session_id, "part": {"type": "step-start"}},
        {"type": "text", "sessionID": session_id, "part": {"type": "text", "text": text}},
        {
            "type": "step_finish",
            "sessionID": session_id,
            "part": {"type": "step-finish", "cost": cost, "tokens": {"total": 100}},
        },
    ]
    return "\n".join(json.dumps(e) for e in events)


def _opencode_error_ndjson(session_id: str = "oc_err", message: str = "Model not found") -> str:
    """OpenCode error event NDJSON."""
    return json.dumps(
        {
            "type": "error",
            "sessionID": session_id,
            "error": {"name": "UnknownError", "data": {"message": message}},
        }
    )


# ===========================================================================
# CodingAgent — Factory Tests
# ===========================================================================


class TestCodingAgent:
    """Test CodingAgent factory — dispatch, config, warnings."""

    def test_creates_claude_code_session(self):
        agent = CodingAgent(harness="claude_code", model="sonnet")
        session = agent.create_session()
        assert isinstance(session, ClaudeCodeSession)
        assert session._model == "sonnet"

    def test_creates_opencode_session(self):
        agent = CodingAgent(harness="opencode", model="sonnet")
        session = agent.create_session()
        assert isinstance(session, OpenCodeSession)

    def test_passes_system_prompt_to_claude_code(self):
        agent = CodingAgent(harness="claude_code", model="sonnet", system_prompt="Be helpful")
        session = agent.create_session()
        assert isinstance(session, ClaudeCodeSession)
        assert session._system_prompt == "Be helpful"

    def test_passes_system_prompt_to_opencode(self):
        agent = CodingAgent(harness="opencode", model="sonnet", system_prompt="Be helpful")
        session = agent.create_session()
        assert isinstance(session, OpenCodeSession)
        assert session._system_prompt == "Be helpful"

    def test_passes_timeout(self):
        agent = CodingAgent(harness="claude_code", model="sonnet", timeout=60)
        session = agent.create_session()
        assert session._timeout == 60

    def test_passes_max_budget_to_claude_code(self):
        agent = CodingAgent(harness="claude_code", model="sonnet", max_budget_usd=5.0)
        session = agent.create_session()
        assert isinstance(session, ClaudeCodeSession)
        assert session._max_budget_usd == 5.0

    def test_warns_max_budget_on_opencode(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CodingAgent(harness="opencode", model="sonnet", max_budget_usd=5.0)
            assert len(w) == 1
            assert "max_budget_usd" in str(w[0].message)
            assert "opencode" in str(w[0].message)

    def test_passes_extra_flags(self):
        agent = CodingAgent(harness="claude_code", model="sonnet", extra_flags=["--verbose"])
        session = agent.create_session()
        assert session._extra_flags == ["--verbose"]

    def test_unknown_harness_raises(self):
        with pytest.raises(ValueError, match="Unknown harness.*badharness"):
            CodingAgent(harness="badharness", model="sonnet")

    def test_harness_property(self):
        agent = CodingAgent(harness="claude_code", model="sonnet")
        assert agent.harness == "claude_code"

    def test_model_property(self):
        agent = CodingAgent(harness="claude_code", model="opus")
        assert agent.model == "opus"

    def test_create_session_returns_new_instance_each_time(self):
        agent = CodingAgent(harness="claude_code", model="sonnet")
        s1 = agent.create_session()
        s2 = agent.create_session()
        assert s1 is not s2

    def test_session_satisfies_protocol(self):
        for harness in ("claude_code", "opencode"):
            agent = CodingAgent(harness=harness, model="sonnet")
            session = agent.create_session()
            assert isinstance(session, Session)


class TestModelPassthrough:
    """Models are passed to the CLI as-is with no alias resolution."""

    def test_opencode_passes_full_model_string(self):
        session = OpenCodeSession(model="openrouter/anthropic/claude-sonnet-4.6")
        assert session._model == "openrouter/anthropic/claude-sonnet-4.6"

    def test_opencode_unknown_model_passthrough(self):
        session = OpenCodeSession(model="custom/my-model")
        assert session._model == "custom/my-model"

    def test_claude_code_passes_model_as_is(self):
        """Claude Code CLI handles its own aliases (sonnet/opus/haiku) natively."""
        session = ClaudeCodeSession(model="sonnet")
        assert session._model == "sonnet"


# ===========================================================================
# OpenCode --system flag
# ===========================================================================


class TestOpenCodeSystemPrompt:
    """Verify --system flag is passed correctly."""

    @patch("gepa.agents.opencode.subprocess.run")
    def test_system_prompt_in_cmd(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson())
        session = OpenCodeSession(model="sonnet", system_prompt="You are a code reviewer")

        session.send("Hello")

        cmd = mock_run.call_args[0][0]
        assert "--system" in cmd
        idx = cmd.index("--system")
        assert cmd[idx + 1] == "You are a code reviewer"

    @patch("gepa.agents.opencode.subprocess.run")
    def test_no_system_flag_when_empty(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson())
        session = OpenCodeSession(model="sonnet")

        session.send("Hello")

        cmd = mock_run.call_args[0][0]
        assert "--system" not in cmd

    @patch("gepa.agents.opencode.subprocess.run")
    def test_fork_preserves_system_prompt(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson(session_id="oc_forked"))
        session = OpenCodeSession(model="sonnet", system_prompt="sys", session_id="oc_1")

        forked = session.fork()
        assert forked._system_prompt == "sys"

    def test_reset_preserves_system_prompt(self):
        session = OpenCodeSession(model="sonnet", system_prompt="sys", session_id="oc_1")
        new = session.reset()
        assert new._system_prompt == "sys"


# ===========================================================================
# ClaudeCodeSession — Unit Tests
# ===========================================================================


class TestClaudeCodeSessionProtocol:
    """Verify ClaudeCodeSession satisfies the Session protocol."""

    def test_is_session_protocol(self):
        session = ClaudeCodeSession()
        assert isinstance(session, Session)

    def test_session_id_before_send(self):
        session = ClaudeCodeSession()
        assert session.session_id == "uninitialized"

    def test_history_returns_empty(self):
        session = ClaudeCodeSession()
        assert session.history == []

    def test_initial_cost_is_zero(self):
        session = ClaudeCodeSession()
        assert session.total_cost == 0.0
        assert session.last_send_cost is None


class TestClaudeCodeSessionSend:
    """Test send() — command construction, parsing, cost tracking."""

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_basic(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_claude_ok())
        session = ClaudeCodeSession(model="sonnet", system_prompt="Be helpful")

        result = session.send("Hello")

        assert result == "Hello world"
        assert session.session_id == "sess_abc123"

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--system-prompt" in cmd
        assert "Be helpful" in cmd
        assert "Hello" in cmd
        assert "--resume" not in cmd

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_handles_list_format(self, mock_run: MagicMock):
        """Real CLI returns a JSON array — last element is the result."""
        mock_run.return_value = _cc_result(stdout=_claude_ok_list(result="list response", session_id="sess_list"))
        session = ClaudeCodeSession()

        result = session.send("Hello")

        assert result == "list response"
        assert session.session_id == "sess_list"

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_resumes_session(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_claude_ok(session_id="sess_2"))
        session = ClaudeCodeSession(session_id="sess_1")

        session.send("Continue")

        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "sess_1"
        assert session.session_id == "sess_2"

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_tracks_cost(self, mock_run: MagicMock):
        mock_run.side_effect = [
            _cc_result(stdout=_claude_ok(cost=0.01)),
            _cc_result(stdout=_claude_ok(session_id="sess_2", cost=0.02)),
        ]
        session = ClaudeCodeSession()

        session.send("First")
        assert session.last_send_cost == 0.01
        assert session.total_cost == 0.01

        session.send("Second")
        assert session.last_send_cost == 0.02
        assert session.total_cost == pytest.approx(0.03)

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_nonzero_exit_raises(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(returncode=1, stderr="something broke")

        session = ClaudeCodeSession()
        with pytest.raises(RuntimeError, match="Claude Code failed.*something broke"):
            session.send("Hello")

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_is_error_raises(self, mock_run: MagicMock):
        error_json = json.dumps({"is_error": True, "result": "rate limited", "session_id": "x"})
        mock_run.return_value = _cc_result(stdout=error_json)

        session = ClaudeCodeSession()
        with pytest.raises(RuntimeError, match="Claude Code error.*rate limited"):
            session.send("Hello")

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_with_max_budget(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_claude_ok())
        session = ClaudeCodeSession(max_budget_usd=5.0)

        session.send("Hello")

        cmd = mock_run.call_args[0][0]
        assert "--max-budget-usd" in cmd
        idx = cmd.index("--max-budget-usd")
        assert cmd[idx + 1] == "5.0"

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_with_extra_flags(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_claude_ok())
        session = ClaudeCodeSession(extra_flags=["--verbose", "--no-cache"])

        session.send("Hello")

        cmd = mock_run.call_args[0][0]
        assert "--verbose" in cmd
        assert "--no-cache" in cmd

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_send_passes_timeout(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_claude_ok())
        session = ClaudeCodeSession(timeout=60)

        session.send("Hello")

        assert mock_run.call_args[1]["timeout"] == 60


class TestClaudeCodeSessionFork:
    """Test fork() — forks via CLI, returns new session."""

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_fork_calls_cli_with_fork_flag(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(
            stdout=json.dumps({"session_id": "sess_forked", "result": "", "total_cost_usd": 0.0, "is_error": False})
        )
        session = ClaudeCodeSession(model="opus", system_prompt="sys", session_id="sess_parent")

        forked = session.fork()

        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        assert "--fork-session" in cmd
        assert forked.session_id == "sess_forked"
        assert session.session_id == "sess_parent"

    def test_fork_without_session_raises(self):
        session = ClaudeCodeSession()
        with pytest.raises(RuntimeError, match="Cannot fork a session that hasn't been used"):
            session.fork()

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_fork_preserves_config(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(
            stdout=json.dumps({"session_id": "sess_forked", "result": "", "total_cost_usd": 0.0, "is_error": False})
        )
        session = ClaudeCodeSession(
            model="opus",
            system_prompt="sys prompt",
            session_id="sess_1",
            timeout=120,
            max_budget_usd=10.0,
            extra_flags=["--flag"],
        )

        forked = session.fork()

        assert forked._model == "opus"
        assert forked._system_prompt == "sys prompt"
        assert forked._timeout == 120
        assert forked._max_budget_usd == 10.0
        assert forked._extra_flags == ["--flag"]

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_fork_nonzero_exit_raises(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(returncode=1, stderr="fork failed")
        session = ClaudeCodeSession(session_id="sess_1")

        with pytest.raises(RuntimeError, match="Claude Code fork failed.*fork failed"):
            session.fork()


class TestClaudeCodeSessionReset:
    """Test reset() — returns new session with same config but no session_id."""

    def test_reset_returns_new_session(self):
        session = ClaudeCodeSession(
            model="opus",
            system_prompt="sys",
            session_id="sess_1",
            timeout=120,
            max_budget_usd=10.0,
            extra_flags=["--flag"],
        )

        new = session.reset()

        assert new is not session
        assert new._model == "opus"
        assert new._system_prompt == "sys"
        assert new._timeout == 120
        assert new._max_budget_usd == 10.0
        assert new._extra_flags == ["--flag"]
        assert new._session_id is None
        assert new.session_id == "uninitialized"

    def test_reset_leaves_original_untouched(self):
        session = ClaudeCodeSession(session_id="sess_1")
        session.reset()
        assert session.session_id == "sess_1"


# ===========================================================================
# OpenCodeSession — Unit Tests
# ===========================================================================


class TestOpenCodeSessionProtocol:
    """Verify OpenCodeSession satisfies the Session protocol."""

    def test_is_session_protocol(self):
        session = OpenCodeSession()
        assert isinstance(session, Session)

    def test_session_id_before_send(self):
        session = OpenCodeSession()
        assert session.session_id == "uninitialized"

    def test_history_returns_empty(self):
        session = OpenCodeSession()
        assert session.history == []

    def test_initial_cost_is_zero(self):
        session = OpenCodeSession()
        assert session.total_cost == 0.0
        assert session.last_send_cost is None


class TestOpenCodeSessionSend:
    """Test send() — command construction, parsing, NDJSON cost tracking."""

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_basic(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson())
        session = OpenCodeSession(model="openrouter/anthropic/claude-sonnet-4.6")

        result = session.send("Hello")

        assert result == "Hello world"
        assert session.session_id == "oc_abc123"

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "opencode"
        assert "run" in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "openrouter/anthropic/claude-sonnet-4.6"
        assert "--format" in cmd
        assert "json" in cmd
        assert "Hello" in cmd
        assert "--session" not in cmd

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_resumes_session(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson(session_id="oc_2"))
        session = OpenCodeSession(session_id="oc_1")

        session.send("Continue")

        cmd = mock_run.call_args[0][0]
        assert "--session" in cmd
        idx = cmd.index("--session")
        assert cmd[idx + 1] == "oc_1"
        assert session.session_id == "oc_2"

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_tracks_cost(self, mock_run: MagicMock):
        mock_run.side_effect = [
            _cc_result(stdout=_opencode_ndjson(cost=0.01)),
            _cc_result(stdout=_opencode_ndjson(session_id="oc_2", cost=0.02)),
        ]
        session = OpenCodeSession()

        session.send("First")
        assert session.last_send_cost == 0.01
        assert session.total_cost == 0.01

        session.send("Second")
        assert session.last_send_cost == 0.02
        assert session.total_cost == pytest.approx(0.03)

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_nonzero_exit_raises(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(returncode=1, stderr="something broke")

        session = OpenCodeSession()
        with pytest.raises(RuntimeError, match="OpenCode failed.*something broke"):
            session.send("Hello")

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_error_event_raises(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(returncode=1, stdout=_opencode_error_ndjson(message="Model not found"))

        session = OpenCodeSession()
        with pytest.raises(RuntimeError, match="OpenCode error.*Model not found"):
            session.send("Hello")

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_concatenates_text_parts(self, mock_run: MagicMock):
        """Multiple text events should be joined."""
        events = [
            {"type": "step_start", "sessionID": "oc_1", "part": {"type": "step-start"}},
            {"type": "text", "sessionID": "oc_1", "part": {"type": "text", "text": "Hello "}},
            {"type": "text", "sessionID": "oc_1", "part": {"type": "text", "text": "world"}},
            {"type": "step_finish", "sessionID": "oc_1", "part": {"type": "step-finish", "cost": 0.01}},
        ]
        mock_run.return_value = _cc_result(stdout="\n".join(json.dumps(e) for e in events))
        session = OpenCodeSession()

        result = session.send("Hello")
        assert result == "Hello world"

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_no_text_returns_empty(self, mock_run: MagicMock):
        """If no text events, return empty string."""
        events = [
            {"type": "step_start", "sessionID": "oc_1", "part": {"type": "step-start"}},
            {"type": "step_finish", "sessionID": "oc_1", "part": {"type": "step-finish", "cost": 0.0}},
        ]
        mock_run.return_value = _cc_result(stdout="\n".join(json.dumps(e) for e in events))
        session = OpenCodeSession()

        result = session.send("Hello")
        assert result == ""

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_missing_session_id_generates_fallback(self, mock_run: MagicMock):
        """If no sessionID in events, use UUID fallback."""
        events = [
            {"type": "step_start", "part": {"type": "step-start"}},
            {"type": "step_finish", "part": {"type": "step-finish", "cost": 0.0}},
        ]
        mock_run.return_value = _cc_result(stdout="\n".join(json.dumps(e) for e in events))
        session = OpenCodeSession()

        session.send("Hello")
        assert session.session_id != "uninitialized"
        assert len(session.session_id) == 12

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_with_extra_flags(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson())
        session = OpenCodeSession(extra_flags=["--thinking"])

        session.send("Hello")

        cmd = mock_run.call_args[0][0]
        assert "--thinking" in cmd

    @patch("gepa.agents.opencode.subprocess.run")
    def test_send_passes_timeout(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson())
        session = OpenCodeSession(timeout=60)

        session.send("Hello")

        assert mock_run.call_args[1]["timeout"] == 60


class TestOpenCodeSessionFork:
    """Test fork() — forks via CLI, returns new session."""

    @patch("gepa.agents.opencode.subprocess.run")
    def test_fork_calls_cli_with_fork_flag(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson(session_id="oc_forked"))
        session = OpenCodeSession(model="openrouter/anthropic/claude-sonnet-4.6", session_id="oc_parent")

        forked = session.fork()

        cmd = mock_run.call_args[0][0]
        assert "--session" in cmd
        assert "--fork" in cmd
        assert forked.session_id == "oc_forked"
        assert session.session_id == "oc_parent"

    def test_fork_without_session_raises(self):
        session = OpenCodeSession()
        with pytest.raises(RuntimeError, match="Cannot fork a session that hasn't been used"):
            session.fork()

    @patch("gepa.agents.opencode.subprocess.run")
    def test_fork_preserves_config(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson(session_id="oc_forked"))
        session = OpenCodeSession(
            model="openrouter/anthropic/claude-sonnet-4.6",
            system_prompt="sys",
            session_id="oc_1",
            timeout=120,
            extra_flags=["--flag"],
        )

        forked = session.fork()

        assert forked._model == "openrouter/anthropic/claude-sonnet-4.6"
        assert forked._system_prompt == "sys"
        assert forked._timeout == 120
        assert forked._extra_flags == ["--flag"]

    @patch("gepa.agents.opencode.subprocess.run")
    def test_fork_nonzero_exit_raises(self, mock_run: MagicMock):
        mock_run.return_value = _cc_result(returncode=1, stderr="fork failed")
        session = OpenCodeSession(session_id="oc_1")

        with pytest.raises(RuntimeError, match="OpenCode fork failed.*fork failed"):
            session.fork()


class TestOpenCodeSessionReset:
    """Test reset() — returns new session with same config but no session_id."""

    def test_reset_returns_new_session(self):
        session = OpenCodeSession(
            model="openrouter/anthropic/claude-sonnet-4.6",
            system_prompt="sys",
            session_id="oc_1",
            timeout=120,
            extra_flags=["--flag"],
        )

        new = session.reset()

        assert new is not session
        assert new._model == "openrouter/anthropic/claude-sonnet-4.6"
        assert new._system_prompt == "sys"
        assert new._timeout == 120
        assert new._extra_flags == ["--flag"]
        assert new._session_id is None
        assert new.session_id == "uninitialized"

    def test_reset_leaves_original_untouched(self):
        session = OpenCodeSession(session_id="oc_1")
        session.reset()
        assert session.session_id == "oc_1"


# ===========================================================================
# SessionManager integration — agents work with the pool
# ===========================================================================


class TestAgentsWithSessionManager:
    """Verify agents can be used with SessionManager."""

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_claude_code_in_session_manager(self, mock_run: MagicMock):
        from gepa.core.session import SessionManager

        mock_run.return_value = _cc_result(stdout=_claude_ok(session_id="sess_1"))

        def create():
            return ClaudeCodeSession(model="sonnet", system_prompt="test")

        manager = SessionManager(create=create)
        session = manager.select()

        assert isinstance(session, ClaudeCodeSession)

        result = session.send("Hello")
        assert result == "Hello world"
        assert session.session_id == "sess_1"

    @patch("gepa.agents.opencode.subprocess.run")
    def test_opencode_in_session_manager(self, mock_run: MagicMock):
        from gepa.core.session import SessionManager

        mock_run.return_value = _cc_result(stdout=_opencode_ndjson(session_id="oc_1"))

        def create():
            return OpenCodeSession(model="openrouter/anthropic/claude-sonnet-4.6")

        manager = SessionManager(create=create)
        session = manager.select()

        assert isinstance(session, OpenCodeSession)

        result = session.send("Hello")
        assert result == "Hello world"

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_make_session_lm_with_claude_code(self, mock_run: MagicMock):
        from gepa.core.session import make_session_lm

        mock_run.return_value = _cc_result(stdout=_claude_ok(result="Generated candidate"))

        session = ClaudeCodeSession(model="sonnet")
        lm = make_session_lm(session)

        result = lm("Write a better prompt")
        assert result == "Generated candidate"

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_coding_agent_as_session_factory(self, mock_run: MagicMock):
        """CodingAgent.create_session works as SessionManager factory."""
        from gepa.core.session import SessionManager

        mock_run.return_value = _cc_result(stdout=_claude_ok(session_id="sess_factory"))

        agent = CodingAgent(harness="claude_code", model="sonnet")
        manager = SessionManager(create=agent.create_session)

        session = manager.select()
        assert isinstance(session, ClaudeCodeSession)

        result = session.send("Hello")
        assert result == "Hello world"


# ===========================================================================
# Agents + Session Strategy — composable tests
# ===========================================================================


class TestAgentsWithSessionStrategy:
    """Test that agents compose with session selectors and actions."""

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_agent_with_fork_strategy(self, mock_run: MagicMock):
        """Agent session can be forked via SessionManager with AlwaysFork."""
        from gepa.core.session import SessionManager
        from gepa.strategies.session_strategy import AlwaysFork

        call_count = [0]

        def mock_subprocess(*args, **kwargs):
            call_count[0] += 1
            sid = f"sess_{call_count[0]}"
            return _cc_result(stdout=_claude_ok(result=f"response_{call_count[0]}", session_id=sid))

        mock_run.side_effect = mock_subprocess

        agent = CodingAgent(harness="claude_code", model="sonnet")
        manager = SessionManager(
            create=agent.create_session,
            strategy=AlwaysFork(),
        )

        s1 = manager.select()
        s1.send("First prompt")
        manager.observe(candidate_idx=0, accepted=True)

        s2 = manager.select()
        assert s2 is not s1
        assert isinstance(s2, ClaudeCodeSession)

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_agent_with_reset_strategy(self, mock_run: MagicMock):
        """Agent session can be reset via SessionManager with AlwaysReset."""
        from gepa.core.session import SessionManager
        from gepa.strategies.session_strategy import AlwaysReset

        mock_run.return_value = _cc_result(stdout=_claude_ok())

        agent = CodingAgent(harness="claude_code", model="sonnet", system_prompt="test sys")
        manager = SessionManager(
            create=agent.create_session,
            strategy=AlwaysReset(),
        )

        s1 = manager.select()  # fresh from factory (uninitialized)
        s1.send("Hello")  # CLI assigns real session_id
        manager.observe(candidate_idx=0, accepted=True)

        s2 = manager.select()  # resets → new session with session_id=None
        assert s2 is not s1
        assert isinstance(s2, ClaudeCodeSession)
        assert s2._session_id is None
        assert s2._system_prompt == "test sys"

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_agent_with_string_strategy(self, mock_run: MagicMock):
        """Agent works with string-based session strategy."""
        from gepa.core.session import SessionManager, resolve_session_strategy

        mock_run.return_value = _cc_result(stdout=_claude_ok())

        strategy = resolve_session_strategy("fork")
        agent = CodingAgent(harness="claude_code", model="sonnet")
        manager = SessionManager(create=agent.create_session, strategy=strategy)

        s1 = manager.select()
        s1.send("Hello")
        manager.observe(candidate_idx=0, accepted=True)

        s2 = manager.select()
        assert s2 is not s1


# ===========================================================================
# Mock End-to-End: agent string → optimize() wiring
# ===========================================================================


class TestAgentAutoWire:
    """Test that passing a CodingAgent to optimize() auto-wires sessions.

    These tests simulate the wiring in ``api.py`` / ``optimize_anything.py``
    without running a full optimization loop.
    """

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_api_auto_wires_coding_agent(self, mock_run: MagicMock):
        """api.py: isinstance(reflection_lm, CodingAgent) → make_session_lm(agent.create_session())."""
        from gepa.core.session import make_session_lm

        mock_run.return_value = _cc_result(stdout=_claude_ok(result="wired!"))

        # This mirrors the api.py branch for CodingAgent
        reflection_lm = CodingAgent(harness="claude_code", model="sonnet")
        assert isinstance(reflection_lm, CodingAgent)

        _agent_session = reflection_lm.create_session()
        reflection_lm_callable = make_session_lm(_agent_session)

        result = reflection_lm_callable("Improve the prompt")
        assert result == "wired!"

    @patch("gepa.agents.opencode.subprocess.run")
    def test_api_auto_wires_opencode_agent(self, mock_run: MagicMock):
        """Same wiring path works for OpenCode agents."""
        from gepa.core.session import make_session_lm

        mock_run.return_value = _cc_result(stdout=_opencode_ndjson(text="wired opencode"))

        reflection_lm = CodingAgent(
            harness="opencode",
            model="openrouter/anthropic/claude-sonnet-4.6",
        )
        _agent_session = reflection_lm.create_session()
        reflection_lm_callable = make_session_lm(_agent_session)

        result = reflection_lm_callable("Improve the prompt")
        assert result == "wired opencode"


class TestAgentWiring:
    """Test that CodingAgent instances auto-wire into a working reflection LM."""

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_coding_agent_creates_working_session(self, mock_run: MagicMock):
        """CodingAgent.create_session() + make_session_lm → callable LM."""
        from gepa.core.session import make_session_lm

        mock_run.return_value = _cc_result(stdout=_claude_ok(result="improved prompt v2"))

        agent = CodingAgent(harness="claude_code", model="sonnet")
        session = agent.create_session()
        lm = make_session_lm(session)

        result = lm("Improve this prompt: v1")
        assert result == "improved prompt v2"

    @patch("gepa.agents.opencode.subprocess.run")
    def test_opencode_agent_with_full_model_string(self, mock_run: MagicMock):
        """OpenCode CodingAgent uses the full provider/model string as-is."""
        mock_run.return_value = _cc_result(stdout=_opencode_ndjson(text="improved"))

        agent = CodingAgent(harness="opencode", model="openrouter/anthropic/claude-sonnet-4.6")
        session = agent.create_session()

        assert isinstance(session, OpenCodeSession)
        assert session._model == "openrouter/anthropic/claude-sonnet-4.6"

        result = session.send("Improve this")
        assert result == "improved"

    @patch("gepa.agents.claude_code.subprocess.run")
    def test_agent_with_session_strategy_wiring(self, mock_run: MagicMock):
        """Full wiring: agent string + session strategy → SessionManager → LM callable."""
        from gepa.core.session import SessionManager, make_session_lm, resolve_session_strategy

        call_count = [0]

        def mock_subprocess(*args, **kwargs):
            call_count[0] += 1
            return _cc_result(
                stdout=_claude_ok(result=f"candidate_v{call_count[0]}", session_id=f"sess_{call_count[0]}")
            )

        mock_run.side_effect = mock_subprocess

        # 1. Build a CodingAgent (what the user does)
        agent = CodingAgent(harness="claude_code", model="sonnet")

        # 2. Build session manager with strategy (what api.py does)
        strategy = resolve_session_strategy("fork")
        manager = SessionManager(create=agent.create_session, strategy=strategy)

        # 3. Create LM from manager (what api.py does)
        lm = make_session_lm(manager.current_session, fallback=None)

        # 4. Simulate optimization loop: LM is called multiple times
        r1 = lm("Improve prompt v0")
        assert r1 == "candidate_v1"

        # Select next session (fork calls subprocess once, then send calls it again)
        manager.observe(candidate_idx=0, accepted=True)
        next_session = manager.select()  # fork → subprocess call (v2)
        r2 = next_session.send("Improve prompt v1")  # send → subprocess call (v3)
        assert r2 == "candidate_v3"


# ===========================================================================
# Integration Tests — real CLI calls (skipped if CLI not available)
# ===========================================================================

_has_claude = shutil.which("claude") is not None
_has_opencode = shutil.which("opencode") is not None


@pytest.mark.skipif(not _has_claude, reason="claude CLI not installed")
class TestClaudeCodeIntegration:
    """Integration tests — actually calls the claude CLI."""

    def test_send_and_get_response(self):
        session = ClaudeCodeSession(model="sonnet", system_prompt="Respond with exactly: PONG", timeout=60)
        result = session.send("PING")

        assert isinstance(result, str)
        assert len(result) > 0
        assert session.session_id != "uninitialized"
        assert session.total_cost >= 0

    def test_send_then_fork(self):
        session = ClaudeCodeSession(model="sonnet", system_prompt="Respond briefly.", timeout=180)
        session.send("Say hello")

        forked = session.fork()
        assert forked.session_id != session.session_id
        assert forked.session_id != "uninitialized"

    def test_reset_creates_fresh(self):
        session = ClaudeCodeSession(model="sonnet", system_prompt="Test", session_id="sess_existing")
        new = session.reset()

        assert new.session_id == "uninitialized"
        assert new._session_id is None

    def test_coding_agent_integration(self):
        """CodingAgent → ClaudeCodeSession → real CLI."""
        agent = CodingAgent(harness="claude_code", model="sonnet", system_prompt="Respond with exactly: PONG")
        session = agent.create_session()
        result = session.send("PING")

        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.skipif(not _has_opencode, reason="opencode CLI not installed")
class TestOpenCodeIntegration:
    """Integration tests — actually calls the opencode CLI."""

    _OPENCODE_MODEL = "openrouter/anthropic/claude-sonnet-4.6"

    def test_send_and_get_response(self):
        session = OpenCodeSession(model=self._OPENCODE_MODEL, timeout=60)
        result = session.send("Say exactly: PONG")

        assert isinstance(result, str)
        assert len(result) > 0
        assert session.session_id != "uninitialized"

    def test_reset_creates_fresh(self):
        session = OpenCodeSession(session_id="oc_existing")
        new = session.reset()

        assert new.session_id == "uninitialized"
        assert new._session_id is None

    def test_coding_agent_integration(self):
        """CodingAgent → OpenCodeSession → real CLI."""
        agent = CodingAgent(harness="opencode", model=self._OPENCODE_MODEL)
        session = agent.create_session()
        result = session.send("Say exactly: PONG")

        assert isinstance(result, str)
        assert len(result) > 0
