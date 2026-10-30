"""Tests for the internal coding-agent seam (gepa#459, minimal slice)."""

from __future__ import annotations

import os
import stat
import textwrap
from pathlib import Path

import pytest

from gepa.oa.agent_backend import (
    AgentRunRequest,
    ClaudeCodeBackend,
    CodexBackend,
    default_backend,
)

ARGV_LOGGER = '#!/bin/sh\nfor a in "$@"; do printf \'%s \' "$a" >> "$ARGV_LOG"; done\necho "" >> "$ARGV_LOG"\n'


def _install_fake_bin(tmp_path: Path, name: str, script: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / name
    exe.write_text(script.lstrip("\n"))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _env(bin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


def test_default_backend_is_claude() -> None:
    assert isinstance(default_backend(), ClaudeCodeBackend)


def test_claude_backend_parses_json_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bin_dir = _install_fake_bin(
        tmp_path,
        "claude",
        textwrap.dedent(
            """
        #!/bin/sh
        echo '{"result":"answer","session_id":"sess1","usage":{"input_tokens":5},"total_cost_usd":0.01}'
        """
        ),
    )
    _env(bin_dir, monkeypatch)
    res = ClaudeCodeBackend().run(AgentRunRequest(prompt="hi", workdir=tmp_path, resume_session_id="old-sess"))
    assert res.text == "answer"
    assert res.session_id == "sess1"
    assert res.cost == pytest.approx(0.01)
    assert res.usage == {"input_tokens": 5}


def test_claude_backend_resume_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "argv.log"
    script = ARGV_LOGGER + 'echo \'{"result":"ok"}\'\n'
    bin_dir = _install_fake_bin(tmp_path, "claude", script)
    _env(bin_dir, monkeypatch)
    monkeypatch.setenv("ARGV_LOG", str(log))
    res = ClaudeCodeBackend().run(AgentRunRequest(prompt="hi", workdir=tmp_path, resume_session_id="old-sess"))
    assert res.text == "ok"
    assert "--resume old-sess" in log.read_text()


def test_codex_backend_streams_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bin_dir = _install_fake_bin(
        tmp_path,
        "codex",
        textwrap.dedent(
            """
        #!/bin/sh
        echo '{"type":"session_configured","session_id":"ro_abc"}'
        echo '{"type":"item.started","item":{"id":"i1"}}'
        echo '{"type":"agent_message","message":"did the thing"}'
        """
        ),
    )
    _env(bin_dir, monkeypatch)
    res = CodexBackend().run(AgentRunRequest(prompt="p", workdir=tmp_path))
    assert res.text == "did the thing"
    assert res.session_id == "ro_abc"


def test_codex_backend_argv_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "argv.log"
    script = ARGV_LOGGER + 'echo \'{"type":"agent_message","message":"ok"}\'\n'
    bin_dir = _install_fake_bin(tmp_path, "codex", script)
    _env(bin_dir, monkeypatch)
    monkeypatch.setenv("ARGV_LOG", str(log))
    res = CodexBackend(model="gpt-5", sandbox_mode="read-only").run(AgentRunRequest(prompt="P", workdir=tmp_path))
    assert res.text == "ok"
    argv_text = log.read_text()
    assert "exec --json -m gpt-5 --sandbox read-only P" in argv_text


def test_codex_backend_full_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "argv.log"
    bin_dir = _install_fake_bin(tmp_path, "codex", ARGV_LOGGER)
    _env(bin_dir, monkeypatch)
    monkeypatch.setenv("ARGV_LOG", str(log))
    CodexBackend(full_auto=True).run(AgentRunRequest(prompt="P", workdir=tmp_path))
    argv_text = log.read_text()
    assert "--full-auto" in argv_text
    assert "--sandbox" not in argv_text
