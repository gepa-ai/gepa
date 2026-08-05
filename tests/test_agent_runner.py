from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gepa.oa.agent_runner import (
    CodexAgentRunner,
    PiAgentRunner,
    normalize_codex_output,
    normalize_pi_output,
)


def _python_command(source: str) -> list[str]:
    return [sys.executable, "-u", "-c", source]


def test_normalize_pi_jsonl_usage_cost_and_completion() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "message_end", "message": {"content": '{"new_texts": {}}'}}),
            json.dumps({"type": "agent_end", "usage": {"input_tokens": 10, "output_tokens": 4}, "cost_usd": 0.25}),
        ]
    )
    usage, cost, text, completed = normalize_pi_output(raw)
    assert usage == {"input_tokens": 10.0, "output_tokens": 4.0}
    assert cost == 0.25
    assert '"new_texts"' in text
    assert completed


def test_normalize_pi_reads_usage_nested_in_message() -> None:
    raw = json.dumps(
        {
            "type": "message_end",
            "message": {
                "usage": {
                    "input": 7,
                    "output": 5,
                    "cost": {"total": 0.33},
                }
            },
        }
    )
    usage, cost, _text, _completed = normalize_pi_output(raw)
    assert usage == {"input_tokens": 7.0, "output_tokens": 5.0}
    assert cost == 0.33


def test_pi_json_runner_uses_read_only_flags_and_normalizes_output(tmp_path: Path) -> None:
    source = (
        "import json; "
        "print(json.dumps({'type':'message_end','message':{'content':'{\\\"answer\\\":\\\"ok\\\"}'}})); "
        "print(json.dumps({'type':'agent_end','usage':{'input_tokens':2,'output_tokens':3},'cost_usd':0.1}))"
    )
    runner = PiAgentRunner(command=_python_command(source), model="provider/model")
    result = runner.run("prompt", work_dir=tmp_path)
    assert result.returncode == 0
    assert result.completed
    assert result.usage["input_tokens"] == 2.0
    assert result.cost_usd == 0.1
    assert "--mode" in result.command
    assert "json" in result.command
    for flag in ("--no-session", "--no-context-files", "--no-extensions", "--no-skills", "--no-approve"):
        assert flag in result.command
    assert result.command[result.command.index("--tools") + 1] == "read,grep,find,ls,bash,edit,write"


def test_pi_rpc_runner_reuses_one_process_for_ralph_continuations(tmp_path: Path) -> None:
    source = (
        "import json,sys\n"
        "for line in sys.stdin:\n"
        "    if line.strip():\n"
        "        data=json.loads(line)\n"
        "        print(json.dumps({'type':'message_end','message':{'content':data['message']}}))\n"
        "        print(json.dumps({'type':'agent_end','usage':{'input_tokens':1,'output_tokens':1},'cost_usd':0.01}))\n"
        "        sys.stdout.flush()\n"
    )
    runner = PiAgentRunner(command=_python_command(source), persistent=True, model="provider/model")
    first = runner.run("first", work_dir=tmp_path)
    second = runner.run("second", work_dir=tmp_path)
    assert first.completed and second.completed
    assert first.session_id == second.session_id
    assert first.final_text == "first"
    assert second.final_text == "second"
    assert first.cost_usd == second.cost_usd == 0.01
    runner.close()


def test_pi_runner_terminates_timed_out_process_group(tmp_path: Path) -> None:
    source = "import time; time.sleep(30)"
    runner = PiAgentRunner(command=_python_command(source))
    result = runner.run("prompt", work_dir=tmp_path, timeout_seconds=0.05)
    assert result.timed_out
    assert "PI_TIMEOUT" in result.stderr
    assert result.returncode != 0


def _codex_jsonl_source() -> str:
    return (
        "import json,sys\n"
        "args=sys.argv[1:]\n"
        "resuming='resume' in args\n"
        "print(json.dumps({'type':'thread.started','thread_id':'thread-123'}))\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'codex answer'}}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100 if not resuming else 20,'output_tokens':50}}))\n"
    )


def test_normalize_codex_jsonl_usage_cost_and_completion() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}}),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 100, "cached_input_tokens": 12, "output_tokens": 50},
                }
            ),
        ]
    )
    usage, cost, text, session_id, completed, cost_known = normalize_codex_output(
        raw,
        input_cost_per_million=2.0,
        output_cost_per_million=4.0,
    )
    assert usage == {"input_tokens": 100, "cache_read_tokens": 12, "output_tokens": 50}
    assert cost == 0.0004
    assert text == "answer"
    assert session_id == "thread-123"
    assert completed
    assert cost_known


def test_normalize_codex_usage_only_marks_cost_unknown() -> None:
    raw = json.dumps(
        {
            "type": "turn.completed",
            "thread_id": "thread-123",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
    )
    usage, cost, _text, _session_id, completed, cost_known = normalize_codex_output(raw)
    assert usage == {"input_tokens": 100, "output_tokens": 50}
    assert cost is None
    assert completed
    assert not cost_known


def test_codex_runner_uses_workspace_write_and_resumes_thread(tmp_path: Path) -> None:
    runner = CodexAgentRunner(command=_python_command(_codex_jsonl_source()), persistent=True)
    first = runner.run("first", work_dir=tmp_path)
    second = runner.run("second", work_dir=tmp_path)

    assert first.returncode == second.returncode == 0
    assert first.completed and second.completed
    assert first.session_id == second.session_id == "thread-123"
    assert first.final_text == second.final_text == "codex answer"
    assert first.usage["input_tokens"] == 100
    assert second.usage["input_tokens"] == 20
    assert "exec" in first.command
    assert "--sandbox" in first.command
    assert first.command[first.command.index("--sandbox") + 1] == "workspace-write"
    assert "resume" not in first.command
    assert "resume" in second.command
    assert "--config" in first.command
    assert 'approval_policy="never"' in first.command
    assert "sandbox_workspace_write.network_access=true" in first.command
    assert second.command[second.command.index("resume") + 1] == "--config"


def test_codex_runner_one_shot_is_ephemeral(tmp_path: Path) -> None:
    runner = CodexAgentRunner(command=_python_command(_codex_jsonl_source()), persistent=False)
    result = runner.run("prompt", work_dir=tmp_path)
    assert result.returncode == 0
    assert "--ephemeral" in result.command
    assert "--sandbox" in result.command


def test_codex_runner_rejects_unsandboxed_mode_and_uncapped_missing_pricing() -> None:
    with pytest.raises(ValueError, match="workspace-write"):
        CodexAgentRunner(sandbox=False)
    runner = CodexAgentRunner(command=["does-not-exist"])
    with pytest.raises(ValueError, match="requires both"):
        runner.run("prompt", work_dir=Path.cwd(), max_budget_usd=1.0)
    with pytest.raises(ValueError, match="both input_cost_per_million"):
        CodexAgentRunner(command=["does-not-exist"], input_cost_per_million=1.0)


def test_codex_runner_rejects_malformed_output_and_missing_thread_id(tmp_path: Path) -> None:
    malformed = CodexAgentRunner(command=_python_command("print('{not-json')"))
    malformed_result = malformed.run("prompt", work_dir=tmp_path)
    assert malformed_result.returncode != 0
    assert "CODEX_MALFORMED_OUTPUT" in malformed_result.stderr

    no_thread_source = (
        "import json; print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))"
    )
    no_thread = CodexAgentRunner(command=_python_command(no_thread_source))
    no_thread_result = no_thread.run("prompt", work_dir=tmp_path)
    assert no_thread_result.returncode != 0
    assert "CODEX_MISSING_SESSION_ID" in no_thread_result.stderr


def test_codex_runner_detects_budget_overrun_after_process_exit(tmp_path: Path) -> None:
    source = (
        "import json; "
        "print(json.dumps({'type':'thread.started','thread_id':'thread-123'})); "
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1000,'output_tokens':1000}}))"
    )
    runner = CodexAgentRunner(
        command=_python_command(source),
        input_cost_per_million=1000.0,
        output_cost_per_million=1000.0,
    )
    result = runner.run("prompt", work_dir=tmp_path, max_budget_usd=0.001)
    assert result.returncode != 0
    assert "CODEX_TOKEN_BUDGET" in result.stderr


def test_codex_runner_terminates_timed_out_process_group(tmp_path: Path) -> None:
    runner = CodexAgentRunner(command=_python_command("import time; time.sleep(30)"))
    result = runner.run("prompt", work_dir=tmp_path, timeout_seconds=0.05)
    assert result.timed_out
    assert result.returncode != 0
    assert "CODEX_TIMEOUT" in result.stderr
