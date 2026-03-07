"""Tests for Claude Code as reflection LM and eval recording."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    _extract_text_from_stream_json,
    _parse_stream_json,
    make_claude_code_lm,
    optimize_anything,
)


def _make_stream_json(result_text: str) -> str:
    """Build fake stream-json JSONL matching real ``claude -p`` output format."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "test-session"}),
        json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": result_text}]},
            "session_id": "test-session",
        }),
        json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result_text,
            "session_id": "test-session",
        }),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_extract_text_from_stream_json():
    raw = _make_stream_json("Hello world")
    assert _extract_text_from_stream_json(raw) == "Hello world"


def test_parse_stream_json_returns_events():
    raw = _make_stream_json("test output")
    text, events = _parse_stream_json(raw)
    assert text == "test output"
    assert len(events) == 3
    assert events[0]["type"] == "system"
    assert events[1]["type"] == "assistant"
    assert events[2]["type"] == "result"


def test_extract_text_skips_blank_and_malformed_lines():
    raw = "\n\nbad json\n" + _make_stream_json("ok")
    assert _extract_text_from_stream_json(raw) == "ok"


def test_extract_text_empty_input():
    assert _extract_text_from_stream_json("") == ""


@patch("subprocess.run")
def test_make_claude_code_lm_string_prompt(mock_run: MagicMock):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=_make_stream_json("new prompt text"),
        stderr="",
    )
    lm = make_claude_code_lm()
    result = lm("improve this prompt")
    assert result == "new prompt text"
    call_kwargs = mock_run.call_args
    assert call_kwargs[0][0][:2] == ["claude", "-p"]
    assert call_kwargs[1]["input"] == "improve this prompt"


@patch("subprocess.run")
def test_make_claude_code_lm_messages_prompt(mock_run: MagicMock):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=_make_stream_json("response"),
        stderr="",
    )
    lm = make_claude_code_lm()
    result = lm([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])
    assert result == "response"
    prompt_input = mock_run.call_args[1]["input"]
    assert "user: hello" in prompt_input
    assert "assistant: hi" in prompt_input


@patch("subprocess.run")
def test_make_claude_code_lm_nonzero_exit(mock_run: MagicMock):
    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="something went wrong",
    )
    lm = make_claude_code_lm()
    try:
        lm("test")
        raise AssertionError("Should have raised")
    except RuntimeError as e:
        assert "exit 1" in str(e)
        assert "something went wrong" in str(e)


# ---------------------------------------------------------------------------
# End-to-end: optimize_anything with reflection_lm="claude_code"
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_e2e_optimize_anything_claude_code(mock_run: MagicMock):
    """Full optimize_anything loop using reflection_lm='claude_code'."""
    call_count = 0

    def fake_subprocess_run(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        candidate_text = f"```\noptimized prompt version {call_count}\n```"
        return MagicMock(
            returncode=0,
            stdout=_make_stream_json(candidate_text),
            stderr="",
        )

    mock_run.side_effect = fake_subprocess_run

    def evaluator(candidate: str) -> float:
        if "version" in candidate:
            try:
                v = int(candidate.split("version")[-1].strip())
                return v / 10.0
            except ValueError:
                pass
        return 0.1

    with tempfile.TemporaryDirectory() as tmpdir:
        result = optimize_anything(
            seed_candidate={"prompt": "initial prompt"},
            evaluator=evaluator,
            objective="Write a good prompt.",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=5, run_dir=tmpdir),
                reflection=ReflectionConfig(
                    reflection_lm="claude_code",
                    reflection_minibatch_size=1,
                ),
            ),
        )

    assert mock_run.call_count >= 1
    for call in mock_run.call_args_list:
        assert call[0][0][0] == "claude"
        assert call[0][0][1] == "-p"
    assert result.best_candidate is not None
    assert isinstance(result.best_candidate, dict)


# ---------------------------------------------------------------------------
# Eval recorder: writes evaluation results to disk
# ---------------------------------------------------------------------------


@patch("subprocess.run")
def test_eval_recorder_writes_files(mock_run: MagicMock):
    """EvalRecorderCallback writes skill.md, meta.json, and task files."""
    call_count = 0

    def fake_subprocess_run(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        candidate_text = f"```\ngreeting version {call_count}\n```"
        return MagicMock(
            returncode=0,
            stdout=_make_stream_json(candidate_text),
            stderr="",
        )

    mock_run.side_effect = fake_subprocess_run

    def evaluator(candidate: str) -> tuple[float, dict]:
        score = 0.5 if "version" in candidate else 0.1
        return score, {"feedback": f"scored {score}"}

    with tempfile.TemporaryDirectory() as tmpdir:
        optimize_anything(
            seed_candidate={"prompt": "hello"},
            evaluator=evaluator,
            objective="Write a greeting.",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=3, run_dir=tmpdir),
                reflection=ReflectionConfig(
                    reflection_lm="claude_code",
                    reflection_minibatch_size=1,
                ),
            ),
        )

        evals_dir = os.path.join(tmpdir, "evals")
        assert os.path.isdir(evals_dir)

        # Candidate dirs (c000, c001, ...) should exist
        cand_dirs = [d for d in os.listdir(evals_dir) if d.startswith("c")]
        assert len(cand_dirs) >= 1

        c000 = os.path.join(evals_dir, "c00000")
        assert os.path.isfile(os.path.join(c000, "skill.md"))
        assert os.path.isfile(os.path.join(c000, "meta.json"))

        # Check meta.json content
        with open(os.path.join(c000, "meta.json")) as f:
            meta = json.load(f)
        assert "candidate_idx" in meta
        assert "average_score" in meta

        # Check skill.md contains candidate text
        with open(os.path.join(c000, "skill.md")) as f:
            skill = f.read()
        assert "prompt" in skill.lower() or "hello" in skill.lower()

        # Check task files exist and contain side_info
        tasks_dir = os.path.join(c000, "tasks")
        assert os.path.isdir(tasks_dir)
        task_files = os.listdir(tasks_dir)
        assert len(task_files) >= 1
        with open(os.path.join(tasks_dir, task_files[0])) as f:
            task_data = json.load(f)
        assert "score" in task_data
        assert "side_info" in task_data
        assert "feedback" in task_data["side_info"]
