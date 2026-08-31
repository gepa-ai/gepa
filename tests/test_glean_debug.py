from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from glean_gepa.debug import set_debug
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.runner import _make_reflection_lm, _parse_args


@pytest.fixture(autouse=True)
def debug_output_disabled():
    set_debug(False)
    yield
    set_debug(False)


def test_debug_flag_is_available_on_runner_cli():
    assert _parse_args(["--debug"]).debug is True


def test_evalcli_diagnostics_are_hidden_by_default(capsys):
    client = EvalCliClient(binary="/fake/evalcli")
    complete = [{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 1}]}]
    with patch.object(client, "_invoke_json", return_value=complete):
        client.wait_for_eval_run("run_123", poll_interval_sec=0)

    assert capsys.readouterr().out == ""


def test_evalcli_diagnostics_are_visible_in_debug_mode(capsys):
    set_debug(True)
    client = EvalCliClient(binary="/fake/evalcli")
    complete = [{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 1}]}]
    with patch.object(client, "_invoke_json", return_value=complete):
        client.wait_for_eval_run("run_123", poll_interval_sec=0)

    output = capsys.readouterr().out
    assert "Waiting for eval run run_123" in output
    assert "completed successfully" in output


def test_reflection_prompt_is_hidden_by_default(capsys):
    client = Mock()
    client.responses.create.return_value = Mock(output_text="ack")
    with patch("glean_gepa.runner.create_qe_openai_client", return_value=client):
        reflection_lm = _make_reflection_lm(
            "test-model",
            qe_project="test-project",
            qe_instance="test-instance",
            authenticated_email="test@example.com",
        )

    assert reflection_lm("sensitive prompt") == "ack"
    assert capsys.readouterr().out == ""
