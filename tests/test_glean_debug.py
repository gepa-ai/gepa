from __future__ import annotations

from unittest.mock import patch

import pytest

from glean_gepa.debug import set_debug
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.runner import _parse_args


@pytest.fixture(autouse=True)
def debug_output_disabled():
    set_debug(False)
    yield
    set_debug(False)


def test_debug_flag_is_available_on_runner_cli():
    assert _parse_args(["--debug"]).debug is True


def test_evalcli_status_diagnostics_are_visible_by_default(capsys):
    client = EvalCliClient(binary="/fake/evalcli")
    complete = [{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 1}]}]
    with patch.object(client, "_invoke_json", return_value=complete):
        client.wait_for_eval_run("run_123", poll_interval_sec=0)

    output = capsys.readouterr().out
    assert "Waiting for eval run run_123" in output
    assert "completed successfully" in output


def test_evalset_payload_is_hidden_by_default(capsys):
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke", return_value=""):
        client.upload_eval_set({"entries": [{"query": "sensitive prompt"}]})

    assert capsys.readouterr().out == ""


def test_evalset_payload_is_visible_in_debug_mode(capsys):
    set_debug(True)
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke", return_value=""):
        client.upload_eval_set({"entries": [{"query": "sensitive prompt"}]})

    assert "Uploading eval set payload" in capsys.readouterr().out
