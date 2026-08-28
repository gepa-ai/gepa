from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import CODING_HARNESS_SC_PARAMS, ALRunner
from glean_gepa.evalcli_client import EvalCliClient, EvalCliError, _subprocess_env


def test_coding_harness_sc_params_selects_coding_agent_loop():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))

    params = runner._build_sc_params("gpt", "")

    assert params.startswith(CODING_HARNESS_SC_PARAMS)
    assert "co.internal_looping_pyagent_default_route_override=coding_agent_loop" in params
    assert "co.py_agent_route_override=o3_agentic_loop" not in params
    assert "co.lo.cao.agentic_loop_sc_params=co.so.enable_for_agentic_loop%3D1%2C" in params
    assert "co.so.ptc_only_tools%3Dglean_search%253Bglean_document_reader" in params


def test_create_eval_run_invokes_evalcli_with_expected_args():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value={"id": "run_123"}) as mock_invoke:
        run_id = client.create_eval_run(
            eval_run_id="run_123",
            eval_set_name="AI Answers Small",
            eval_set_version="20260403",
            deployment_ids=["scio-prod"],
            description="GEPA eval run for AI Answers Small:20260403",
            sc_params="co.debug_mode=1",
            eval_params="experimental_queue=temp-system-prompt-optimization,gleanchat_agent=FAST",
        )

    assert run_id == "run_123"
    mock_invoke.assert_called_once()
    args = mock_invoke.call_args[0]
    assert args[0:4] == ("run", "create", "--eval-set", "AI Answers Small:20260403")
    preset_idx = args.index("--preset")
    assert args[preset_idx + 1] == "Coding Harness"
    assert "--sc-params" in args
    assert "--eval-params" in args


def test_create_judge_run_parses_response_list():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value=[{"id": "judge_456", "status": "SUBMITTED"}]):
        judge_id = client.create_judge_run(student_eval_id="student", teacher_eval_id="teacher")

    assert judge_id == "judge_456"


def test_list_eval_set_versions_returns_version_rows():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value={"evalSetVersions": [{"version": "20260827"}]}) as mock_invoke:
        rows = client.list_eval_set_versions(eval_set_name="Glean Chat V2 Medium", deployment_ids=["scio-prod"])

    assert rows == [{"version": "20260827"}]
    assert mock_invoke.call_args[0] == (
        "evalsets",
        "list",
        "--name",
        "Glean Chat V2 Medium",
        "--deployment-ids",
        "scio-prod",
    )


def test_wait_for_judge_run_raises_on_failure():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value={"id": "judge_456", "status": "FAILED"}):
        with pytest.raises(EvalCliError, match="ended with status FAILED"):
            client.wait_for_judge_run("judge_456", poll_interval_sec=0, timeout_sec=1)


def test_subprocess_env_replaces_unreliable_ssl_cert(monkeypatch, tmp_path):
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("fake-ca", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", "/var/folders/abc/socketFirewallCa.crt")
    monkeypatch.setenv("SSL_CERT_DIR", "/var/folders/abc")
    monkeypatch.setattr(
        "glean_gepa.evalcli_client._resolve_ca_bundle",
        lambda: str(ca_bundle),
    )

    env = _subprocess_env()

    assert env["SSL_CERT_FILE"] == str(ca_bundle)
    assert env["REQUESTS_CA_BUNDLE"] == str(ca_bundle)
    assert "SSL_CERT_DIR" not in env


def test_subprocess_env_sets_ssl_cert_when_missing(monkeypatch, tmp_path):
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("fake-ca", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setattr(
        "glean_gepa.evalcli_client._resolve_ca_bundle",
        lambda: str(ca_bundle),
    )

    env = _subprocess_env()

    assert env["SSL_CERT_FILE"] == str(ca_bundle)
    assert env["REQUESTS_CA_BUNDLE"] == str(ca_bundle)


def test_subprocess_env_preserves_existing_ssl_cert(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/ca.pem")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    env = _subprocess_env()

    assert env["SSL_CERT_FILE"] == "/custom/ca.pem"
    assert "REQUESTS_CA_BUNDLE" not in env


def test_wait_for_eval_run_retries_transient_errors():
    client = EvalCliClient(binary="/fake/evalcli")
    in_progress = [{"taskCountsByStatus": [{"status": "TASK_SUBMITTED", "count": 1}]}]
    complete = [{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 1}]}]
    with patch.object(
        client,
        "_invoke_json",
        side_effect=[
            EvalCliError("stderr: Error: API request failed: 502\nResponse: Connection refused"),
            in_progress,
            complete,
        ],
    ) as mock_invoke:
        with patch("glean_gepa.evalcli_client.time.sleep"):
            client.wait_for_eval_run("run_123", poll_interval_sec=0, timeout_sec=10)

    assert mock_invoke.call_count == 3


def test_wait_for_eval_run_raises_on_non_transient_errors():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(
        client,
        "_invoke_json",
        side_effect=EvalCliError("stderr: auth failed"),
    ):
        with pytest.raises(EvalCliError, match="auth failed"):
            client.wait_for_eval_run("run_123", poll_interval_sec=0, timeout_sec=1)


def test_invoke_raises_on_nonzero_exit():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch("glean_gepa.evalcli_client.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="auth failed")
        with pytest.raises(EvalCliError, match="auth failed"):
            client._invoke("whoami")
