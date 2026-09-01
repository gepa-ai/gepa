from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from glean_gepa.openai_client import create_qe_openai_client, format_exception_chain, get_perfeval_secret
from glean_gepa.runner import _make_reflection_lm


def test_create_qe_openai_client_uses_instance_hostname() -> None:
    ssl_context = Mock()
    http_client = Mock()
    with (
        patch("glean_gepa.openai_client.truststore.SSLContext", return_value=ssl_context),
        patch("glean_gepa.openai_client.openai.DefaultHttpxClient", return_value=http_client) as http_client_cls,
        patch("glean_gepa.openai_client.openai.OpenAI") as openai_cls,
    ):
        create_qe_openai_client("glean-dev")

    http_client_cls.assert_called_once_with(verify=ssl_context)
    openai_cls.assert_called_once_with(
        base_url="https://glean-dev-be.glean.com/qe/llm",
        api_key="dummy",
        timeout=600.0,
        max_retries=5,
        http_client=http_client,
    )


def test_format_exception_chain_includes_transport_root_cause() -> None:
    root = OSError("temporary DNS failure")
    outer = RuntimeError("Connection error")
    outer.__cause__ = root

    assert format_exception_chain(outer) == ("RuntimeError: Connection error <- OSError: temporary DNS failure")


def test_reflection_lm_uses_qe_responses_auth_body(capsys: pytest.CaptureFixture[str]) -> None:
    response = Mock(output_text="ack")
    client = Mock()
    client.responses.create.return_value = response

    with patch("glean_gepa.runner.create_qe_openai_client", return_value=client):
        reflection_lm = _make_reflection_lm(
            "OPEN_AI:GPT5_LATEST",
            qe_project="dev-sandbox-334901",
            qe_instance="glean-dev",
            authenticated_email="cathy.chen@glean.com",
        )

    assert reflection_lm("Just say ack") == "ack"
    assert capsys.readouterr().out == (
        "QE reflection LLM call 1: requesting model=OPEN_AI:GPT5_LATEST, prompt_chars=12\n"
        "QE reflection LLM call 1 prompt:\n"
        "Just say ack\n"
        "QE reflection LLM call 1: received response_chars=3\n"
        "QE reflection LLM call 1 response:\n"
        "ack\n"
    )
    client.responses.create.assert_called_once_with(
        model="OPEN_AI:GPT5_LATEST",
        input="Just say ack",
        max_output_tokens=4096,
        extra_body={
            "perf_eval_secret": get_perfeval_secret("dev-sandbox-334901"),
            "source_info": {
                "clientInitiator": "USER",
                "feature": "INTEGRATION_TEST",
            },
            "authenticated_email": "cathy.chen@glean.com",
        },
    )


def test_reflection_lm_raises_for_an_empty_qe_response() -> None:
    client = Mock()
    client.responses.create.return_value = Mock(output_text="   ")

    with patch("glean_gepa.runner.create_qe_openai_client", return_value=client):
        reflection_lm = _make_reflection_lm(
            "OPEN_AI:GPT5_LATEST",
            qe_project="dev-sandbox-334901",
            qe_instance="glean-dev",
            authenticated_email="cathy.chen@glean.com",
        )

    with pytest.raises(RuntimeError, match="QE reflection LLM returned an empty response"):
        reflection_lm("Just say ack")


def test_reflection_lm_surfaces_qe_request_errors() -> None:
    client = Mock()
    client.responses.create.side_effect = ConnectionError("proxy unavailable")

    with patch("glean_gepa.runner.create_qe_openai_client", return_value=client):
        reflection_lm = _make_reflection_lm(
            "OPEN_AI:GPT5_LATEST",
            qe_project="dev-sandbox-334901",
            qe_instance="glean-dev",
            authenticated_email="cathy.chen@glean.com",
        )

    with pytest.raises(RuntimeError, match="ConnectionError: proxy unavailable"):
        reflection_lm("Just say ack")
