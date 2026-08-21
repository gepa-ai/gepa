from __future__ import annotations

from unittest.mock import MagicMock

from glean_gepa.evalcli_client import EvalCliError
from glean_gepa.focused_evalset import (
    build_upload_entry,
    build_upload_eval_set_request,
    ensure_focused_eval_set,
    focused_eval_set_name,
    focused_eval_set_version,
    select_source_entries,
)

SOURCE_ENTRY = {
    "id": "entry-1",
    "user": "someone@glean.com",
    "deploymentId": "scio-prod",
    "input": {"query": "how do I list shell files"},
    "sourceTrackingInfo": {
        "traceId": "trace-1",
        "sessionTrackingToken": "stt-1",
        "queryTrackingToken": "qtt-1",
        "runId": "run-1",
    },
}


def test_focused_eval_set_version_is_deterministic_and_order_independent():
    first = focused_eval_set_version("20260806", ["b", "a"])
    second = focused_eval_set_version("20260806", ["a", "b"])

    assert first == second
    assert first.startswith("20260806_hs_")
    assert focused_eval_set_version("20260806", ["a"]) != first


def test_focused_eval_set_name_derives_from_base():
    assert focused_eval_set_name("Glean Chat V2") == "gepa-high-signal-glean-chat-v2"


def test_focused_eval_set_name_removes_unsupported_characters():
    name = focused_eval_set_name("Glean.Chat / V2 (Small)")

    assert name == "gepa-high-signal-glean-chat-v2-small"
    assert name.replace("-", "").replace("_", "").isalnum()


def test_build_upload_entry_maps_replay_handles():
    entry = build_upload_entry(SOURCE_ENTRY)

    assert entry == {
        "deploymentId": "scio-prod",
        "user": "someone@glean.com",
        "traceId": "trace-1",
        "stt": "stt-1",
        "qtt": "qtt-1",
        "runId": "run-1",
        "query": "how do I list shell files",
    }


def test_build_upload_entry_falls_back_to_source_trace_id():
    entry = build_upload_entry(
        {
            "id": "entry-2",
            "deploymentId": "scio-prod",
            "sourceTrace": {"id": "trace-9"},
        }
    )

    assert entry is not None
    assert entry["traceId"] == "trace-9"


def test_build_upload_entry_requires_deployment_and_replay_handle():
    assert build_upload_entry({"id": "x", "input": {"query": "q"}}) is None
    assert build_upload_entry({"id": "x", "deploymentId": "scio-prod"}) is None


def test_build_upload_eval_set_request_shape():
    request = build_upload_eval_set_request(
        name="gepa-high-signal-glean-chat-v2",
        version="20260806_hs_deadbeef",
        entries=[{"deploymentId": "scio-prod", "query": "q"}],
        base_eval_set_name="Glean Chat V2",
        base_eval_set_version="20260806",
    )

    assert request["useUploadJob"] is True
    assert request["bucketType"] == "SESSION"
    assert request["entries"] == [{"deploymentId": "scio-prod", "query": "q"}]
    assert request["metadata"]["gepaSourceEvalSetVersion"] == "20260806"


def test_select_source_entries_filters_by_id():
    entries = [SOURCE_ENTRY, {"id": "entry-other", "deploymentId": "scio-prod"}]

    assert select_source_entries(entries, ["entry-1"]) == [SOURCE_ENTRY]


def test_ensure_focused_eval_set_uploads_only_high_signal_entries():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [
        SOURCE_ENTRY,
        {"id": "entry-clean", "deploymentId": "scio-prod", "input": {"query": "clean"}},
    ]
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-1"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Glean Chat V2",
        base_eval_set_version="20260806",
        deployment_ids=["scio-prod"],
        entry_ids=["entry-1"],
    )

    assert focused is not None
    assert focused.name == "gepa-high-signal-glean-chat-v2"
    assert focused.entry_count == 1

    uploaded = evalcli.upload_eval_set.call_args[0][0]
    assert len(uploaded["entries"]) == 1
    assert uploaded["entries"][0]["traceId"] == "trace-1"


def test_ensure_focused_eval_set_reuses_existing_version():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = {"name": "gepa-high-signal-glean-chat-v2"}
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "existing-1"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Glean Chat V2",
        base_eval_set_version="20260806",
        deployment_ids=["scio-prod"],
        entry_ids=["entry-1"],
    )

    assert focused is not None
    assert focused.entry_count == 1
    evalcli.wait_for_eval_set_entries.assert_called_once_with(
        eval_set_name="gepa-high-signal-glean-chat-v2",
        eval_set_version=focused.version,
        deployment_ids=["scio-prod"],
        expected_count=1,
    )
    evalcli.upload_eval_set.assert_not_called()


def test_ensure_focused_eval_set_reuses_version_created_during_upload():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [SOURCE_ENTRY]
    evalcli.upload_eval_set.side_effect = EvalCliError(
        'Eval set with name "gepa-high-signal-glean-chat-v2" already exists'
    )
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "existing-1"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Glean Chat V2",
        base_eval_set_version="20260806",
        deployment_ids=["scio-prod"],
        entry_ids=["entry-1"],
    )

    assert focused is not None
    assert focused.entry_count == 1


def test_ensure_focused_eval_set_returns_none_without_entry_ids():
    evalcli = MagicMock()

    assert (
        ensure_focused_eval_set(
            evalcli,
            base_eval_set_name="Glean Chat V2",
            base_eval_set_version="20260806",
            deployment_ids=["scio-prod"],
            entry_ids=[],
        )
        is None
    )
    evalcli.upload_eval_set.assert_not_called()


def test_ensure_focused_eval_set_returns_none_when_entries_unresolvable():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [{"id": "entry-1"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Glean Chat V2",
        base_eval_set_version="20260806",
        deployment_ids=["scio-prod"],
        entry_ids=["entry-1"],
    )

    assert focused is None
    evalcli.upload_eval_set.assert_not_called()
