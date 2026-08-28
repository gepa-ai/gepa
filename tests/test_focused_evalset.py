from unittest.mock import MagicMock

from glean_gepa.focused_evalset import (
    build_upload_eval_set_request,
    ensure_focused_eval_set,
    focused_eval_set_version,
)


def test_focused_eval_set_version_is_stable_for_the_same_entries():
    assert focused_eval_set_version("v1", ["b", "a"]) == focused_eval_set_version("v1", ["a", "b"])


def test_build_upload_request_keeps_source_metadata():
    request = build_upload_eval_set_request(
        name="gepa-high-signal-example",
        version="v1_hs_123",
        entries=[{"deploymentId": "prod", "stt": "session"}],
        base_eval_set_name="Example",
        base_eval_set_version="v1",
    )

    assert request["useUploadJob"] is True
    assert request["metadata"] == {
        "gepaSourceEvalSetName": "Example",
        "gepaSourceEvalSetVersion": "v1",
    }


def test_ensure_focused_eval_set_uploads_only_requested_entries():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [
        {"id": "keep", "deploymentId": "prod", "stt": "session-1"},
        {"id": "drop", "deploymentId": "prod", "stt": "session-2"},
    ]
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
    )

    assert focused is not None
    request = evalcli.upload_eval_set.call_args.args[0]
    assert request["entries"] == [{"deploymentId": "prod", "stt": "session-1"}]
    assert focused.entry_count == 1


def test_ensure_focused_eval_set_reuses_an_existing_version():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = {"name": "existing"}
    evalcli.list_eval_set_entries.return_value = [{"id": "existing-entry"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
    )

    assert focused is not None
    evalcli.upload_eval_set.assert_not_called()
