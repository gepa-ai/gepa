"""Build small eval sets containing only the high-signal entries of a larger eval set.

Cortex has no way to run a subset of an existing eval set, so reproducing a handful of
failing entries requires uploading them as their own eval set version.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from glean_gepa.evalcli_client import EvalCliClient, EvalCliError

FOCUSED_EVAL_SET_NAME_PREFIX = "gepa-high-signal"
DEFAULT_BUCKET_TYPE = "SESSION"


@dataclass(frozen=True)
class FocusedEvalSet:
    name: str
    version: str
    entry_count: int


def focused_eval_set_name(base_eval_set_name: str) -> str:
    """Return a DSE-compatible lowercase slug for the focused eval set."""
    base_slug = re.sub(r"[^a-z0-9_-]+", "-", base_eval_set_name.lower()).strip("-_")
    return f"{FOCUSED_EVAL_SET_NAME_PREFIX}-{base_slug or 'eval-set'}"


def focused_eval_set_version(base_eval_set_version: str, entry_ids: Sequence[str]) -> str:
    """Deterministic version so the same entry set is reused instead of re-uploaded."""
    digest = hashlib.md5(",".join(sorted(entry_ids)).encode()).hexdigest()[:12]
    return f"{base_eval_set_version}_hs_{digest}"


def build_upload_entry(source_entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Map a listed EvalSetEntry onto the UploadEvalSetEntry shape.

    Returns None when the entry lacks both a deployment and any replayable handle.
    """
    deployment_id = source_entry.get("deploymentId")
    if not deployment_id:
        return None

    tracking = source_entry.get("sourceTrackingInfo") or {}
    source_trace = source_entry.get("sourceTrace") or {}
    entry_input = source_entry.get("input") or {}

    entry: dict[str, Any] = {"deploymentId": deployment_id}

    optional_fields = {
        "user": source_entry.get("user"),
        "traceId": tracking.get("traceId") or source_trace.get("id"),
        "stt": tracking.get("sessionTrackingToken"),
        "qtt": tracking.get("queryTrackingToken"),
        "runId": tracking.get("runId"),
        "query": entry_input.get("query"),
    }
    entry.update({key: value for key, value in optional_fields.items() if value})

    if not any(entry.get(key) for key in ("traceId", "stt", "qtt", "query")):
        return None
    return entry


def build_upload_eval_set_request(
    *,
    name: str,
    version: str,
    entries: Sequence[Mapping[str, Any]],
    bucket_type: str = DEFAULT_BUCKET_TYPE,
    base_eval_set_name: str | None = None,
    base_eval_set_version: str | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "name": name,
        "version": version,
        "bucketType": bucket_type,
        "entries": [dict(entry) for entry in entries],
        "useUploadJob": True,
    }
    if base_eval_set_name or base_eval_set_version:
        request["metadata"] = {
            "gepaSourceEvalSetName": base_eval_set_name,
            "gepaSourceEvalSetVersion": base_eval_set_version,
        }
    return request


def select_source_entries(
    source_entries: Sequence[Mapping[str, Any]],
    entry_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    wanted = set(entry_ids)
    return [entry for entry in source_entries if str(entry.get("id") or "") in wanted]


def ensure_focused_eval_set(
    evalcli: EvalCliClient,
    *,
    base_eval_set_name: str,
    base_eval_set_version: str,
    deployment_ids: list[str],
    entry_ids: Sequence[str],
    bucket_type: str = DEFAULT_BUCKET_TYPE,
) -> FocusedEvalSet | None:
    """Create (or reuse) an eval set version holding only `entry_ids`.

    Returns None when none of the requested entries can be replayed.
    """
    if not entry_ids:
        return None

    name = focused_eval_set_name(base_eval_set_name)
    version = focused_eval_set_version(base_eval_set_version, entry_ids)

    existing = evalcli.get_eval_set_version(eval_set_name=name, eval_set_version=version)
    if existing is not None:
        print(f"[Focused eval set] Found existing {name}:{version}; waiting for entries")
        try:
            existing_entries = evalcli.wait_for_eval_set_entries(
                eval_set_name=name,
                eval_set_version=version,
                deployment_ids=deployment_ids,
                expected_count=len(entry_ids),
            )
        except EvalCliError as exc:
            print(f"[Focused eval set] {exc}")
            return None
        print(f"[Focused eval set] Reusing {name}:{version} with {len(existing_entries)} entries")
        return FocusedEvalSet(name=name, version=version, entry_count=len(existing_entries))

    source_entries = evalcli.list_eval_set_entries(
        eval_set_name=base_eval_set_name,
        eval_set_version=base_eval_set_version,
        deployment_ids=deployment_ids,
    )
    selected = select_source_entries(source_entries, entry_ids)
    upload_entries = [
        upload_entry
        for entry in selected
        if (upload_entry := build_upload_entry(entry)) is not None
    ]

    if not upload_entries:
        print(
            f"[Focused eval set] None of the {len(entry_ids)} high-signal entries could be "
            f"resolved from {base_eval_set_name}:{base_eval_set_version}"
        )
        return None

    print(f"[Focused eval set] Uploading {name}:{version} with {len(upload_entries)} entries")
    try:
        evalcli.upload_eval_set(
            build_upload_eval_set_request(
                name=name,
                version=version,
                entries=upload_entries,
                bucket_type=bucket_type,
                base_eval_set_name=base_eval_set_name,
                base_eval_set_version=base_eval_set_version,
            )
        )
    except EvalCliError as exc:
        if "already exists" not in str(exc).lower():
            raise
        print(f"[Focused eval set] {name}:{version} was created concurrently; reusing it")

    try:
        ingested = evalcli.wait_for_eval_set_entries(
            eval_set_name=name,
            eval_set_version=version,
            deployment_ids=deployment_ids,
            expected_count=len(upload_entries),
        )
    except EvalCliError as exc:
        print(f"[Focused eval set] {exc}")
        return None

    return FocusedEvalSet(name=name, version=version, entry_count=len(ingested))
