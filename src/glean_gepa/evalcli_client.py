"""Subprocess wrapper for Glean's evalcli."""

from __future__ import annotations

import json
import os
import ssl
import subprocess
import tempfile
import time
from typing import Any

from glean_gepa.debug import debug_print

CODING_HARNESS_PRESET = "Coding Harness"
DEFAULT_PRESET = CODING_HARNESS_PRESET

CORRECTNESS_INPUT_MAPPINGS = json.dumps(
    [
        {"entryType": "TEST", "name": "Query", "path": "Query", "sourceType": "EVAL_SET_PROTO"},
        {
            "entryType": "TEST",
            "name": "Response",
            "path": "EvalChatResponseInfo.ActResponse",
            "sourceType": "EVAL_RUN_OUTPUT",
        },
        {
            "entryType": "BASE",
            "name": "CanonicalAnswer",
            "path": "EvalChatResponseInfo.ActResponse",
            "sourceType": "EVAL_RUN_OUTPUT",
        },
    ]
)

CORRECTNESS_RUN_PARAMS = json.dumps(
    {"Judge Type": "DIRECT_CORRECTNESS", "Llm model": "default", "Use Cache": "true"}
)

TERMINAL_JUDGE_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}

TERMINAL_TASK_STATUSES = {
    "TASK_SUCCEEDED",
    "TASK_FAILED",
    "TASK_DEPENDENCY_FAILED",
    "TASK_CANCELLED",
    "TASK_TIMED_OUT",
    "TASK_ABORTED",
}

TRANSIENT_EVALCLI_PATTERNS = (
    "API request failed: 502",
    "API request failed: 503",
    "API request failed: 504",
    "Connection refused",
    "Connection reset",
)


class EvalCliError(RuntimeError):
    pass


def _is_unreliable_ca_bundle(path: str) -> bool:
    # Cursor sandbox and some proxies inject ephemeral temp CA files that break child Python tools.
    normalized = path.lower()
    return (
        "/var/folders/" in path
        or normalized.startswith("/tmp/")
        or "socketfirewallca.crt" in normalized
    )


def _resolve_ca_bundle() -> str | None:
    candidates = [
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]

    defaults = ssl.get_default_verify_paths()
    if defaults.openssl_cafile and os.path.isfile(defaults.openssl_cafile):
        candidates.append(defaults.openssl_cafile)
    if (
        defaults.cafile
        and os.path.isfile(defaults.cafile)
        and not _is_unreliable_ca_bundle(defaults.cafile)
    ):
        candidates.append(defaults.cafile)

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    try:
        import certifi

        return certifi.where()
    except ImportError:
        return None


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    ca_bundle = _resolve_ca_bundle()
    current = env.get("SSL_CERT_FILE")
    if ca_bundle and (not current or _is_unreliable_ca_bundle(current)):
        env["SSL_CERT_FILE"] = ca_bundle
        env["REQUESTS_CA_BUNDLE"] = ca_bundle
        env.pop("SSL_CERT_DIR", None)
    return env


def _is_transient_evalcli_error(exc: EvalCliError) -> bool:
    message = str(exc)
    return any(pattern in message for pattern in TRANSIENT_EVALCLI_PATTERNS)


def _is_eval_complete(status: dict[str, Any]) -> bool:
    task_counts = status.get("taskCountsByStatus") or []
    active_counts = [entry for entry in task_counts if (entry.get("count") or 0) > 0]
    if not active_counts:
        return False
    return all(entry.get("status") in TERMINAL_TASK_STATUSES for entry in active_counts)


class EvalCliClient:
    """Thin wrapper around the evalcli binary for eval runs, judges, and analysis."""

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("EVALCLI_BIN", "evalcli")

    def _invoke(self, *args: str, expect_json: bool = False) -> Any:
        cmd = [self.binary, *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env())
        if proc.returncode != 0:
            raise EvalCliError(
                f"evalcli failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                f"stderr: {proc.stderr.strip()}\n"
                f"stdout: {proc.stdout.strip()}"
            )
        stdout = proc.stdout.strip()
        if expect_json:
            return json.loads(stdout) if stdout else None
        return stdout

    def _invoke_json(self, *args: str) -> Any:
        if "--json" not in args:
            args = (*args, "--json")
        return self._invoke(*args, expect_json=True)

    def create_eval_run(
        self,
        *,
        eval_run_id: str,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        description: str,
        sc_params: str | None = None,
        eval_params: str | None = None,
        preset: str = DEFAULT_PRESET,
    ) -> str:
        eval_set = f"{eval_set_name}:{eval_set_version}"
        cmd = [
            "run",
            "create",
            "--eval-set",
            eval_set,
            "--preset",
            preset,
            "--deployment-ids",
            *deployment_ids,
            "--id",
            eval_run_id,
            "--description",
            description,
        ]
        if sc_params:
            cmd.extend(["--sc-params", sc_params])
        if eval_params:
            cmd.extend(["--eval-params", eval_params])

        result = self._invoke_json(*cmd)
        if not isinstance(result, dict):
            raise EvalCliError(f"Unexpected eval run create response: {result!r}")
        return str(result.get("id") or eval_run_id)

    def wait_for_eval_run(
        self,
        eval_run_id: str,
        *,
        poll_interval_sec: int = 60,
        timeout_sec: int | None = None,
    ) -> None:
        print(f"Waiting for eval run {eval_run_id} to complete...")
        started_at = time.monotonic()
        while True:
            if timeout_sec is not None and time.monotonic() - started_at >= timeout_sec:
                raise EvalCliError(f"Eval run {eval_run_id} timed out after {timeout_sec}s")
            try:
                statuses = self._invoke_json("run", "status", "--id", eval_run_id)
            except EvalCliError as exc:
                if not _is_transient_evalcli_error(exc):
                    raise
                print(
                    f"Transient Cortex error while polling {eval_run_id}; "
                    f"retrying in {poll_interval_sec}s..."
                )
                time.sleep(poll_interval_sec)
                continue

            print(f"Eval run {eval_run_id} status: {json.dumps(statuses, sort_keys=True, default=str)}")
            if isinstance(statuses, list) and statuses and _is_eval_complete(statuses[0]):
                print(f"Eval run {eval_run_id} completed successfully")
                return

            time.sleep(poll_interval_sec)

    def get_eval_set_version(self, *, eval_set_name: str, eval_set_version: str) -> dict[str, Any] | None:
        """Return the eval set version, or None when it does not exist yet."""
        try:
            result = self._invoke_json(
                "evalsets",
                "get",
                "--name",
                eval_set_name,
                "--version",
                eval_set_version,
            )
        except EvalCliError:
            return None
        return result if isinstance(result, dict) else None

    def list_eval_set_versions(self, *, eval_set_name: str, deployment_ids: list[str]) -> list[dict[str, Any]]:
        """List the available versions of an eval set for the given deployments."""
        result = self._invoke_json(
            "evalsets",
            "list",
            "--name",
            eval_set_name,
            "--deployment-ids",
            *deployment_ids,
        )
        if isinstance(result, list):
            rows = result
        elif isinstance(result, dict):
            rows = (
                result.get("evalSetVersions")
                or result.get("versions")
                or result.get("evalSets")
                or result.get("items")
                or []
            )
        else:
            raise EvalCliError(f"Unexpected eval set versions response: {result!r}")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise EvalCliError(f"Unexpected eval set versions response: {result!r}")
        return rows

    def list_eval_set_entries(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch every entry of an eval set version, paging until exhausted."""
        entries: list[dict[str, Any]] = []
        page = 1
        while True:
            result = self._invoke_json(
                "evalsets",
                "entries",
                "--name",
                eval_set_name,
                "--version",
                eval_set_version,
                "--deployment-ids",
                *deployment_ids,
                "--page",
                str(page),
                "--page-size",
                str(page_size),
            )
            if not isinstance(result, dict):
                raise EvalCliError(f"Unexpected evalsets entries response: {result!r}")

            batch = result.get("evalSetEntries") or []
            entries.extend(entry for entry in batch if isinstance(entry, dict))

            total_pages = (result.get("pageInfo") or {}).get("totalPages") or 1
            if not batch or page >= total_pages:
                return entries
            page += 1

    def upload_eval_set(self, request: dict[str, Any]) -> None:
        """Upload a new eval set version. Entries are ingested asynchronously."""
        debug_print(f"Uploading eval set payload:\n{json.dumps(request, indent=2)}")
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump(request, handle)
            handle.close()
            self._invoke("evalsets", "upload", "--file", handle.name)
        finally:
            os.unlink(handle.name)

    def wait_for_eval_set_entries(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        expected_count: int,
        poll_interval_sec: int = 15,
        timeout_sec: int = 900,
    ) -> list[dict[str, Any]]:
        """Poll until the uploaded eval set version has ingested all of its entries."""
        print(f"Waiting for eval set {eval_set_name}:{eval_set_version} to ingest {expected_count} entries...")
        elapsed = 0
        entries: list[dict[str, Any]] = []
        while elapsed < timeout_sec:
            try:
                entries = self.list_eval_set_entries(
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                )
            except EvalCliError:
                entries = []

            if len(entries) >= expected_count:
                print(f"Eval set {eval_set_name}:{eval_set_version} ready with {len(entries)} entries")
                return entries

            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec

        raise EvalCliError(
            f"Eval set {eval_set_name}:{eval_set_version} only ingested {len(entries)}/{expected_count} "
            f"entries after {timeout_sec}s"
        )

    def create_judge_run(self, *, student_eval_id: str, teacher_eval_id: str) -> str:
        results = self._invoke_json(
            "judge",
            "create",
            "--eval-run-id",
            student_eval_id,
            "--base-eval-run-id",
            teacher_eval_id,
            "--judge-type",
            "CORRECTNESS",
            "--run-params",
            CORRECTNESS_RUN_PARAMS,
            "--input-mappings",
            CORRECTNESS_INPUT_MAPPINGS,
        )
        if not results:
            raise EvalCliError("judge create returned empty response")

        judge_run = results[0] if isinstance(results, list) else results
        if not isinstance(judge_run, dict):
            raise EvalCliError(f"Unexpected judge create response: {judge_run!r}")

        judge_run_id = judge_run.get("id")
        if not judge_run_id:
            raise EvalCliError(f"judge create response missing id: {judge_run}")
        return str(judge_run_id)

    def wait_for_judge_run(
        self,
        judge_run_id: str,
        *,
        poll_interval_sec: int = 60,
        timeout_sec: int = 3600,
    ) -> None:
        print(f"Waiting for judge run {judge_run_id} to complete...")
        elapsed = 0
        while elapsed < timeout_sec:
            run = self._invoke_json("judge", "get", "--id", judge_run_id)
            status = run.get("status") if isinstance(run, dict) else None
            if status in TERMINAL_JUDGE_STATUSES:
                if status != "SUCCEEDED":
                    raise EvalCliError(f"Judge run {judge_run_id} ended with status {status}")
                print(f"Judge run {judge_run_id} completed successfully")
                return
            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec
        raise EvalCliError(f"Judge run {judge_run_id} timed out after {timeout_sec}s")

    def get_analysis_view(self, student_eval_id: str, teacher_eval_id: str) -> dict[str, Any]:
        result = self._invoke_json(
            "analyze",
            "view",
            "--test-run-ids",
            student_eval_id,
            "--base-run-id",
            teacher_eval_id,
        )
        if not isinstance(result, dict):
            raise EvalCliError(f"Unexpected analysis view response: {result!r}")
        return result

    def get_analysis_details(
        self,
        *,
        entry_ids: list[str],
        eval_run_ids: list[str],
        deployment_id: str,
    ) -> list[dict[str, Any]]:
        if not entry_ids:
            return []
        result = self._invoke_json(
            "analyze",
            "details",
            "--entry-ids",
            *entry_ids,
            "--eval-run-ids",
            *eval_run_ids,
            "--deployment-id",
            deployment_id,
        )
        if not isinstance(result, list):
            raise EvalCliError(f"Unexpected analysis details response: {result!r}")
        return result

    def get_analysis_trace(
        self,
        *,
        deployment_id: str,
        trace_id: str,
        start_time_millis: int,
        end_time_millis: int,
    ) -> dict[str, Any]:
        result = self._invoke_json(
            "analyze",
            "trace",
            "--deployment-id",
            deployment_id,
            "--trace-id",
            trace_id,
            "--start-time-millis",
            str(start_time_millis),
            "--end-time-millis",
            str(end_time_millis),
        )
        if not isinstance(result, dict):
            raise EvalCliError(f"Unexpected analysis trace response: {result!r}")
        return result
