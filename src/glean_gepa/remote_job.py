"""Remote one-shot job wrapper for :mod:`glean_gepa.runner`.

The wrapper keeps deployment-specific concerns out of the optimizer. It turns
Cloud Run environment variables into a durable run directory, accepts the
ordinary runner arguments as JSON, and writes a small status record alongside
the GEPA checkpoints.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from glean_gepa.runner import main as runner_main

DEFAULT_RUN_ROOT = "/mnt/gepa/runs"
DEFAULT_EVALCLI_PATH = "/opt/evalcli/eval_cli"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not safe:
        raise ValueError("GEPA_RUN_ID must contain at least one letter or digit")
    return safe


def _contains_option(args: Sequence[str], option: str) -> bool:
    return option in args or any(arg.startswith(f"{option}=") for arg in args)


def build_runner_args(env: Mapping[str, str]) -> tuple[list[str], Path]:
    raw_args = env.get("GEPA_RUNNER_ARGS_JSON", "[]")
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ValueError("GEPA_RUNNER_ARGS_JSON must be a JSON array of strings") from exc
    if not isinstance(parsed, list) or not all(isinstance(value, str) for value in parsed):
        raise ValueError("GEPA_RUNNER_ARGS_JSON must be a JSON array of strings")

    args = list(parsed)
    run_id = _safe_run_id(
        env.get("GEPA_RUN_ID")
        or env.get("CLOUD_RUN_EXECUTION")
        or env.get("CLOUD_RUN_JOB")
        or "manual"
    )
    run_dir = Path(env.get("GEPA_RUN_ROOT", DEFAULT_RUN_ROOT)).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not _contains_option(args, "--run_dir"):
        args.extend(["--run_dir", str(run_dir)])

    seed_json = env.get("GEPA_SEED_CANDIDATE_JSON")
    if seed_json and not _contains_option(args, "--seed_candidate"):
        try:
            seed = json.loads(seed_json)
        except json.JSONDecodeError as exc:
            raise ValueError("GEPA_SEED_CANDIDATE_JSON must contain valid JSON") from exc
        if not isinstance(seed, dict):
            raise ValueError("GEPA_SEED_CANDIDATE_JSON must contain a JSON object")
        seed_path = run_dir / "seed_candidate.json"
        seed_path.write_text(json.dumps(seed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.extend(["--seed_candidate", str(seed_path)])

    return args, run_dir


def _write_status(run_dir: Path, **fields: Any) -> None:
    path = run_dir / "remote_job_status.json"
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    payload = {**previous, **fields, "updated_at": _utc_now()}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _configure_evalcli(args: Sequence[str], env: Mapping[str, str]) -> None:
    if "--fake_flow" in args:
        return

    configured = env.get("EVALCLI_BIN") or shutil.which("evalcli") or DEFAULT_EVALCLI_PATH
    evalcli_path = Path(configured)
    if not evalcli_path.is_file():
        raise RuntimeError(
            "evalcli is required for a real Cortex run. Build a Linux evalcli bundle into the image "
            "or set EVALCLI_BIN to an installed executable."
        )
    os.environ["EVALCLI_BIN"] = str(evalcli_path)

    cookie = env.get("CORTEX_IAP_COOKIE")
    if cookie:
        subprocess.run(
            [str(evalcli_path), "login", "--cookie-value", cookie],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def main() -> None:
    args, run_dir = build_runner_args(os.environ)
    execution = os.environ.get("CLOUD_RUN_EXECUTION")
    attempt = int(os.environ.get("CLOUD_RUN_TASK_ATTEMPT", "0"))
    _write_status(
        run_dir,
        status="running",
        started_at=_utc_now(),
        execution=execution,
        attempt=attempt,
        runner_args=args,
    )
    try:
        _configure_evalcli(args, os.environ)
        runner_main(args)
    except BaseException as exc:
        _write_status(
            run_dir,
            status="failed",
            finished_at=_utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(traceback.format_exception(exc)),
        )
        raise
    else:
        _write_status(run_dir, status="succeeded", finished_at=_utc_now())


if __name__ == "__main__":
    main()
