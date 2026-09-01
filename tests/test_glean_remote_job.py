from __future__ import annotations

import json

import pytest

from glean_gepa.remote_job import build_runner_args


def test_build_runner_args_uses_cloud_run_execution_and_writes_seed(tmp_path):
    args, run_dir = build_runner_args(
        {
            "CLOUD_RUN_EXECUTION": "gepa-optimize-abc123",
            "GEPA_RUN_ROOT": str(tmp_path),
            "GEPA_RUNNER_ARGS_JSON": json.dumps(["--max_metric_calls", "5"]),
            "GEPA_SEED_CANDIDATE_JSON": json.dumps({"WRITING_CODE": "seed"}),
        }
    )

    assert run_dir == tmp_path / "gepa-optimize-abc123"
    assert args[-4:-2] == ["--run_dir", str(run_dir)]
    assert args[-2:] == ["--seed_candidate", str(run_dir / "seed_candidate.json")]
    assert json.loads((run_dir / "seed_candidate.json").read_text()) == {"WRITING_CODE": "seed"}


def test_build_runner_args_preserves_explicit_run_dir(tmp_path):
    explicit = tmp_path / "explicit"
    args, _run_dir = build_runner_args(
        {
            "GEPA_RUN_ROOT": str(tmp_path),
            "GEPA_RUNNER_ARGS_JSON": json.dumps(["--fake_flow", f"--run_dir={explicit}"]),
        }
    )

    assert args == ["--fake_flow", f"--run_dir={explicit}"]


@pytest.mark.parametrize("value", ["{}", "[1]", "not-json"])
def test_build_runner_args_rejects_invalid_json(value, tmp_path):
    with pytest.raises(ValueError, match="JSON array of strings"):
        build_runner_args({"GEPA_RUN_ROOT": str(tmp_path), "GEPA_RUNNER_ARGS_JSON": value})
