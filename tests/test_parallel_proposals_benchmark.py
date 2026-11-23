# Copyright (c) 2026 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Exercise the offline benchmark's CLI, accounting, and reproducibility."""

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "parallel_proposals" / "benchmark.py"


def test_offline_benchmark_accounting_and_worker_independence(tmp_path):
    output = tmp_path / "results.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workers",
            "1",
            "2",
            "--pxn",
            "1x1",
            "1x2",
            "--minibatches",
            "2",
            "--eval-latency-ms",
            "0",
            "--reflection-latency-ms",
            "0",
            "--metric-budget",
            "24",
            "--train-size",
            "8",
            "--val-size",
            "8",
            "--test-size",
            "8",
            "--categories",
            "8",
            "--seeds",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    data = json.loads(output.read_text())
    runs = data["runs"]
    assert len(runs) == 4
    assert data["environment"]["git_commit"]
    with output.with_suffix(".csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(runs)
    for run in runs:
        assert run["actual_metric_calls"] == run["engine_metric_calls"]
        assert run["actual_metric_calls"] >= run["metric_budget"]
        assert run["budget_overshoot"] == run["actual_metric_calls"] - run["metric_budget"]
        assert run["proposals"] > 0
        assert 0 <= run["test_fixture_accuracy"] <= 1
        assert run["wall_seconds"] > 0
    for width in (1, 2):
        matching = [run for run in runs if run["width"] == width]
        keys = ("best_rules", "actual_metric_calls", "proposals", "accepted_candidates", "test_fixture_accuracy")
        assert {tuple(run[key] for key in keys) for run in matching} == {tuple(matching[0][key] for key in keys)}


def test_offline_benchmark_rejects_invalid_latency(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--eval-latency-ms", "nan", "--output", str(tmp_path / "unused.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "finite and nonnegative" in result.stderr
    assert not (tmp_path / "unused.json").exists()
