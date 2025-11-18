#!/usr/bin/env python3
"""
Lightweight helper to inspect a previous TurboGEPA run.

Usage (from repo root):
    source .envrc && source .venv/bin/activate
    python scripts/analyze_turbo_run.py              # analyze latest metrics file
    python scripts/analyze_turbo_run.py --run-id XYZ # pick metrics for a specific run_id

It scans `.turbo_gepa/metrics/metrics_*.txt`, extracts key fields such as
run_id, time_to_target, evaluations, LLM call counts, mutation counts, and
best observed shard/quality, and prints a compact summary that is easy to
parse or eyeball.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


METRICS_DIR = Path(".turbo_gepa") / "metrics"


@dataclass
class RunSummary:
    run_id: str | None = None
    time_to_target: float | None = None
    target_shard: float | None = None
    baseline: float | None = None
    target_quality: float | None = None
    total_evaluations: int | None = None
    peak_concurrency: int | None = None
    total_llm_calls: int | None = None
    task_calls: int | None = None
    reflection_calls: int | None = None
    spec_calls: int | None = None
    total_mutations: int | None = None
    best_quality: float | None = None
    best_shard: float | None = None
    wall_time: float | None = None


def _latest_metrics_file() -> Optional[Path]:
    if not METRICS_DIR.exists():
        return None
    candidates = list(METRICS_DIR.glob("metrics_*.txt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _match_run_file(run_id: str) -> Optional[Path]:
    if not METRICS_DIR.exists():
        return None
    # Try exact names first
    explicit = METRICS_DIR / f"metrics_{run_id}.txt"
    if explicit.exists():
        return explicit
    explicit_latest = METRICS_DIR / f"metrics_{run_id}_latest.txt"
    if explicit_latest.exists():
        return explicit_latest
    # Fallback: any metrics file mentioning the run_id in its name
    candidates = [p for p in METRICS_DIR.glob("metrics_*.txt") if run_id in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_metrics(path: Path) -> RunSummary:
    summary = RunSummary()
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    for ln in lines:
        if ln.startswith("Run ID:"):
            summary.run_id = ln.split(":", 1)[1].strip()
        elif ln.startswith("Total calls:"):
            # "Total calls: 143"
            try:
                summary.total_llm_calls = int(ln.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif "- Task evaluations:" in ln:
            m = re.search(r"- Task evaluations:\s*(\d+)", ln)
            if m:
                summary.task_calls = int(m.group(1))
        elif "- Reflections:" in ln:
            m = re.search(r"- Reflections:\s*(\d+)", ln)
            if m:
                summary.reflection_calls = int(m.group(1))
        elif "- Spec induction:" in ln:
            m = re.search(r"- Spec induction:\s*(\d+)", ln)
            if m:
                summary.spec_calls = int(m.group(1))
        elif ln.startswith("Total evaluations:"):
            try:
                summary.total_evaluations = int(ln.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif ln.startswith("Peak concurrency:"):
            try:
                summary.peak_concurrency = int(ln.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif ln.startswith("Time:") and "s" in ln and summary.wall_time is None:
            # "Time: 85.4s"
            try:
                val = ln.split(":", 1)[1].strip()
                summary.wall_time = float(val.replace("s", ""))
            except ValueError:
                pass
        elif "Time to target:" in ln:
            m = re.search(r"Time to target:\s*([0-9.]+)s", ln)
            if m:
                summary.time_to_target = float(m.group(1))
        elif ln.startswith("Target shard:"):
            try:
                summary.target_shard = float(ln.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif "Baseline=" in ln and "Target=" in ln:
            # "Baseline=0.295 → Target=0.733"
            m = re.search(r"Baseline=([0-9.]+)", ln)
            if m:
                try:
                    summary.baseline = float(m.group(1))
                except ValueError:
                    pass
            m2 = re.search(r"Target=([0-9.]+)", ln)
            if m2:
                try:
                    summary.target_quality = float(m2.group(1))
                except ValueError:
                    pass
        elif ln.startswith("Total mutations:"):
            try:
                summary.total_mutations = int(ln.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif ln.startswith("Best observed:"):
            # "Best observed: Quality=1.000 @ shard 1.00"
            mq = re.search(r"Quality=([0-9.]+)", ln)
            ms = re.search(r"shard\s+([0-9.]+)", ln)
            if mq:
                try:
                    summary.best_quality = float(mq.group(1))
                except ValueError:
                    pass
            if ms:
                try:
                    summary.best_shard = float(ms.group(1))
                except ValueError:
                    pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a previous TurboGEPA run from .turbo_gepa/metrics.")
    parser.add_argument(
        "--run-id",
        help="Run ID to match (e.g. 1639cd0e-solo). Defaults to latest metrics_* file.",
    )
    args = parser.parse_args()

    if args.run_id:
        path = _match_run_file(args.run_id)
        if path is None:
            raise SystemExit(f"No metrics file found for run_id={args.run_id!r} in {METRICS_DIR}")
    else:
        path = _latest_metrics_file()
        if path is None:
            raise SystemExit(f"No metrics_*.txt files found in {METRICS_DIR}")

    summary = _parse_metrics(path)

    print(f"Metrics file: {path}")
    print(f"Run ID:            {summary.run_id}")
    print(f"Wall time (s):     {summary.wall_time}")
    print(f"Time to target (s):{summary.time_to_target}")
    print(f"Target shard:      {summary.target_shard}")
    print(f"Baseline quality:  {summary.baseline}")
    print(f"Target quality:    {summary.target_quality}")
    print(f"Total evaluations: {summary.total_evaluations}")
    print(f"Peak concurrency:  {summary.peak_concurrency}")
    print(f"LLM calls (total): {summary.total_llm_calls}")
    print(f"  Task eval calls: {summary.task_calls}")
    print(f"  Reflections:     {summary.reflection_calls}")
    print(f"  Spec induction:  {summary.spec_calls}")
    print(f"Total mutations:   {summary.total_mutations}")
    print(f"Best observed q:   {summary.best_quality} @ shard {summary.best_shard}")


if __name__ == "__main__":
    main()

