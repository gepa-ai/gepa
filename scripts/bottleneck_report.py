#!/usr/bin/env python3
"""
Aggregate bottleneck diagnostics across TurboGEPA runs.

Scans evolution summaries in .turbo_gepa/evolution and matches metrics files
in .turbo_gepa/metrics to compute simple heuristics that explain slowdowns:

- Final-rung tail latency (high p95, few/no stragglers on shard=1.00)
- Over-promotion to final rung (many promotions from rung 0, almost no pruning)
- Under-utilized concurrency (low peak concurrency, low eval throughput)

Usage:
  source .envrc && source .venv/bin/activate && \
  python scripts/bottleneck_report.py [--runs-dir .turbo_gepa/evolution] [--metrics-dir .turbo_gepa/metrics]

Optional: pass --run-id to limit to a single run.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunSummary:
    run_id: str
    evaluations: int
    edges: int
    parents: int
    children: int
    max_depth: int
    gen2: int
    best_full_quality: float
    stop_reason: Optional[str]


@dataclass
class Metrics:
    latency_mean: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    total_evals: Optional[int] = None
    throughput: Optional[float] = None
    peak_concurrency: Optional[int] = None
    shard_one_stragglers: Optional[int] = None
    shard_one_mean: Optional[float] = None
    promoted: Optional[int] = None
    pruned: Optional[int] = None
    completed: Optional[int] = None
    promotions_by_rung0: Optional[int] = None


def load_run_summary(runs_dir: Path, run_id: Optional[str]) -> list[RunSummary]:
    runs: list[RunSummary] = []
    if run_id:
        files = [runs_dir / f"{run_id}.summary.json"]
    else:
        files = sorted(runs_dir.glob("*.summary.json"))
    for path in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        runs.append(
            RunSummary(
                run_id=data.get("run_id") or path.stem.split(".")[0],
                evaluations=int(data.get("evaluations", 0)),
                edges=int(data.get("evolution_edges", 0)),
                parents=int(data.get("unique_parents", 0)),
                children=int(data.get("unique_children", 0)),
                max_depth=int(data.get("max_depth", 0)),
                gen2=int(data.get("gen2_count", 0)),
                best_full_quality=float(data.get("best_full_quality", 0.0)),
                stop_reason=data.get("stop_reason"),
            )
        )
    return runs


def index_metrics(metrics_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in metrics_dir.glob("metrics_*.txt"):
        try:
            with p.open("r", encoding="utf-8") as f:
                head = f.read(256)
        except Exception:
            continue
        m = re.search(r"Run ID:\s*([a-f0-9]+)", head)
        if m:
            idx[m.group(1)] = p
    return idx


def parse_metrics_text(text: str) -> Metrics:
    m = Metrics()
    # Latency line: "Latency: mean=4.54s, p50=2.31s, p95=11.87s"
    lat = re.search(r"Latency:\s*mean=([0-9.]+)s,\s*p50=([0-9.]+)s,\s*p95=([0-9.]+)s", text)
    if lat:
        m.latency_mean = float(lat.group(1))
        m.latency_p50 = float(lat.group(2))
        m.latency_p95 = float(lat.group(3))
    # Evaluation throughput: "Total evaluations: 94" and "Throughput: 0.21 evals/sec"
    te = re.search(r"Total evaluations:\s*([0-9]+)", text)
    if te:
        m.total_evals = int(te.group(1))
    thr = re.search(r"Throughput:\s*([0-9.]+) evals/sec", text)
    if thr:
        m.throughput = float(thr.group(1))
    pc = re.search(r"Peak concurrency:\s*([0-9]+)", text)
    if pc:
        m.peak_concurrency = int(pc.group(1))
    # Shard 1.00 line: "shard=1.00: coverage=100.0%, stragglers=0, ... mean_duration=4.8s"
    s1 = re.search(r"shard=1\.00:\s*coverage=[0-9.]+%,\s*stragglers=([0-9]+).+?mean_duration=([0-9.]+)s", text)
    if s1:
        m.shard_one_stragglers = int(s1.group(1))
        m.shard_one_mean = float(s1.group(2))
    # Scheduler Decisions section
    pr = re.search(r"Promoted:\s*([0-9]+)", text)
    if pr:
        m.promoted = int(pr.group(1))
    pruned = re.search(r"Pruned:\s*([0-9]+)", text)
    if pruned:
        m.pruned = int(pruned.group(1))
    comp = re.search(r"Completed:\s*([0-9]+)", text)
    if comp:
        m.completed = int(comp.group(1))
    # Promotions by rung: {0: 9, 1: 2}
    pbr = re.search(r"Promotions by rung:\s*\{([^}]+)\}", text)
    if pbr:
        # crude parse
        try:
            payload = "{" + pbr.group(1) + "}"
            mapping = json.loads(payload.replace("'", '"'))
            m.promotions_by_rung0 = int(mapping.get("0", 0))
        except Exception:
            pass
    return m


def diagnose(r: RunSummary, m: Metrics) -> list[str]:
    findings: list[str] = []
    # Final-rung tail latency
    if (m.latency_p95 and m.latency_p95 > 30.0) and (m.shard_one_stragglers is not None and m.shard_one_stragglers == 0):
        findings.append("final-rung tail latency (high p95, no stragglers)")
    # Over-promotion to final rung
    if (m.promoted and m.promoted > 0) and (m.pruned is not None and m.pruned == 0):
        if m.promotions_by_rung0 is not None and m.promotions_by_rung0 >= max(3, int(0.5 * m.promoted)):
            findings.append("over-promotion at rung 0 (no pruning)")
    # Under-utilized concurrency
    if (m.peak_concurrency and m.peak_concurrency >= 4) and (m.throughput and m.throughput < max(0.2, 0.02 * m.peak_concurrency)):
        findings.append("under-utilized concurrency (low evals/sec vs peak)")
    # Depth check
    if r.max_depth < 2:
        findings.append("shallow lineage (no grandchildren)")
    return findings or ["no clear bottleneck detected"]


def main() -> None:
    ap = argparse.ArgumentParser(description="TurboGEPA bottleneck report across runs")
    ap.add_argument("--runs-dir", type=Path, default=Path(".turbo_gepa/evolution"))
    ap.add_argument("--metrics-dir", type=Path, default=Path(".turbo_gepa/metrics"))
    ap.add_argument("--run-id")
    args = ap.parse_args()

    runs = load_run_summary(args.runs_dir, args.run_id)
    if not runs:
        raise SystemExit("No runs found. Make sure summary JSONs exist.")

    mindex = index_metrics(args.metrics_dir)
    print("Bottleneck Report\n================\n")
    for r in runs:
        mp = mindex.get(r.run_id)
        metrics = Metrics()
        if mp and mp.exists():
            text = mp.read_text(encoding="utf-8")
            metrics = parse_metrics_text(text)
        findings = diagnose(r, metrics)
        print(f"Run {r.run_id}:")
        print(f"  evals={r.evaluations} edges={r.edges} parents={r.parents} children={r.children} depth={r.max_depth} gen2={r.gen2}")
        if metrics.latency_mean is not None:
            print(f"  latency: mean={metrics.latency_mean:.2f}s p50={metrics.latency_p50:.2f}s p95={metrics.latency_p95:.2f}s")
        if metrics.shard_one_mean is not None:
            print(f"  shard=1.00: stragglers={metrics.shard_one_stragglers} mean_duration={metrics.shard_one_mean:.1f}s")
        if metrics.peak_concurrency is not None:
            print(f"  peak_concurrency={metrics.peak_concurrency} throughput={metrics.throughput or 0.0:.2f} evals/sec")
        if metrics.promoted is not None:
            print(f"  promoted={metrics.promoted} pruned={metrics.pruned} completed={metrics.completed} rung0_promotions={metrics.promotions_by_rung0}")
        print("  findings:")
        for f in findings:
            print(f"    - {f}")
        print("")


if __name__ == "__main__":
    main()

