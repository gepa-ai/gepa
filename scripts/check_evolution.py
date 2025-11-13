#!/usr/bin/env python3
"""
Quick verification for a TurboGEPA run.

Reads .turbo_gepa/evolution/<run_id>.summary.json (or falls back to the
full JSON) and prints simple OK/FAIL checks: grandchildren present, edges,
promotions, and best full‑shard quality.

Usage:
  source .envrc && source .venv/bin/activate && \
  python scripts/check_evolution.py [--run-id <id>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(run_id: str | None) -> tuple[dict, Path | None]:
    evo_dir = Path(".turbo_gepa/evolution")
    evo_dir.mkdir(parents=True, exist_ok=True)
    if run_id:
        summary = evo_dir / f"{run_id}.summary.json"
        if summary.exists():
            return json.loads(summary.read_text(encoding="utf-8")), summary
        # fallback to full json
        full = evo_dir / f"{run_id}.json"
        if full.exists():
            data = json.loads(full.read_text(encoding="utf-8"))
            return _summarize_from_full(data), None
        raise SystemExit(f"No artifacts for run_id={run_id}")

    # No run_id provided: use current pointer
    cur = evo_dir / "current.json"
    if not cur.exists():
        raise SystemExit("No current.json. Run TurboGEPA first.")
    rid = json.loads(cur.read_text()).get("run_id")
    return load_summary(rid)


def _summarize_from_full(payload: dict) -> dict:
    evo = payload.get("evolution_stats") or {}
    lineage = payload.get("lineage") or []
    parent_children = evo.get("parent_children") or {}
    # depth
    nodes = set(parent_children.keys()) | {c for kids in parent_children.values() for c in kids}
    indeg = {n: 0 for n in nodes}
    for p, kids in parent_children.items():
        for c in kids:
            indeg[c] = indeg.get(c, 0) + 1
    roots = [n for n in nodes if indeg.get(n, 0) == 0]
    from collections import deque
    depth = {n: 0 for n in roots}
    q = deque(roots)
    while q:
        n = q.popleft()
        for c in parent_children.get(n, []):
            d = depth.get(n, 0) + 1
            if d > depth.get(c, -1):
                depth[c] = d
                q.append(c)
    max_depth = max(depth.values()) if depth else 0
    gen2 = sum(1 for x in lineage if int(x.get("generation", 0)) >= 2)
    return {
        "run_id": payload.get("run_id"),
        "evaluations": int(payload.get("run_metadata", {}).get("evaluations", 0)),
        "unique_parents": int(len(parent_children)),
        "unique_children": int(evo.get("unique_children", 0)),
        "evolution_edges": int(evo.get("evolution_edges", 0)),
        "mutations_generated": int(evo.get("mutations_generated", 0)),
        "mutations_promoted": int(evo.get("mutations_promoted", 0)),
        "max_depth": int(max_depth),
        "has_grandchild": bool(max_depth >= 2 or gen2 > 0),
        "gen2_count": int(gen2),
        "best_full_quality": float(payload.get("run_metadata", {}).get("best_quality", 0.0)),
        "full_shard_fraction": float(payload.get("run_metadata", {}).get("best_quality_shard", 1.0)),
        "stop_reason": payload.get("run_metadata", {}).get("stop_reason"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify TurboGEPA evolution summary")
    ap.add_argument("--run-id")
    args = ap.parse_args()

    summary, path = load_summary(args.run_id)
    rid = summary.get("run_id")
    print(f"Run: {rid}")
    if path:
        print(f"Summary file: {path}")

    # Simple checks
    checks = [
        (summary.get("evolution_edges", 0) > 0, "edges>0"),
        (summary.get("mutations_promoted", 0) >= 1, "promotions>=1"),
        (summary.get("max_depth", 0) >= 2 or summary.get("gen2_count", 0) > 0, "grandchildren present"),
    ]
    for ok, name in checks:
        print(("OK" if ok else "FAIL"), name)

    print("Stats:")
    for k in [
        "evaluations",
        "unique_parents",
        "unique_children",
        "evolution_edges",
        "mutations_generated",
        "mutations_promoted",
        "max_depth",
        "gen2_count",
        "best_full_quality",
        "full_shard_fraction",
        "stop_reason",
    ]:
        print(f"  {k}: {summary.get(k)}")


if __name__ == "__main__":
    main()

