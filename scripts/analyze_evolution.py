#!/usr/bin/env python3
"""
Analyze a TurboGEPA evolution JSON and report lineage depth and examples
of multi‑generation chains (parent → child → grandchild).

Usage:
  source .envrc && source .venv/bin/activate && \
  python scripts/analyze_evolution.py [--input .turbo_gepa/evolution/<run_id>.json]

If --input is omitted, the most recent evolution JSON under
.turbo_gepa/evolution is used.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _latest_json(root: Path) -> Path | None:
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(parent_children: Dict[str, List[str]]) -> Tuple[set[str], Dict[str, List[str]], Dict[str, int]]:
    children_all: set[str] = set()
    for kids in parent_children.values():
        children_all.update(kids)
    nodes = set(parent_children.keys()) | children_all
    indeg = defaultdict(int)
    for p, kids in parent_children.items():
        for c in kids:
            indeg[c] += 1
        indeg.setdefault(p, indeg.get(p, 0))
    return nodes, parent_children, indeg


def compute_depth(parent_children: Dict[str, List[str]]) -> Tuple[int, Dict[int, int], Dict[str, int]]:
    nodes, edges, indeg = build_graph(parent_children)
    # roots = nodes with indeg 0
    roots = [n for n in nodes if indeg.get(n, 0) == 0]
    depth = {n: 0 for n in roots}
    q = deque(roots)
    while q:
        n = q.popleft()
        for c in edges.get(n, []):
            d = depth.get(n, 0) + 1
            if d > depth.get(c, -1):
                depth[c] = d
                q.append(c)
    max_depth = max(depth.values()) if depth else 0
    by_level = defaultdict(int)
    for v in depth.values():
        by_level[v] += 1
    return max_depth, dict(sorted(by_level.items())), depth


def pick_example_chain(parent_children: Dict[str, List[str]], depth: Dict[str, int]) -> List[str]:
    # Find any node with depth >= 2 and reconstruct a short chain up to root
    target = None
    for n, d in depth.items():
        if d >= 2:
            target = n
            break
    if not target:
        return []
    # Build reverse edges to backtrack
    rev = defaultdict(list)
    for p, kids in parent_children.items():
        for c in kids:
            rev[c].append(p)
    # Walk back to a root
    chain = [target]
    cur = target
    while True:
        parents = rev.get(cur, [])
        if not parents:
            break
        # pick parent with highest depth
        par = max(parents, key=lambda x: depth.get(x, 0))
        chain.append(par)
        cur = par
        if depth.get(cur, 0) == 0:
            break
    chain.reverse()
    return chain


def summarize_lineage(lineage: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    total = len(lineage)
    scored = sum(1 for x in lineage if isinstance(x.get("quality"), (int, float)))
    full = sum(1 for x in lineage if abs(float(x.get("shard_fraction") or 0.0) - 1.0) < 1e-6)
    return total, scored, full


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze TurboGEPA evolution JSON for branching depth")
    ap.add_argument("--input", type=Path, default=None)
    args = ap.parse_args()

    src = args.input or _latest_json(Path(".turbo_gepa/evolution"))
    if not src or not src.exists():
        raise SystemExit("No evolution JSON found. Run a TurboGEPA job first.")

    payload = load_payload(src)
    run_id = str(payload.get("run_id") or src.stem)
    evo = payload.get("evolution_stats") or {}
    parent_children = evo.get("parent_children") or {}
    lineage = payload.get("lineage") or []

    max_depth, by_level, depth_map = compute_depth(parent_children)
    chain = pick_example_chain(parent_children, depth_map)
    total_nodes, scored_nodes, full_nodes = summarize_lineage(lineage)

    print(f"Run: {run_id}")
    print(f"Nodes: {len(set(list(parent_children.keys()) + [c for v in parent_children.values() for c in v]))}")
    print(f"Edges: {sum(len(v) for v in parent_children.values())}")
    print(f"Max depth: {max_depth}")
    print(f"Nodes by depth: {by_level}")
    print(f"Lineage entries: total={total_nodes}, with_quality={scored_nodes}, at_full_shard={full_nodes}")
    if chain:
        print("Sample chain (parent→child→…):")
        print("  " + " → ".join(chain))
    else:
        print("No grandchild chain found (depth < 2).")


if __name__ == "__main__":
    main()

