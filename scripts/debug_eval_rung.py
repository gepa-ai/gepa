#!/usr/bin/env python3
"""
Debug a single rung evaluation deterministically to verify scoring.

This script replicates the seed evaluation on a specific rung fraction
(e.g., 0.40 or 0.50) using the same canonical sampler as TurboGEPA, then
prints a per-example breakdown of expected vs extracted answers and the
computed quality. Use this to confirm or disprove "100% on 40%" claims.

Example:
  source .envrc && source .venv/bin/activate && \
  python scripts/debug_eval_rung.py \
    --dataset-size 30 \
    --rung 0.40 \
    --task-lm openrouter/openai/gpt-oss-20b:nitro \
    --seed-prompt "You are a helpful assistant."
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, List

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.sampler import InstanceSampler


def load_aime_subset(limit: int) -> list[dict]:
    train, _val, _ = gepa.examples.aime.init_dataset()
    return train[:limit]


def build_adapter(dataset: list[dict], task_lm: str) -> DefaultAdapter:
    data = [
        DefaultDataInst(
            input=ex["input"],
            answer=ex.get("answer", ""),
            additional_context=ex.get("additional_context"),
            id=f"aime_{i}",
        )
        for i, ex in enumerate(dataset)
    ]
    cfg = Config()
    cfg.eval_concurrency = 8
    adapter = DefaultAdapter(dataset=data, task_lm=task_lm, reflection_lm=task_lm, config=cfg, auto_config=False)
    return adapter


async def eval_seed_on_rung(adapter: DefaultAdapter, seed_prompt: str, rung: float, seed: int = 42) -> None:
    ids = [d.id for d in adapter.dataset]
    sampler = InstanceSampler(ids, seed=seed)
    size = max(1, int(round(len(ids) * rung)))
    shard = sampler.sample_canonical(rung, size, island_id=0)

    # Evaluate one example at a time (serially for clarity)
    from turbo_gepa.interfaces import Candidate

    cand = Candidate(text=seed_prompt, meta={})
    total = 0
    correct = 0
    rows: list[tuple[str, float]] = []
    print(f"Evaluating {len(shard)} examples at rung {rung:.0%}...")
    for ex_id in shard:
        metrics = await adapter._task_runner(cand, ex_id)
        q = float(metrics.get("quality", 0.0))
        rows.append((ex_id, q))
        total += 1
        correct += 1 if q >= 0.999 else 0
        expected = metrics.get("expected_answer")
        output = metrics.get("output")
        print(f" - {ex_id}: q={q:.3f} expected={expected!r} output_snip={(output or '')[:80]!r}")
    acc = correct / max(1, total)
    print(f"\nSummary: correct={correct}/{total} ({acc:.1%}) at rung {rung:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug rung evaluation scoring")
    ap.add_argument("--dataset-size", type=int, default=30)
    ap.add_argument("--rung", type=float, default=0.40)
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--seed-prompt", default="You are a helpful assistant.")
    args = ap.parse_args()

    ds = load_aime_subset(args.dataset_size)
    adapter = build_adapter(ds, args.task_lm)
    try:
        asyncio.run(eval_seed_on_rung(adapter, args.seed_prompt, args.rung))
    finally:
        try:
            asyncio.run(adapter.aclose())
        except Exception:
            pass


if __name__ == "__main__":
    main()

