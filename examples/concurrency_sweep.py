#!/usr/bin/env python3
"""
Synthetic concurrency sweep for TurboGEPA.

Runs a small matrix of latency profiles × eval concurrency × concurrency
strategy (static vs adaptive) using synthetic task/reflector stubs so we can
measure throughput without hitting real LLMs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import types

from turbo_gepa.adapters import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


LatencyFn = Callable[[], float]


def latency_nice() -> float:
    return max(0.05, random.gauss(2.0, 0.4))


def latency_tail_moderate() -> float:
    return 2.0 if random.random() < 0.9 else random.uniform(10, 20)


def latency_tail_bad() -> float:
    return 2.0 if random.random() < 0.9 else random.uniform(40, 80)


LATENCY_PROFILES: Dict[str, LatencyFn] = {
    "nice": latency_nice,
    "moderate": latency_tail_moderate,
    "bad": latency_tail_bad,
}


def _latest_metrics(metrics_dir: Path) -> Path | None:
    """Return the most recent metrics_*.txt file under metrics_dir, if any."""
    if not metrics_dir.exists():
        return None
    candidates = list(metrics_dir.glob("metrics_*.txt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_metrics(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Time to target:"):
            try:
                data["time_to_target"] = float(line.split(":")[1].strip().replace("s", ""))
            except ValueError:
                data["time_to_target"] = None
        elif line.startswith("Time:"):
            try:
                data["wall_time"] = float(line.split(":")[1].strip().replace("s", ""))
            except ValueError:
                data["wall_time"] = None
        elif line.startswith("Total evaluations:"):
            try:
                data["evaluations"] = int(line.split(":")[1].strip())
            except ValueError:
                data["evaluations"] = None
        elif line.startswith("Throughput:"):
            try:
                data["throughput"] = float(line.split(":")[1].strip().split()[0])
            except ValueError:
                data["throughput"] = None
        elif line.startswith("Latency:"):
            # Example: Latency: mean=9.58s, p50=5.43s, p95=26.53s
            parts = line.split(",")
            for part in parts:
                if "mean=" in part:
                    data["latency_mean"] = float(part.split("mean=")[1].replace("s", ""))
                if "p50=" in part:
                    data["latency_p50"] = float(part.split("p50=")[1].replace("s", ""))
                if "p95=" in part:
                    data["latency_p95"] = float(part.split("p95=")[1].replace("s", ""))
        elif line.startswith("Peak concurrency:"):
            try:
                data["peak_concurrency"] = int(line.split(":")[1].strip())
            except ValueError:
                data["peak_concurrency"] = None
        elif line.startswith("Run ID:"):
            data["run_id"] = line.split(":", 1)[1].strip()
    return data


def _build_dataset(size: int) -> List[DefaultDataInst]:
    dataset: List[DefaultDataInst] = []
    for idx in range(size):
        dataset.append(
            DefaultDataInst(
                input=f"Solve problem {idx}",
                answer="42",
                additional_context=None,
                id=f"synthetic_{idx}",
            )
        )
    return dataset


def _patch_adapter(adapter: DefaultAdapter, latency_fn: LatencyFn) -> None:
    async def fake_task_runner(self, candidate, example_id):
        await asyncio.sleep(latency_fn())
        base = 0.6 + 0.05 * random.randint(0, 3)
        jitter = random.uniform(-0.05, 0.05)
        quality = max(0.0, min(1.0, base + jitter))
        return {"quality": quality, "tokens": 50.0}

    async def fake_reflection_runner(parent_contexts, num_mutations, _examples=None):
        mutations: List[str] = []
        for ctx in parent_contexts:
            candidate = ctx.get("candidate")
            base_text = candidate.text if candidate else "candidate"
            for _ in range(num_mutations):
                mutations.append(f"{base_text} :: mut {random.randint(1, 1000)}")
        if not parent_contexts:
            for _ in range(num_mutations):
                mutations.append(f"seed :: mut {random.randint(1, 1000)}")
        return mutations[:num_mutations]

    adapter._task_runner = types.MethodType(fake_task_runner, adapter)
    adapter.mutator.batch_reflection_runner = fake_reflection_runner  # type: ignore[attr-defined]
    adapter.mutator.spec_induction_runner = None  # type: ignore[attr-defined]


def run_experiment(latency_name: str, strategy: str, eval_conc: int, args: argparse.Namespace) -> Dict[str, object]:
    latency_fn = LATENCY_PROFILES[latency_name]
    dataset = _build_dataset(args.dataset_size)
    with tempfile.TemporaryDirectory(prefix=f"concurrency_{latency_name}_{strategy}_{eval_conc}_") as tmp:
        tmp_path = Path(tmp)
        cache_dir = tmp_path / "cache"
        log_dir = tmp_path / "logs"
        control_dir = tmp_path / "control"
        for path in (cache_dir, log_dir, control_dir):
            path.mkdir(parents=True, exist_ok=True)

        config = Config(
            n_islands=1,
            eval_concurrency=eval_conc,
            shards=(0.33, 1.0),
            queue_limit=max(32, eval_conc * 4),
            max_mutations_per_round=max(8, eval_conc),
            target_quality=args.target_quality,
            target_shard_fraction=1.0,
            verification_speed_bias=args.verification_speed_bias,
            cache_path=str(cache_dir),
            log_path=str(log_dir),
            control_dir=str(control_dir),
            auto_scale_eval_concurrency=(strategy == "adaptive"),
        )
        config.max_optimization_time_seconds = args.max_runtime
        adapter = DefaultAdapter(
            dataset=dataset,
            task_lm="synthetic/task",
            reflection_lm="synthetic/reflection",
            config=config,
            auto_config=False,
        )
        _patch_adapter(adapter, latency_fn)
        try:
            adapter.optimize(
                seeds=["You are a synthetic assistant."],
                max_rounds=None,
                max_evaluations=None,
                display_progress=False,
            )
        except Exception as exc:
            return {
                "latency": latency_name,
                "strategy": strategy,
                "eval_concurrency": eval_conc,
                "error": str(exc),
            }

        # Metrics are written by the orchestrator to ".turbo_gepa/metrics"
        # under the current working directory.
        metrics_dir = Path.cwd() / ".turbo_gepa" / "metrics"
        latest = _latest_metrics(metrics_dir)
        if latest is None:
            return {
                "latency": latency_name,
                "strategy": strategy,
                "eval_concurrency": eval_conc,
                "error": "missing metrics",
            }
        data = _parse_metrics(latest)
        data.update(
            {
                "latency": latency_name,
                "strategy": strategy,
                "eval_concurrency": eval_conc,
                "metrics_path": str(latest),
            }
        )
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic concurrency benchmark sweep.")
    parser.add_argument("--dataset-size", type=int, default=15)
    parser.add_argument("--target-quality", type=float, default=0.75)
    parser.add_argument("--verification-speed-bias", type=float, default=0.8)
    parser.add_argument("--max-runtime", type=int, default=240)
    parser.add_argument("--eval-concurrency", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--latency-profiles", nargs="+", default=list(LATENCY_PROFILES.keys()))
    parser.add_argument("--strategies", nargs="+", default=["static", "adaptive"])
    args = parser.parse_args()

    results: List[Dict[str, object]] = []
    for latency_name in args.latency_profiles:
        for strategy in args.strategies:
            for conc in args.eval_concurrency:
                print(f"Running {latency_name} / {strategy} / eval_concurrency={conc} ...")
                data = run_experiment(latency_name, strategy, conc, args)
                results.append(data)

    print("\n=== Summary ===")
    header = [
        "latency",
        "strategy",
        "eval_conc",
        "time_to_target",
        "wall_time",
        "evals",
        "lat_p95",
        "metrics_path",
    ]
    for row in results:
        latency = row.get("latency", "n/a")
        strategy = row.get("strategy", "n/a")
        eval_conc = row.get("eval_concurrency", "n/a")
        time_to_target = row.get("time_to_target", "n/a")
        wall_time = row.get("wall_time", "n/a")
        evaluations = row.get("evaluations", "n/a")
        lat_p95 = row.get("latency_p95", "n/a")
        metrics_path = row.get("metrics_path", row.get("error", "n/a"))
        print(
            f"{str(latency):>8} | {str(strategy):>8} | {str(eval_conc):>4} | "
            f"{str(time_to_target):>10} | {str(wall_time):>8} | "
            f"{str(evaluations):>5} | {str(lat_p95):>6} | {metrics_path}"
        )

    output = Path("concurrency_results.json")
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved raw results to {output.resolve()}")


if __name__ == "__main__":
    main()
