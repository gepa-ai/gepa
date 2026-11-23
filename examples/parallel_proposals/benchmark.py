# Copyright (c) 2026 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Offline latency sweep through GEPA's actual parallel proposal engine.

The evaluator and reflector are deterministic local fixtures with configurable
sleep latency. Their accuracy is a fixture check, not an LLM quality benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gepa
from gepa.core.adapter import EvaluationBatch, ProposalFn
from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionJob,
    ReflectionLM,
    ReflectionProposal,
)
from gepa.strategies.proposal_sampling import PxNSampling


@dataclass(frozen=True)
class Example:
    instance_id: int
    category: str
    label: int


@dataclass(frozen=True)
class RunConfig:
    workers: int
    parents: int
    mutations: int
    minibatch_size: int
    eval_latency_ms: float
    reflection_latency_ms: float
    metric_budget: int
    seed: int
    train_size: int
    val_size: int
    test_size: int
    categories: int


class QuietLogger:
    def log(self, message: str) -> None:
        pass


class LocalAdapter:
    """Evaluate category-to-label rules using one bounded worker pool."""

    propose_new_texts: ProposalFn | None = None

    def __init__(self, pool: ThreadPoolExecutor, latency_ms: float):
        self.pool = pool
        self.latency_ms = latency_ms
        self.metric_calls = 0
        self.wall_seconds = 0.0

    def _evaluate_one(self, item: tuple[dict[str, int], Example]) -> dict[str, Any]:
        rules, example = item
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        prediction = rules.get(example.category, 0)
        return {"category": example.category, "expected": example.label, "predicted": prediction}

    def evaluate(
        self,
        batch: list[Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], int]:
        return self.batch_evaluate([(candidate, batch)], capture_traces=capture_traces)[0]

    def batch_evaluate(
        self,
        items: list[tuple[dict[str, str], list[Example]]],
        *,
        capture_traces: bool = True,
    ) -> list[EvaluationBatch[dict[str, Any], int]]:
        started = time.perf_counter()
        tasks = []
        for candidate, batch in items:
            rules = json.loads(candidate["rules"])
            tasks.extend((rules, example) for example in batch)
        traces = list(self.pool.map(self._evaluate_one, tasks))
        self.metric_calls += len(tasks)
        self.wall_seconds += time.perf_counter() - started
        results = []
        offset = 0
        for _, batch in items:
            rows = traces[offset : offset + len(batch)]
            offset += len(batch)
            results.append(
                EvaluationBatch(
                    outputs=[row["predicted"] for row in rows],
                    scores=[float(row["predicted"] == row["expected"]) for row in rows],
                    trajectories=rows if capture_traces else None,
                    num_metric_calls=len(batch),
                )
            )
        return results

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], int],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert eval_batch.trajectories is not None
        return dict.fromkeys(components_to_update, eval_batch.trajectories)


class LocalReflection:
    """Correct one observed error; plan mutations before dispatch for determinism."""

    def __init__(self, pool: ThreadPoolExecutor, latency_ms: float, seed: int):
        self.pool = pool
        self.latency_ms = latency_ms
        self.rng = random.Random(seed)
        self.proposals = 0
        self.wall_seconds = 0.0

    def _finish(self, proposal: ReflectionProposal) -> tuple[ReflectionProposal, ReflectionLM]:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        return proposal, self

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ReflectionLM]:
        return self.reflect_many([(candidate, reflective_dataset, components_to_update)])[0]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ReflectionLM]]:
        started = time.perf_counter()
        proposals = []
        for candidate, dataset, _ in jobs:
            rules = json.loads(candidate["rules"])
            errors = [row for row in dataset["rules"] if row["predicted"] != row["expected"]]
            if errors:
                correction = self.rng.choice(errors)
                rules[correction["category"]] = correction["expected"]
            proposals.append(ReflectionProposal(new_texts={"rules": json.dumps(rules, sort_keys=True)}))
        results = list(self.pool.map(self._finish, proposals))
        self.proposals += len(jobs)
        self.wall_seconds += time.perf_counter() - started
        return results


def make_splits(config: RunConfig) -> tuple[list[Example], list[Example], list[Example]]:
    labels = [i % 2 for i in range(config.categories)]
    random.Random(config.seed).shuffle(labels)
    # Each split contains separate examples drawn from the same category rules.
    splits = []
    offset = 0
    for size in (config.train_size, config.val_size, config.test_size):
        split = [Example(offset + i, str(i % config.categories), labels[i % config.categories]) for i in range(size)]
        offset += size
        random.Random(config.seed + size).shuffle(split)
        splits.append(split)
    return splits[0], splits[1], splits[2]


def run_one(config: RunConfig) -> dict[str, Any]:
    train, val, test = make_splits(config)
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        adapter = LocalAdapter(pool, config.eval_latency_ms)
        reflector = LocalReflection(pool, config.reflection_latency_ms, config.seed)
        started = time.perf_counter()
        result = gepa.optimize(
            seed_candidate={"rules": "{}"},
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_strategy=reflector,
            sampling_strategy=PxNSampling(p=config.parents, n=config.mutations),
            reflection_minibatch_size=config.minibatch_size,
            max_metric_calls=config.metric_budget,
            skip_perfect_score=False,
            cache_evaluation=False,
            logger=QuietLogger(),
            display_progress_bar=False,
            track_best_outputs=False,
            seed=config.seed,
        )
        elapsed = time.perf_counter() - started
        # Held-out fixture scoring is outside the measured run and metric budget.
        best = result.candidates[result.best_idx]
        rules = json.loads(best["rules"])
        test_accuracy = sum(rules.get(example.category, 0) == example.label for example in test) / len(test)
        return {
            **asdict(config),
            "width": config.parents * config.mutations,
            "wall_seconds": elapsed,
            "evaluation_wall_seconds": adapter.wall_seconds,
            "reflection_wall_seconds": reflector.wall_seconds,
            "actual_metric_calls": adapter.metric_calls,
            "engine_metric_calls": result.total_metric_calls,
            "budget_overshoot": max(0, adapter.metric_calls - config.metric_budget),
            "metric_calls_per_second": adapter.metric_calls / elapsed,
            "proposals": reflector.proposals,
            "accepted_candidates": len(result.candidates) - 1,
            "validation_fixture_accuracy": result.best_score,
            "test_fixture_accuracy": test_accuracy,
            "best_rules": best["rules"],
        }


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def shape(value: str) -> tuple[int, int]:
    try:
        parents, mutations = value.lower().split("x")
        return positive_int(parents), positive_int(mutations)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected positive PxN, for example 2x4") from error


def environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
    return {
        "git_commit": commit.stdout.strip(),
        "working_tree_dirty": bool(dirty.stdout),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=positive_int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--pxn", type=shape, nargs="+", default=[(1, 1), (1, 4), (2, 4)])
    parser.add_argument("--minibatches", type=positive_int, nargs="+", default=[2, 8])
    parser.add_argument("--eval-latency-ms", type=nonnegative_float, nargs="+", default=[5.0, 20.0])
    parser.add_argument("--reflection-latency-ms", type=nonnegative_float, default=10.0)
    parser.add_argument("--metric-budget", type=positive_int, default=192)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-size", type=positive_int, default=64)
    parser.add_argument("--val-size", type=positive_int, default=32)
    parser.add_argument("--test-size", type=positive_int, default=64)
    parser.add_argument("--categories", type=positive_int, default=32)
    parser.add_argument("--output", type=Path, default=Path("parallel-proposals-results.json"))
    args = parser.parse_args()
    if max(args.minibatches) > args.train_size:
        parser.error("minibatch size must not exceed train size")
    if args.metric_budget <= args.val_size:
        parser.error("metric budget must exceed the initial validation size")

    configs = [
        RunConfig(
            workers,
            parents,
            mutations,
            minibatch,
            latency,
            args.reflection_latency_ms,
            args.metric_budget,
            seed,
            args.train_size,
            args.val_size,
            args.test_size,
            args.categories,
        )
        for workers, (parents, mutations), minibatch, latency, seed in itertools.product(
            args.workers, args.pxn, args.minibatches, args.eval_latency_ms, args.seeds
        )
    ]
    # Seeded shuffle reduces correlation between configuration and machine drift.
    random.Random(0).shuffle(configs)
    rows = []
    metadata = environment()
    for index, config in enumerate(configs, start=1):
        row = run_one(config)
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"environment": metadata, "runs": rows}, indent=2) + "\n")
        with args.output.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"[{index}/{len(configs)}] {config.parents}x{config.mutations}, workers={config.workers}, "
            f"minibatch={config.minibatch_size}: {row['wall_seconds']:.3f}s, "
            f"calls={row['actual_metric_calls']}, fixture accuracy={row['validation_fixture_accuracy']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
