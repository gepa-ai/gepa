#!/usr/bin/env python3
"""
Run the TurboGEPA regression matrix described in the stabilization plan.

Each scenario executes a short optimization run with a different configuration
to surface bottlenecks such as promotion stalls, queue saturation, resume
failures, or straggler handling problems. All scenarios use the approved models:

    task_lm="openrouter/openai/gpt-oss-20b:nitro"
    reflection_lm="openrouter/x-ai/grok-4-fast"

Usage:
    python scripts/run_evolution_matrix.py --all
    python scripts/run_evolution_matrix.py --scenario seed-baseline high-concurrency

Set OPENROUTER_API_KEY (and any other provider credentials) in the environment
before running. The script writes caches/logs under the provided output folder
(default: ./regression_runs/<scenario-name>).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import shutil
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.interfaces import Candidate
from turbo_gepa.logging.logger import LogLevel, StdOutLogger

TASK_LM = "openrouter/openai/gpt-oss-20b:nitro"
REFLECTION_LM = "openrouter/x-ai/grok-4-fast"


BASE_PROMPTS: list[tuple[str, str]] = [
    ("Solve: 3x + 4 = 19. Provide steps.", "5"),
    ("Compute the derivative of x^2 + 3x.", "2x + 3"),
    ("What is the capital of France?", "Paris"),
    ("Explain the water cycle in two sentences.", "Condensation..."),
    ("Add 17 and 29.", "46"),
    ("Fact-check: The moon is made of cheese.", "False"),
    ("Translate 'hello world' to Spanish.", "hola mundo"),
    ("Simplify the fraction 18/24.", "3/4"),
]


def build_dataset(size: int) -> list[DefaultDataInst]:
    examples: list[DefaultDataInst] = []
    for i in range(size):
        prompt, answer = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        examples.append(DefaultDataInst(id=f"ex{i}", input=prompt, answer=answer))
    return examples


DEFAULT_SEEDS = [
    "You are a meticulous problem solver. Think step by step and justify every decision.",
    "Adopt the role of a careful tutor: explain reasoning first, then give the final answer.",
]


@dataclass(slots=True)
class StragglerProfile:
    probability: float
    delay_seconds: float


@dataclass(slots=True)
class Scenario:
    name: str
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    seeds: list[str] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    dataset_size: int = 16
    max_rounds: int | None = 4
    max_evaluations: int | None = None
    resume_drill: bool = False
    straggler_profile: StragglerProfile | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        name="seed-baseline",
        description="No mutations – validates pure promotion behavior and final-rung throughput.",
        config_overrides={
            "max_mutations_per_round": 0,
            "eval_concurrency": 4,
            "max_final_shard_inflight": 2,
            "shards": (0.5, 1.0),
        },
        dataset_size=12,
        max_rounds=2,
        max_evaluations=24,
    ),
    Scenario(
        name="high-concurrency",
        description="Stress final-rung saturation with many evaluators.",
        config_overrides={
            "eval_concurrency": 32,
            "max_final_shard_inflight": 3,
            "max_mutations_per_round": 32,
            "queue_limit": 96,
            "shards": (0.2, 0.6, 1.0),
        },
        dataset_size=48,
        max_rounds=4,
        max_evaluations=200,
    ),
    Scenario(
        name="mixed-parent-pool",
        description="Ensures rung-0 parents remain in rotation until higher-rung quorum is met.",
        config_overrides={
            "shards": (0.1, 0.6, 1.0),
            "max_mutations_per_round": 24,
            "eval_concurrency": 12,
        },
        dataset_size=24,
        max_rounds=5,
    ),
    Scenario(
        name="neg-cost-objective",
        description="Smoke test for promote_objective='neg_cost'.",
        config_overrides={
            "promote_objective": "neg_cost",
            "shards": (0.5, 1.0),
            "max_mutations_per_round": 8,
            "eval_concurrency": 6,
        },
        dataset_size=16,
        max_rounds=3,
    ),
    Scenario(
        name="resume-drill",
        description="Interrupt mid-run, then resume from cache.",
        config_overrides={
            "shards": (0.2, 0.6, 1.0),
            "eval_concurrency": 10,
            "max_mutations_per_round": 10,
        },
        dataset_size=20,
        max_rounds=6,
        max_evaluations=120,
        resume_drill=True,
    ),
    Scenario(
        name="multi-island-migration",
        description="Two islands exchanging elites to verify migration safety.",
        config_overrides={
            "n_islands": 2,
            "migration_period": 2,
            "migration_k": 2,
            "eval_concurrency": 8,
            "max_mutations_per_round": 12,
        },
        dataset_size=24,
        max_rounds=5,
    ),
    Scenario(
        name="straggler-tail",
        description="Inject artificial slowness into some evaluations to test straggler replay.",
        config_overrides={
            "eval_concurrency": 12,
            "max_final_shard_inflight": 2,
            "max_mutations_per_round": 14,
        },
        dataset_size=32,
        max_rounds=4,
        straggler_profile=StragglerProfile(probability=0.15, delay_seconds=20.0),
    ),
]


async def run_scenario(scenario: Scenario, output_root: Path) -> None:
    run_dir = output_root / scenario.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    config = Config()
    config.cache_path = str(run_dir / "cache")
    config.log_path = str(run_dir / "logs")
    config.latest_results_limit = 2048
    for key, value in scenario.config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"Config has no attribute '{key}' (scenario {scenario.name})")

    dataset = build_dataset(scenario.dataset_size)
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        config=config,
    )

    if scenario.straggler_profile:
        _inject_straggler_latency(adapter, scenario.straggler_profile)

    seeds = scenario.seeds or DEFAULT_SEEDS

    async def _run(adapter_instance: DefaultAdapter) -> dict[str, Any]:
        return await adapter_instance.optimize_async(
            seeds=seeds,
            max_rounds=scenario.max_rounds,
            max_evaluations=scenario.max_evaluations,
            display_progress=True,
        )

    start = time.time()
    if scenario.resume_drill:
        # First pass builds cache/state.
        await _run(adapter)
        # Second adapter loads the same cache and should resume automatically.
        adapter_resume = DefaultAdapter(
            dataset=dataset,
            task_lm=TASK_LM,
            reflection_lm=REFLECTION_LM,
            config=config,
        )
        result = await _run(adapter_resume)
    else:
        result = await _run(adapter)
    elapsed = time.time() - start

    best_entry = _get_best_entry(result["pareto_entries"])
    best_quality = best_entry.result.objectives.get(config.promote_objective, 0.0) if best_entry else 0.0
    summary = (
        f"[{scenario.name}] completed in {elapsed:.1f}s | "
        f"pareto={len(result['pareto_entries'])} "
        f"best_{config.promote_objective}={best_quality:.4f}"
    )
    adapter.logger.log(summary, LogLevel.WARNING)


def _inject_straggler_latency(adapter: DefaultAdapter, profile: StragglerProfile) -> None:
    original_runner = adapter._task_runner

    async def wrapped(self, candidate: Candidate, example_id: str):
        if random.random() < profile.probability:
            await asyncio.sleep(profile.delay_seconds)
        return await original_runner(candidate, example_id)

    adapter._task_runner = types.MethodType(wrapped, adapter)


def _get_best_entry(entries: Iterable[Any]):
    best = None
    best_quality = float("-inf")
    for entry in entries:
        quality = entry.result.objectives.get("quality", 0.0)
        if quality > best_quality:
            best_quality = quality
            best = entry
    return best


async def main(selected: list[str], output_root: Path) -> None:
    scenarios = [s for s in SCENARIOS if not selected or s.name in selected]
    if not scenarios:
        raise SystemExit("No scenarios selected.")

    for scenario in scenarios:
        print(f"\n=== Running scenario: {scenario.name} ===\n{scenario.description}\n")
        try:
            await run_scenario(scenario, output_root)
        except Exception as exc:  # pragma: no cover - harness script
            print(f"⚠️  Scenario {scenario.name} failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboGEPA regression matrix runner.")
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        metavar="NAME",
        help="Run only the named scenario (can be supplied multiple times).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every scenario in the matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./regression_runs"),
        help="Directory where caches/logs for each scenario will be created.",
    )
    args = parser.parse_args()
    if not args.all and not args.scenarios:
        parser.error("Specify --all or one or more --scenario values.")
    selection = [] if args.all else args.scenarios
    asyncio.run(main(selection or [], args.output.resolve()))
