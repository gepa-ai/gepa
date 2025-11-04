#!/usr/bin/env python3
"""
Smoke test that mirrors examples/quickstart.py with the production models.

Run with:
    pytest tests/turbo_gepa/test_quickstart_models.py -s

Requires OPENROUTER_API_KEY in the environment. Skipped automatically otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure src/ is importable when running the test directly
SRC_ROOT = Path(__file__).parent.parent.parent / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

import gepa  # type: ignore  # Imported after path adjustment
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


REQUIRED_ENV = "OPENROUTER_API_KEY"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    REQUIRED_ENV not in os.environ,
    reason="OPENROUTER_API_KEY required to exercise real models",
)
def test_quickstart_models_smoke():
    """
    Minimal end-to-end run using the canonical task/reflection models.

    Mirrors examples/quickstart.py but trims the dataset and time budgets so
    we can get a fast indicator that everything is wired correctly.
    """
    # Load a handful of AIME problems
    trainset, _, _ = gepa.examples.aime.init_dataset()
    dataset = [
        DefaultDataInst(input=ex["input"], answer=ex["answer"], id=f"aime_{i}")
        for i, ex in enumerate(trainset[:3])
    ]

    # Keep the run tiny: small concurrency, 1 round, short timeout
    config = Config(
        eval_concurrency=2,
        n_islands=1,
        shards=(0.6, 1.0),
        batch_size=2,
        max_mutations_per_round=2,
        max_optimization_time_seconds=60,
        log_level="WARNING",
    )

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="openrouter/openai/gpt-oss-20b:nitro",
        reflection_lm="openrouter/x-ai/grok-4-fast",
        config=config,
        auto_config=False,
    )

    result = adapter.optimize(
        seeds=[
            "You are a helpful math assistant. Solve the problem step by step and put your final answer after ###."
        ],
        max_rounds=1,
        enable_auto_stop=False,
        display_progress=False,
    )

    # Basic sanity checks
    pareto_entries = result.get("pareto_entries") or []
    assert pareto_entries, "Expected at least one Pareto candidate"

    best = max(
        pareto_entries,
        key=lambda e: e.result.objectives.get("quality", 0.0),
    )
    quality = best.result.objectives.get("quality", 0.0)
    assert 0.0 <= quality <= 1.0

    # Ensure evolution stats are populated
    evolution_stats = result.get("evolution_stats", {}) or {}
    assert evolution_stats.get("total_evaluations", 0) > 0
***
