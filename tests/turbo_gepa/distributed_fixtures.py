from __future__ import annotations

import asyncio
import os
from typing import Sequence

from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


def _build_dataset(size: int = 6) -> list[DefaultDataInst]:
    dataset: list[DefaultDataInst] = []
    for idx in range(size):
        dataset.append(
            DefaultDataInst(
                input=f"Example {idx}",
                answer=str(idx),
                id=f"fixture-{idx}",
            )
        )
    return dataset


def build_test_adapter() -> tuple[DefaultAdapter, Sequence[str]]:
    """Factory used by distributed runner tests."""

    config = Config(
        n_islands=4,
        eval_concurrency=2,
        max_mutations_per_round=0,
        shards=(0.5, 1.0),
        cache_path=os.environ.get("TURBOGEPA_CACHE_PATH", ".turbo_gepa/cache"),
        log_path=os.environ.get("TURBOGEPA_LOG_PATH", ".turbo_gepa/logs"),
        migration_backend="volume",
        migration_path=os.path.join(os.environ.get("TURBOGEPA_CACHE_PATH", ".turbo_gepa/cache"), "migrations"),
    )
    dataset = _build_dataset()
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="local/test-model",
        reflection_lm="local/test-model",
        config=config,
        auto_config=False,
    )

    async def fake_task_runner(candidate, example_id: str):
        await asyncio.sleep(0)
        base = (len(candidate.text) + len(example_id)) % 10
        quality = 0.4 + (base * 0.05)
        return {"quality": min(quality, 0.95), "neg_cost": 0.0, "tokens": 1.0}

    adapter._task_runner = fake_task_runner  # type: ignore[assignment]
    return adapter, (
        "You are a steady assistant. Respond concisely.",
        "Favor deterministic reasoning before answering.",
    )
