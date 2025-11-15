from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from turbo_gepa.adapters import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


@pytest.mark.asyncio
async def test_control_dir_creates_stop_signal(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    control_dir = tmp_path / "control"
    config = Config(
        eval_concurrency=1,
        n_islands=1,
        shards=(1.0,),
        cache_path=str(cache_dir),
        log_path=str(log_dir),
        control_dir=str(control_dir),
        max_mutations_per_round=0,
        target_quality=0.6,
    )
    dataset = [
        DefaultDataInst(
            input="What is 1+1?",
            answer="2",
            id="ex0",
        )
    ]
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="local/test-model",
        reflection_lm="local/test-model",
        config=config,
        auto_config=False,
    )

    async def fake_task_runner(_candidate, _example_id: str):
        await asyncio.sleep(0)
        return {"quality": 1.0, "neg_cost": 0.0, "tokens": 1.0}

    adapter._task_runner = fake_task_runner  # type: ignore[assignment]

    await adapter.optimize_async(
        ["You are a helpful assistant."],
        max_rounds=1,
        max_evaluations=4,
        display_progress=False,
    )

    stop_file = control_dir / "stop.json"
    assert stop_file.exists(), "expected stop signal to be written"
    heartbeat_files = list(control_dir.glob("heartbeat_*.json"))
    assert heartbeat_files, "expected worker heartbeat file"
