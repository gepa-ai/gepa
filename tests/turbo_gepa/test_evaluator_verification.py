import asyncio
from pathlib import Path

import pytest

from turbo_gepa.cache import DiskCache
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate, EvalResult


async def _run_eval(
    tmp_path: Path,
    quality: float,
    *,
    target_quality: float | None = None,
    is_final: bool = True,
) -> tuple[EvalResult, AsyncEvaluator, list[str]]:
    calls: list[str] = []

    async def runner(candidate: Candidate, example_id: str) -> dict[str, float]:
        calls.append(example_id)
        await asyncio.sleep(0)  # ensure scheduler yields
        return {"quality": quality}

    cache_dir = tmp_path / ("cache_final" if is_final else "cache_mid")
    cache_dir.mkdir(exist_ok=True)
    evaluator = AsyncEvaluator(
        cache=DiskCache(str(cache_dir), namespace="test"),
        task_runner=runner,
        metrics_mapper=lambda metrics: metrics,
        timeout_seconds=None,
        min_improve=0.0,
        skip_final_straggler_cutoff=False,
        promote_objective="quality",
        cancel_stragglers_immediately=False,
        replay_stragglers=False,
        min_samples_for_confidence=5,
        target_quality=target_quality,
    )
    example_ids = [f"ex{i}" for i in range(40)]
    candidate = Candidate(text="seed", meta={"parent_objectives": {"quality": 0.4}})
    result = await evaluator.eval_on_shard(
        candidate,
        example_ids,
        concurrency=4,
        shard_fraction=1.0 if is_final else 0.5,
        show_progress=False,
        is_final_shard=is_final,
    )
    return result, evaluator, calls


@pytest.mark.asyncio
async def test_async_evaluator_stops_early_for_success(tmp_path: Path) -> None:
    result, evaluator, calls = await _run_eval(tmp_path, quality=0.95, target_quality=0.7)
    # Should stop well before exhausting all examples
    assert result.n_examples < 40
    # Tail ratio should be recorded regardless of logging mode
    assert evaluator.tail_latency_ratio >= 1.0


@pytest.mark.asyncio
async def test_async_evaluator_stops_early_for_failure(tmp_path: Path) -> None:
    result, evaluator, calls = await _run_eval(tmp_path, quality=0.1, target_quality=0.7)
    assert result.n_examples < 40
    assert evaluator.tail_latency_ratio >= 1.0
