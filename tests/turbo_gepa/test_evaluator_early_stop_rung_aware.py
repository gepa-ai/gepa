import asyncio
import os
import tempfile

import pytest

from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate
from turbo_gepa.cache import DiskCache


@pytest.mark.asyncio
async def test_early_stop_uses_parent_rung_scores():
    # Parent rung 0 baseline = 0.80; child examples at 0.40 should trigger early stop quickly
    # After two examples of 0.4 out of 5 total, best_possible=(0.8+3*1.0)/5=0.76 < 0.80 → stop
    tmpdir = tempfile.mkdtemp(prefix="tg_eval_cache_")
    try:
        cache = DiskCache(tmpdir)

        async def task_runner(_candidate: Candidate, _example_id: str):
            await asyncio.sleep(0.01)
            return {"quality": 0.40, "neg_cost": 0.0, "tokens": 1.0}

        evaluator = AsyncEvaluator(
            cache=cache,
            task_runner=task_runner,
            timeout_seconds=5.0,
            min_improve=0.0,
        )

        candidate = Candidate(
            text="test",
            meta={
                "parent_rung_scores": {0.2: 0.80},  # explicit rung-aware baseline
            },
        )
        example_ids = [f"ex{i}" for i in range(5)]
        result = await evaluator.eval_on_shard(
            candidate,
            example_ids,
            concurrency=5,
            shard_fraction=0.2,
            show_progress=False,
        )
        # Should have stopped early, not evaluate all examples
        assert result.n_examples < len(example_ids), "Early stop should cut off remaining examples"
    finally:
        try:
            # Best-effort cleanup
            for root, _dirs, files in os.walk(tmpdir, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                os.rmdir(root)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_early_stop_shrinkage_baseline():
    # No parent_rung_scores provided → use shrinkage from parent@final
    # Parent final = 0.90, alpha(0.2)=0.7 → baseline ≈ 0.78
    # With three examples, at two results of 0.4 best_possible=(0.8+1)/3=0.6 < 0.78 → stop
    tmpdir = tempfile.mkdtemp(prefix="tg_eval_cache_")
    try:
        cache = DiskCache(tmpdir)

        async def task_runner(_candidate: Candidate, _example_id: str):
            await asyncio.sleep(0.01)
            return {"quality": 0.40, "neg_cost": 0.0, "tokens": 1.0}

        evaluator = AsyncEvaluator(
            cache=cache,
            task_runner=task_runner,
            timeout_seconds=5.0,
            min_improve=0.0,
        )

        candidate = Candidate(
            text="test",
            meta={
                "parent_objectives": {"quality": 0.90},  # final parent score only
            },
        )
        example_ids = [f"ex{i}" for i in range(3)]
        result = await evaluator.eval_on_shard(
            candidate,
            example_ids,
            concurrency=3,
            shard_fraction=0.2,
            show_progress=False,
        )
        assert result.n_examples < len(example_ids), "Early stop via shrinkage baseline should cut off tasks"
    finally:
        try:
            for root, _dirs, files in os.walk(tmpdir, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                os.rmdir(root)
        except Exception:
            pass

