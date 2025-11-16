import asyncio
import os
import tempfile

from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate
from turbo_gepa.cache import DiskCache


def _cleanup_tmp(path: str) -> None:
    try:
        for root, _dirs, files in os.walk(path, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            os.rmdir(root)
    except Exception:
        pass


def test_early_stop_uses_parent_rung_scores():
    tmpdir = tempfile.mkdtemp(prefix="tg_eval_cache_")
    try:
        cache = DiskCache(tmpdir)

        async def task_runner(_candidate: Candidate, _example_id: str):
            await asyncio.sleep(0.01)
            return {"quality": 0.40, "neg_cost": 0.0, "tokens": 1.0}

        async def run_eval() -> int:
            evaluator = AsyncEvaluator(
                cache=cache,
                task_runner=task_runner,
                timeout_seconds=5.0,
                min_improve=0.0,
            )
            candidate = Candidate(
                text="test",
                meta={"parent_rung_scores": {0.2: 0.80}},
            )
            example_ids = [f"ex{i}" for i in range(5)]
            result = await evaluator.eval_on_shard(
                candidate,
                example_ids,
                concurrency=5,
                shard_fraction=0.2,
                show_progress=False,
            )
            return result.n_examples

        evaluated = asyncio.run(run_eval())
        assert evaluated < 5, "Early stop should cut off remaining examples"
    finally:
        _cleanup_tmp(tmpdir)


def test_early_stop_shrinkage_baseline():
    tmpdir = tempfile.mkdtemp(prefix="tg_eval_cache_")
    try:
        cache = DiskCache(tmpdir)

        async def task_runner(_candidate: Candidate, _example_id: str):
            await asyncio.sleep(0.01)
            return {"quality": 0.40, "neg_cost": 0.0, "tokens": 1.0}

        async def run_eval() -> int:
            evaluator = AsyncEvaluator(
                cache=cache,
                task_runner=task_runner,
                timeout_seconds=5.0,
                min_improve=0.0,
            )
            candidate = Candidate(
                text="test",
                meta={"parent_objectives": {"quality": 0.90}},
            )
            example_ids = [f"ex{i}" for i in range(3)]
            result = await evaluator.eval_on_shard(
                candidate,
                example_ids,
                concurrency=3,
                shard_fraction=0.2,
                show_progress=False,
            )
            return result.n_examples

        evaluated = asyncio.run(run_eval())
        assert evaluated < 3, "Early stop via shrinkage baseline should cut off tasks"
    finally:
        _cleanup_tmp(tmpdir)
