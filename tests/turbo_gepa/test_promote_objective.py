import asyncio
import os
import shutil
import tempfile
import types

import pytest

from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.interfaces import Candidate


@pytest.mark.asyncio
async def test_neg_cost_promote_objective(monkeypatch):
    """Ensure non-quality promote_objective flows through evaluator/mutator/orchestrator."""

    tmpdir = tempfile.mkdtemp(prefix="tg_obj_test_")
    try:
        dataset = [
            DefaultDataInst(id="ex0", input="foo", answer="bar"),
            DefaultDataInst(id="ex1", input="baz", answer="qux"),
        ]
        cfg = Config(
            shards=(0.5, 1.0),
            eval_concurrency=2,
            max_mutations_per_round=0,
            cache_path=os.path.join(tmpdir, "cache"),
            log_path=os.path.join(tmpdir, "logs"),
            scoring_fn=lambda ctx: float(ctx.result.objectives.get("neg_cost", 0.0)),
        )

        adapter = DefaultAdapter(
            dataset=dataset,
            task_lm="dummy/task",
            reflection_lm="dummy/reflection",
            config=cfg,
        )

        async def fake_task_runner(self, candidate: Candidate, example_id: str):
            await asyncio.sleep(0)
            return {"quality": 0.0, "neg_cost": -10.0, "tokens": 10.0}

        adapter._task_runner = types.MethodType(fake_task_runner, adapter)

        orchestrator = adapter._build_orchestrator(display_progress=False)
        seed = Candidate(text="seed prompt")
        await orchestrator._seed_archive([seed])

        assert orchestrator.metrics.best_rung_quality == -10.0
        assert any(
            "neg_cost" in res.objectives
            for res in orchestrator.latest_results.values()
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
