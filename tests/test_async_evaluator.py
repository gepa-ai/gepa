# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for async evaluator support in optimize_anything and optimize."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import gepa
from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    optimize_anything,
    aoptimize_anything,
    EvaluatorWrapper,
    _run_coroutine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_engine_state():
    mock_state = MagicMock()
    mock_state.program_candidates = [{"current_candidate": "v0"}, {"current_candidate": "v1"}]
    mock_state.parent_program_for_candidate = [[None], [0]]
    mock_state.prog_candidate_val_subscores = [{}, {}]
    mock_state.program_at_pareto_front_valset = {}
    mock_state.program_full_scores_val_set = [0.5, 0.8]
    mock_state.num_metric_calls_by_discovery = [0, 1]
    mock_state.total_num_evals = 5
    mock_state.num_full_ds_evals = 1
    mock_state.best_outputs_valset = None
    mock_state.objective_pareto_front = {}
    mock_state.program_at_pareto_front_objectives = {}
    return mock_state


# ---------------------------------------------------------------------------
# _run_coroutine helper
# ---------------------------------------------------------------------------

class TestRunCoroutine:
    def test_runs_simple_coroutine(self):
        async def _coro():
            return 42

        result = _run_coroutine(_coro())
        assert result == 42

    def test_runs_from_existing_loop(self):
        """_run_coroutine works even when called from inside a running loop."""
        async def _outer():
            async def _inner():
                return "hello"
            # Call _run_coroutine from within a running loop — should use thread fallback
            return _run_coroutine(_inner())

        result = asyncio.run(_outer())
        assert result == "hello"


# ---------------------------------------------------------------------------
# EvaluatorWrapper — async detection
# ---------------------------------------------------------------------------

class TestEvaluatorWrapperAsync:
    def test_sync_evaluator_not_async(self):
        def sync_eval(candidate, example=None):
            return 1.0, {}

        wrapper = EvaluatorWrapper(sync_eval, single_instance_mode=True)
        assert wrapper.is_async is False
        assert wrapper._raw_fn is None

    def test_async_evaluator_detected(self):
        async def async_eval(candidate, example=None):
            return 1.0, {}

        wrapper = EvaluatorWrapper(async_eval, single_instance_mode=True)
        assert wrapper.is_async is True
        assert wrapper._raw_fn is async_eval

    def test_sync_evaluator_call_works(self):
        def sync_eval(candidate):
            return 0.7, {"info": "ok"}

        wrapper = EvaluatorWrapper(sync_eval, single_instance_mode=True)
        score, _, si = wrapper({"current_candidate": "test"})
        assert score == 0.7

    def test_async_evaluator_call_bridged(self):
        """Calling wrapper() on an async evaluator bridges via _run_coroutine."""
        async def async_eval(candidate):
            await asyncio.sleep(0)
            return 0.9, {"async": True}

        wrapper = EvaluatorWrapper(async_eval, single_instance_mode=True)
        score, _, si = wrapper({"current_candidate": "test"})
        assert score == 0.9
        assert si["async"] is True

    def test_async_call_method(self):
        """async_call returns a coroutine that can be awaited."""
        async def async_eval(candidate):
            return 0.5, {"key": "val"}

        wrapper = EvaluatorWrapper(async_eval, single_instance_mode=True)

        async def _run():
            return await wrapper.async_call({"current_candidate": "x"})

        score, _, si = asyncio.run(_run())
        assert score == 0.5
        assert si["key"] == "val"

    def test_async_call_with_example(self):
        """async_call correctly passes example in non-single-instance mode."""
        received = {}

        async def async_eval(candidate, example=None):
            received["example"] = example
            return 1.0, {}

        wrapper = EvaluatorWrapper(async_eval, single_instance_mode=False)

        async def _run():
            return await wrapper.async_call({"current_candidate": "x"}, example={"id": 42})

        asyncio.run(_run())
        assert received["example"] == {"id": 42}


# ---------------------------------------------------------------------------
# optimize_anything — sync evaluator (regression)
# ---------------------------------------------------------------------------

class TestOptimizeAnythingSyncRegression:
    def test_sync_evaluator_unchanged(self):
        call_count = [0]

        def sync_eval(candidate: str) -> float:
            call_count[0] += 1
            return float(len(candidate)) / 100.0

        with patch("gepa.optimize_anything.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_engine_state()
            mock_cls.return_value = mock_engine

            result = optimize_anything(
                seed_candidate="short",
                evaluator=sync_eval,
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=5),
                    reflection=ReflectionConfig(reflection_lm=MagicMock(return_value="```\nshort\n```")),
                ),
            )

        assert result is not None
        assert result.best_candidate is not None


# ---------------------------------------------------------------------------
# optimize_anything — async evaluator
# ---------------------------------------------------------------------------

class TestOptimizeAnythingAsyncEvaluator:
    def test_async_evaluator_accepted(self):
        """optimize_anything works with an async evaluator via sync bridge."""
        async def async_eval(candidate: str) -> tuple[float, dict]:
            await asyncio.sleep(0)  # simulate I/O
            return float(len(candidate)) / 100.0, {"async": True}

        with patch("gepa.optimize_anything.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_engine_state()
            mock_cls.return_value = mock_engine

            result = optimize_anything(
                seed_candidate="hello",
                evaluator=async_eval,
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=5),
                    reflection=ReflectionConfig(reflection_lm=MagicMock(return_value="```\nhello\n```")),
                ),
            )

        assert result is not None

    def test_async_evaluator_batch_gathered(self):
        """Async evaluators in the adapter are gathered concurrently."""
        from gepa.optimize_anything import EvaluatorWrapper
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter

        order: list[int] = []

        async def async_eval(candidate: str, example=None) -> tuple[float, dict]:
            idx = example.get("id", 0) if example else 0
            await asyncio.sleep(0.01 * (3 - idx))  # reverse delay
            order.append(idx)
            return 0.5, {}

        wrapper = EvaluatorWrapper(async_eval, single_instance_mode=False)
        adapter = OptimizeAnythingAdapter(evaluator=wrapper, parallel=True)

        batch = [{"id": 0}, {"id": 1}, {"id": 2}]
        eval_batch = adapter.evaluate(batch, {"current_candidate": "x"})

        # All 3 examples evaluated
        assert len(eval_batch.scores) == 3
        # With gather, shorter delays finish first: 2, 1, 0
        assert sorted(order) == [0, 1, 2]
        # Concurrent execution: id=2 (shortest delay) arrives before id=0
        assert order[0] == 2


# ---------------------------------------------------------------------------
# aoptimize_anything — async entry point
# ---------------------------------------------------------------------------

class TestAOptimizeAnything:
    def test_aoptimize_anything_returns_coroutine(self):
        """aoptimize_anything returns an awaitable."""
        import inspect

        async def async_eval(candidate: str) -> float:
            return 1.0

        coro = aoptimize_anything(
            seed_candidate="x",
            evaluator=async_eval,
            config=GEPAConfig(engine=EngineConfig(max_metric_calls=3)),
        )
        assert inspect.isawaitable(coro)
        coro.close()  # clean up without running

    def test_aoptimize_anything_runs(self):
        """aoptimize_anything completes successfully."""

        async def async_eval(candidate: str) -> tuple[float, dict]:
            await asyncio.sleep(0)
            return float(len(candidate)) / 100.0, {}

        with patch("gepa.optimize_anything.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_engine_state()
            mock_cls.return_value = mock_engine

            async def _run():
                return await aoptimize_anything(
                    seed_candidate="hello",
                    evaluator=async_eval,
                    config=GEPAConfig(
                        engine=EngineConfig(max_metric_calls=5),
                        reflection=ReflectionConfig(reflection_lm=MagicMock(return_value="```\nhello\n```")),
                    ),
                )

            result = asyncio.run(_run())

        assert result is not None
        assert result.best_candidate is not None


# ---------------------------------------------------------------------------
# aoptimize — async entry point
# ---------------------------------------------------------------------------

class TestAOptimize:
    def test_aoptimize_exported(self):
        """gepa.aoptimize is importable and awaitable."""
        import inspect
        assert hasattr(gepa, "aoptimize")
        assert inspect.iscoroutinefunction(gepa.aoptimize)

    def test_aoptimize_runs(self):
        from gepa.core.adapter import EvaluationBatch

        class SyncAdapter:
            propose_new_texts = None

            def evaluate(self, batch, candidate, capture_traces=False):
                scores = [0.5] * len(batch)
                return EvaluationBatch(outputs=scores, scores=scores)

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                return {c: [] for c in components_to_update}

        with patch("gepa.api.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_engine_state()
            mock_cls.return_value = mock_engine

            async def _run():
                return await gepa.aoptimize(
                    seed_candidate={"system_prompt": "hello"},
                    trainset=[{"q": "x"}],
                    valset=[{"q": "y"}],
                    adapter=SyncAdapter(),
                    reflection_lm=MagicMock(return_value="```\nhello\n```"),
                    max_metric_calls=5,
                )

            result = asyncio.run(_run())

        assert result is not None
