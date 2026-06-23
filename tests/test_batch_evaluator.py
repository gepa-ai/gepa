# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for batch_evaluator support in optimize_anything."""

from unittest.mock import MagicMock, patch

import pytest

from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    optimize_anything,
)
from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_state():
    s = MagicMock()
    s.program_candidates = [{"current_candidate": "v0"}, {"current_candidate": "v1"}]
    s.parent_program_for_candidate = [[None], [0]]
    s.prog_candidate_val_subscores = [{}, {}]
    s.program_at_pareto_front_valset = {}
    s.program_full_scores_val_set = [0.5, 0.8]
    s.num_metric_calls_by_discovery = [0, 1]
    s.total_num_evals = 3
    s.num_full_ds_evals = 1
    s.best_outputs_valset = None
    s.objective_pareto_front = {}
    s.program_at_pareto_front_objectives = {}
    return s


def _dummy_evaluator(candidate, example=None):
    return 0.5, {}


# ---------------------------------------------------------------------------
# _call_batch_evaluator — normalisation
# ---------------------------------------------------------------------------

class TestCallBatchEvaluator:
    def _make_adapter(self, batch_evaluator):
        from gepa.optimize_anything import EvaluatorWrapper
        wrapper = EvaluatorWrapper(_dummy_evaluator, single_instance_mode=False)
        return OptimizeAnythingAdapter(evaluator=wrapper, batch_evaluator=batch_evaluator, cache_mode="off")

    def test_score_only_results(self):
        """batch_evaluator returning plain floats is normalised correctly."""
        def batch_eval(pairs):
            return [0.7] * len(pairs)

        adapter = self._make_adapter(batch_eval)
        results = adapter._call_batch_evaluator([{"id": 0}, {"id": 1}], {"current_candidate": "x"})
        assert len(results) == 2
        assert results[0] == (0.7, None, {})

    def test_score_and_side_info_results(self):
        """batch_evaluator returning (score, side_info) tuples."""
        def batch_eval(pairs):
            return [(0.9, {"key": i}) for i, _ in enumerate(pairs)]

        adapter = self._make_adapter(batch_eval)
        results = adapter._call_batch_evaluator([{"id": 0}, {"id": 1}, {"id": 2}], {"current_candidate": "x"})
        assert len(results) == 3
        assert results[1][0] == 0.9
        assert results[1][2] == {"key": 1}

    def test_pairs_passed_correctly(self):
        """batch_evaluator receives (candidate, example) pairs."""
        received = []

        def batch_eval(pairs):
            received.extend(pairs)
            return [0.5] * len(pairs)

        adapter = self._make_adapter(batch_eval)
        candidate = {"current_candidate": "my_code"}
        batch = [{"id": 0}, {"id": 1}]
        adapter._call_batch_evaluator(batch, candidate)

        assert len(received) == 2
        assert received[0] == (candidate, {"id": 0})
        assert received[1] == (candidate, {"id": 1})

    def test_length_mismatch_raises(self):
        """Mismatched result length raises a clear ValueError."""
        def bad_batch_eval(pairs):
            return [0.5]  # always returns 1 result regardless of batch size

        adapter = self._make_adapter(bad_batch_eval)
        with pytest.raises(ValueError, match="lengths must match"):
            adapter._call_batch_evaluator([{"id": 0}, {"id": 1}], {"current_candidate": "x"})


# ---------------------------------------------------------------------------
# OptimizeAnythingAdapter.evaluate — batch_evaluator takes priority
# ---------------------------------------------------------------------------

class TestAdapterEvaluateBatchPath:
    def _make_adapter(self, batch_evaluator):
        from gepa.optimize_anything import EvaluatorWrapper
        wrapper = EvaluatorWrapper(_dummy_evaluator, single_instance_mode=False)
        return OptimizeAnythingAdapter(
            evaluator=wrapper,
            batch_evaluator=batch_evaluator,
            parallel=True,
            cache_mode="off",
        )

    def test_batch_evaluator_called_not_thread_pool(self):
        """When batch_evaluator is set, ThreadPoolExecutor is NOT used."""
        call_log = []

        def batch_eval(pairs):
            call_log.append(len(pairs))
            return [0.6] * len(pairs)

        adapter = self._make_adapter(batch_eval)

        with patch("gepa.adapters.optimize_anything_adapter.optimize_anything_adapter.ThreadPoolExecutor") as mock_tpe:
            result = adapter.evaluate(
                [{"id": 0}, {"id": 1}, {"id": 2}],
                {"current_candidate": "code"},
            )

        # ThreadPoolExecutor was never instantiated
        mock_tpe.assert_not_called()
        # batch_evaluator was called once with the full batch
        assert call_log == [3]
        assert len(result.scores) == 3
        assert all(s == 0.6 for s in result.scores)

    def test_single_example_batch(self):
        """batch_evaluator is called even for single-example batches."""
        call_log = []

        def batch_eval(pairs):
            call_log.append(len(pairs))
            return [(0.8, {"single": True})]

        adapter = self._make_adapter(batch_eval)
        result = adapter.evaluate([{"id": 0}], {"current_candidate": "x"})

        assert call_log == [1]
        assert result.scores[0] == 0.8


# ---------------------------------------------------------------------------
# optimize_anything — batch_evaluator integration
# ---------------------------------------------------------------------------

class TestOptimizeAnythingBatchEvaluator:
    def test_batch_evaluator_wires_through(self):
        """batch_evaluator passed to optimize_anything reaches the adapter."""
        from gepa.optimize_anything import EvaluatorWrapper

        captured_adapter: list[OptimizeAnythingAdapter] = []

        original_init = OptimizeAnythingAdapter.__init__

        def spy_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured_adapter.append(self)

        batch_call_log = []

        def my_batch_eval(pairs):
            batch_call_log.append(len(pairs))
            return [0.5] * len(pairs)

        with patch.object(OptimizeAnythingAdapter, "__init__", spy_init):
            with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
                mock_engine = MagicMock()
                mock_engine.run.return_value = _make_mock_state()
                mock_engine_cls.return_value = mock_engine

                optimize_anything(
                    seed_candidate="hello",
                    evaluator=_dummy_evaluator,
                    batch_evaluator=my_batch_eval,
                    config=GEPAConfig(
                        engine=EngineConfig(max_metric_calls=5),
                        reflection=ReflectionConfig(
                            reflection_lm=MagicMock(return_value="```\nhello\n```")
                        ),
                    ),
                )

        assert len(captured_adapter) == 1
        assert captured_adapter[0].batch_evaluator is my_batch_eval

    def test_no_batch_evaluator_unchanged(self):
        """Without batch_evaluator, behaviour is identical to before."""
        with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_state()
            mock_engine_cls.return_value = mock_engine

            result = optimize_anything(
                seed_candidate="hello",
                evaluator=_dummy_evaluator,
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=5),
                    reflection=ReflectionConfig(
                        reflection_lm=MagicMock(return_value="```\nhello\n```")
                    ),
                ),
            )

        assert result is not None

    def test_batch_evaluator_with_dataset(self):
        """batch_evaluator works in multi-task (dataset) mode."""
        batch_log = []

        def batch_eval(pairs):
            batch_log.append([(c, e) for c, e in pairs])
            return [(0.7, {"batch": True})] * len(pairs)

        with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_state()
            mock_engine_cls.return_value = mock_engine

            result = optimize_anything(
                seed_candidate="code",
                evaluator=_dummy_evaluator,
                batch_evaluator=batch_eval,
                dataset=[{"id": 0}, {"id": 1}, {"id": 2}],
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=5),
                    reflection=ReflectionConfig(
                        reflection_lm=MagicMock(return_value="```\ncode\n```")
                    ),
                ),
            )

        assert result is not None

    def test_batch_evaluator_only_no_evaluator(self):
        """batch_evaluator alone (no evaluator) is accepted."""
        def batch_eval(pairs):
            return [0.5] * len(pairs)

        with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_state()
            mock_engine_cls.return_value = mock_engine

            result = optimize_anything(
                seed_candidate="code",
                batch_evaluator=batch_eval,
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=5),
                    reflection=ReflectionConfig(
                        reflection_lm=MagicMock(return_value="```\ncode\n```")
                    ),
                ),
            )

        assert result is not None

    def test_neither_evaluator_nor_batch_raises(self):
        """Providing neither evaluator nor batch_evaluator raises ValueError."""
        with pytest.raises(ValueError, match="Either 'evaluator' or 'batch_evaluator'"):
            optimize_anything(
                seed_candidate="code",
                config=GEPAConfig(engine=EngineConfig(max_metric_calls=5)),
            )

    def test_capture_stdio_with_batch_evaluator_raises(self):
        """capture_stdio=True alongside batch_evaluator raises ValueError."""
        with pytest.raises(ValueError, match="capture_stdio"):
            optimize_anything(
                seed_candidate="code",
                batch_evaluator=lambda pairs: [0.5] * len(pairs),
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=5, capture_stdio=True),
                    reflection=ReflectionConfig(
                        reflection_lm=MagicMock(return_value="```\ncode\n```")
                    ),
                ),
            )

    def test_both_evaluator_and_batch_warns(self):
        """Passing both evaluator and batch_evaluator emits a UserWarning."""

        def batch_eval(pairs):
            return [0.5] * len(pairs)

        with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.run.return_value = _make_mock_state()
            mock_engine_cls.return_value = mock_engine

            with pytest.warns(UserWarning, match="batch_evaluator"):
                optimize_anything(
                    seed_candidate="code",
                    evaluator=_dummy_evaluator,
                    batch_evaluator=batch_eval,
                    config=GEPAConfig(
                        engine=EngineConfig(max_metric_calls=5),
                        reflection=ReflectionConfig(
                            reflection_lm=MagicMock(return_value="```\ncode\n```")
                        ),
                    ),
                )
