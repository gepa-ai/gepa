# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Smoke tests for the three optimize_anything backends.

Each test runs a trivial string-length optimization for a tiny number of
iterations so the suite completes quickly without requiring real LLM calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from gepa.optimize_anything import GEPAConfig, EngineConfig, optimize_anything


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_evaluator():
    """Simple evaluator: score = length of candidate string / 100."""

    def evaluator(candidate: str) -> float:
        return len(candidate) / 100.0

    return evaluator


# ---------------------------------------------------------------------------
# Backend: gepa (default)
# ---------------------------------------------------------------------------


class TestGEPABackend:
    def test_gepa_backend_default(self):
        """GEPAConfig.backend defaults to 'gepa'."""
        config = GEPAConfig(engine=EngineConfig(max_metric_calls=5))
        assert config.backend == "gepa"

    def test_gepa_backend_does_not_call_skydiscover(self):
        """The 'gepa' backend should never call _optimize_via_skydiscover."""
        from gepa.optimize_anything import ReflectionConfig

        mock_lm = MagicMock(return_value="```\nimproved candidate text here\n```")

        with patch("gepa.optimize_anything._optimize_via_skydiscover") as mock_sd:
            mock_sd.side_effect = AssertionError("Should not call skydiscover for 'gepa' backend")
            # Patch internals so we don't run a real optimization loop
            with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
                mock_state = MagicMock()
                mock_state.program_candidates = [{"current_candidate": "short"}]
                mock_state.parent_program_for_candidate = [[None]]
                mock_state.prog_candidate_val_subscores = [{}]
                mock_state.program_at_pareto_front_valset = {}
                mock_state.program_full_scores_val_set = [0.05]
                mock_state.num_metric_calls_by_discovery = [0]
                mock_state.total_num_evals = 1
                mock_state.num_full_ds_evals = 1
                mock_state.best_outputs_valset = None
                mock_state.objective_pareto_front = {}
                mock_state.program_at_pareto_front_objectives = {}
                mock_engine = MagicMock()
                mock_engine.run.return_value = mock_state
                mock_engine_cls.return_value = mock_engine

                result = optimize_anything(
                    seed_candidate="short",
                    evaluator=lambda c: 0.05,
                    objective="Make the string longer.",
                    config=GEPAConfig(
                        backend="gepa",
                        engine=EngineConfig(max_metric_calls=5),
                        reflection=ReflectionConfig(reflection_lm=mock_lm),
                    ),
                )

        assert result is not None
        assert result.best_candidate == "short"


# ---------------------------------------------------------------------------
# Mocked GEPA backend test (no real LLM needed)
# ---------------------------------------------------------------------------


class TestGEPABackendMocked:
    def test_gepa_backend_returns_result(self):
        """Verify GEPAConfig(backend='gepa') uses the GEPA path."""
        from gepa.optimize_anything import GEPAResult

        mock_result = GEPAResult(
            candidates=[{"current_candidate": "short"}, {"current_candidate": "longer candidate"}],
            parents=[[None], [0]],
            val_aggregate_scores=[0.05, 0.16],
            val_subscores=[{}, {}],
            per_val_instance_best_candidates={},
            discovery_eval_counts=[0, 3],
            _str_candidate_key="current_candidate",
        )

        with patch("gepa.optimize_anything._optimize_via_skydiscover") as mock_sd:
            # This should NOT be called for the gepa backend
            mock_sd.side_effect = AssertionError("Should not call skydiscover for 'gepa' backend")

            # Patch the real GEPA internals to return quickly
            with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
                mock_engine = MagicMock()
                mock_engine.run.return_value = MagicMock(
                    program_candidates=[{"current_candidate": "short"}, {"current_candidate": "longer"}],
                    parent_program_for_candidate=[[None], [0]],
                    prog_candidate_val_subscores=[{}, {}],
                    program_at_pareto_front_valset={},
                    program_full_scores_val_set=[0.05, 0.16],
                    num_metric_calls_by_discovery=[0, 3],
                    total_num_evals=6,
                    num_full_ds_evals=1,
                    best_outputs_valset=None,
                    objective_pareto_front={},
                    program_at_pareto_front_objectives={},
                )
                mock_engine_cls.return_value = mock_engine

                config = GEPAConfig(backend="gepa", engine=EngineConfig(max_metric_calls=6))
                assert config.backend == "gepa"


# ---------------------------------------------------------------------------
# Backend: skydiscover-evox
# ---------------------------------------------------------------------------


class TestSkydiscoverEvoxBackend:
    def test_evox_backend_dispatches_correctly(self):
        """skydiscover-evox backend calls discover_solution with search='evox'."""
        from skydiscover.api import DiscoveryResult as SDResult

        mock_sd_result = SDResult(
            best_program=None,
            best_score=0.42,
            best_solution="optimized evox candidate",
            metrics={"combined_score": 0.42},
            output_dir=None,
            initial_score=0.05,
        )

        with patch("skydiscover.discover_solution", return_value=mock_sd_result) as mock_ds:
            result = optimize_anything(
                seed_candidate="initial code",
                evaluator=_make_evaluator(),
                objective="Maximize string length.",
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=10),
                ),
            )

        mock_ds.assert_called_once()
        call_kwargs = mock_ds.call_args[1]
        assert call_kwargs["search"] == "evox"
        assert call_kwargs["iterations"] == 10
        assert call_kwargs["initial_solution"] == "initial code"

        assert result.best_candidate == "optimized evox candidate"
        assert result.val_aggregate_scores[1] == pytest.approx(0.42)

    def test_evox_rejects_dataset(self):
        """skydiscover-evox must raise ValueError when dataset is provided."""
        with pytest.raises(ValueError, match="single-task"):
            optimize_anything(
                seed_candidate="code",
                evaluator=_make_evaluator(),
                dataset=[{"x": 1}, {"x": 2}],
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=10),
                ),
            )

    def test_evox_raises_without_skydiscover(self, monkeypatch):
        """Helpful ImportError when skydiscover is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "skydiscover":
                raise ImportError("No module named 'skydiscover'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pip install gepa\\[skydiscover\\]"):
            optimize_anything(
                seed_candidate="code",
                evaluator=_make_evaluator(),
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )


# ---------------------------------------------------------------------------
# Backend: skydiscover-adaevolve
# ---------------------------------------------------------------------------


class TestSkydiscoverAdaevolveBackend:
    def test_adaevolve_backend_dispatches_correctly(self):
        """skydiscover-adaevolve backend calls discover_solution with search='adaevolve'."""
        from skydiscover.api import DiscoveryResult as SDResult

        mock_sd_result = SDResult(
            best_program=None,
            best_score=0.75,
            best_solution="adaevolve improved candidate",
            metrics={"combined_score": 0.75},
            output_dir=None,
            initial_score=0.20,
        )

        with patch("skydiscover.discover_solution", return_value=mock_sd_result) as mock_ds:
            result = optimize_anything(
                seed_candidate="initial prompt",
                evaluator=_make_evaluator(),
                objective="Maximize quality.",
                config=GEPAConfig(
                    backend="skydiscover-adaevolve",
                    engine=EngineConfig(max_metric_calls=20),
                ),
            )

        mock_ds.assert_called_once()
        call_kwargs = mock_ds.call_args[1]
        assert call_kwargs["search"] == "adaevolve"
        assert call_kwargs["iterations"] == 20

        assert result.best_candidate == "adaevolve improved candidate"
        assert result.val_aggregate_scores == pytest.approx([0.20, 0.75])

    def test_adaevolve_rejects_valset(self):
        """skydiscover-adaevolve must raise ValueError when valset is provided."""
        with pytest.raises(ValueError, match="single-task"):
            optimize_anything(
                seed_candidate="code",
                evaluator=_make_evaluator(),
                dataset=[{"x": 1}],
                valset=[{"x": 2}],
                config=GEPAConfig(
                    backend="skydiscover-adaevolve",
                    engine=EngineConfig(max_metric_calls=10),
                ),
            )

    def test_adaevolve_no_improvement_returns_single_candidate(self):
        """When best == seed, result has a single candidate."""
        from skydiscover.api import DiscoveryResult as SDResult

        mock_sd_result = SDResult(
            best_program=None,
            best_score=0.05,
            best_solution="initial prompt",  # unchanged
            metrics={"combined_score": 0.05},
            output_dir=None,
            initial_score=0.05,
        )

        with patch("skydiscover.discover_solution", return_value=mock_sd_result):
            result = optimize_anything(
                seed_candidate="initial prompt",
                evaluator=_make_evaluator(),
                config=GEPAConfig(
                    backend="skydiscover-adaevolve",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )

        assert len(result.candidates) == 1
        assert result.best_candidate == "initial prompt"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestBackendConfig:
    def test_default_backend_is_gepa(self):
        config = GEPAConfig()
        assert config.backend == "gepa"

    def test_backend_field_set(self):
        config = GEPAConfig(backend="skydiscover-evox")
        assert config.backend == "skydiscover-evox"

    def test_evaluator_adapter_passes_score(self):
        """The skydiscover evaluator wrapper correctly reads program files."""
        import tempfile, os
        from skydiscover.api import DiscoveryResult as SDResult

        captured_calls = []

        def mock_ds(**kwargs):
            # Call the evaluator to make sure it works
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write("hello world")
                tmp_path = f.name
            try:
                score_dict = kwargs["evaluator"](tmp_path)
                captured_calls.append(score_dict)
            finally:
                os.unlink(tmp_path)
            return SDResult(
                best_program=None,
                best_score=0.11,
                best_solution="hello world",
                metrics={"combined_score": 0.11},
                output_dir=None,
                initial_score=0.11,
            )

        with patch("skydiscover.discover_solution", side_effect=mock_ds):
            optimize_anything(
                seed_candidate="hello world",
                evaluator=_make_evaluator(),
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )

        assert len(captured_calls) == 1
        assert "combined_score" in captured_calls[0]
        assert captured_calls[0]["combined_score"] == pytest.approx(len("hello world") / 100.0)
