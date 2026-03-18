# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Smoke tests for the three optimize_anything backends.

Uses a lightweight inline circle-packing evaluator (no subprocess, no heavy
deps) so the suite completes quickly.  Each backend is exercised for a handful
of iterations against the same circle-packing problem.
"""

import os
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything


# ---------------------------------------------------------------------------
# Lightweight circle-packing evaluator (inline, no subprocess)
# ---------------------------------------------------------------------------

_N = 10  # small n for speed


def _eval_circle_packing(candidate: str) -> tuple[float, dict]:
    """Execute candidate code inline and return (score, side_info).

    Expects the candidate to define ``def main() -> dict`` returning
    ``{"circles": ndarray(_N, 3)}`` where each row is (x, y, radius).
    """
    ns: dict = {}
    try:
        exec(compile(candidate, "<candidate>", "exec"), ns)  # noqa: S102
        result = ns["main"]()
        circles = np.array(result["circles"])
        if circles.shape != (_N, 3):
            return 0.0, {"error": f"Expected ({_N},3), got {circles.shape}"}
        cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(cx - r < -1e-6) or np.any(cx + r > 1 + 1e-6):
            return 0.0, {"error": "boundary violation x"}
        if np.any(cy - r < -1e-6) or np.any(cy + r > 1 + 1e-6):
            return 0.0, {"error": "boundary violation y"}
        if np.any(r < 0):
            return 0.0, {"error": "negative radius"}
        for i in range(_N):
            for j in range(i + 1, _N):
                d = float(np.linalg.norm(circles[i, :2] - circles[j, :2]))
                if d < circles[i, 2] + circles[j, 2] - 1e-6:
                    return 0.0, {"error": f"overlap {i}-{j}"}
        score = float(r.sum())
        return score, {"sum_radii": score, "circles": circles.tolist()}
    except Exception as e:
        return 0.0, {"error": str(e)}


_SEED = textwrap.dedent(f"""\
    import numpy as np

    N = {_N}

    def main():
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        cx = 0.5 + 0.35 * np.cos(angles)
        cy = 0.5 + 0.35 * np.sin(angles)
        r_boundary = np.minimum.reduce([cx, cy, 1 - cx, 1 - cy])
        r_pair = np.full(N, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(np.array([cx[i] - cx[j], cy[i] - cy[j]]))
                half = d / 2.0
                r_pair[i] = min(r_pair[i], half)
                r_pair[j] = min(r_pair[j], half)
        radii = np.minimum(r_boundary, r_pair) * 0.98
        circles = np.column_stack([cx, cy, radii])
        return {{"circles": circles}}
""")

_OBJECTIVE = (
    f"Maximize the sum of radii for {_N} non-overlapping circles in a unit square. "
    "Candidate must define 'def main()' returning {'circles': ndarray(N,3)} "
    "where each row is (x, y, radius)."
)

_BACKGROUND = textwrap.dedent(f"""\
    N = {_N}.  All circles must stay inside [0,1]x[0,1].
    No two circles may overlap.
    Return {{"circles": numpy array of shape (N, 3)}} from main().
    Goal: maximize sum of all radii.
""")


def _seed_score() -> float:
    score, _ = _eval_circle_packing(_SEED)
    return score


# ---------------------------------------------------------------------------
# Backend: gepa (default)
# ---------------------------------------------------------------------------


class TestGEPABackend:
    def test_gepa_backend_default(self):
        """GEPAConfig.backend defaults to 'gepa'."""
        assert GEPAConfig().backend == "gepa"

    def test_gepa_backend_does_not_call_skydiscover(self):
        """The 'gepa' backend must never reach _optimize_via_skydiscover."""
        with patch("gepa.optimize_anything._optimize_via_skydiscover") as mock_sd:
            mock_sd.side_effect = AssertionError("Must not be called for 'gepa' backend")
            with patch("gepa.optimize_anything.GEPAEngine") as mock_engine_cls:
                mock_state = MagicMock()
                mock_state.program_candidates = [{"current_candidate": _SEED}]
                mock_state.parent_program_for_candidate = [[None]]
                mock_state.prog_candidate_val_subscores = [{}]
                mock_state.program_at_pareto_front_valset = {}
                mock_state.program_full_scores_val_set = [_seed_score()]
                mock_state.num_metric_calls_by_discovery = [0]
                mock_state.total_num_evals = 8
                mock_state.num_full_ds_evals = 1
                mock_state.best_outputs_valset = None
                mock_state.objective_pareto_front = {}
                mock_state.program_at_pareto_front_objectives = {}
                mock_engine = MagicMock()
                mock_engine.run.return_value = mock_state
                mock_engine_cls.return_value = mock_engine

                result = optimize_anything(
                    seed_candidate=_SEED,
                    evaluator=_eval_circle_packing,
                    objective=_OBJECTIVE,
                    background=_BACKGROUND,
                    config=GEPAConfig(
                        backend="gepa",
                        engine=EngineConfig(max_metric_calls=8),
                        reflection=ReflectionConfig(reflection_lm=MagicMock(return_value=f"```python\n{_SEED}\n```")),
                    ),
                )

        assert result is not None
        assert result.val_aggregate_scores[0] == pytest.approx(_seed_score())

    def test_gepa_runs_circle_packing(self):
        """'gepa' backend actually evaluates the seed and produces a valid result."""
        mock_lm = MagicMock(return_value=f"```python\n{_SEED}\n```")

        result = optimize_anything(
            seed_candidate=_SEED,
            evaluator=_eval_circle_packing,
            objective=_OBJECTIVE,
            background=_BACKGROUND,
            config=GEPAConfig(
                backend="gepa",
                engine=EngineConfig(max_metric_calls=8),
                reflection=ReflectionConfig(reflection_lm=mock_lm),
            ),
        )

        assert result.best_candidate is not None
        assert result.val_aggregate_scores[0] == pytest.approx(_seed_score(), abs=1e-4)
        # best_candidate unwraps to str when seed_candidate was a str
        assert isinstance(result.best_candidate, str)


# ---------------------------------------------------------------------------
# Backend: skydiscover-evox
# ---------------------------------------------------------------------------


class TestSkydiscoverEvoxBackend:
    def test_evox_circle_packing(self):
        """'skydiscover-evox' runs circle packing and returns a GEPAResult."""
        from skydiscover.api import DiscoveryResult as SDResult

        seed = _seed_score()
        better = seed + 0.05
        better_code = _SEED.replace("* 0.98", "* 0.99")

        mock_result = SDResult(
            best_program=None,
            best_score=better,
            best_solution=better_code,
            metrics={"combined_score": better},
            output_dir=None,
            initial_score=seed,
        )

        with patch("skydiscover.discover_solution", return_value=mock_result) as mock_ds:
            result = optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                objective=_OBJECTIVE,
                background=_BACKGROUND,
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=8),
                ),
            )

        kw = mock_ds.call_args[1]
        assert kw["search"] == "evox"
        assert kw["iterations"] == 8
        assert kw["initial_solution"] == _SEED

        assert result.best_candidate == better_code
        assert result.val_aggregate_scores == pytest.approx([seed, better])
        assert len(result.candidates) == 2

    def test_evox_rejects_dataset(self):
        """skydiscover-evox raises ValueError when dataset is provided."""
        with pytest.raises(ValueError, match="single-task"):
            optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                dataset=[{"x": 1}],
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )

    def test_evox_raises_without_skydiscover(self, monkeypatch):
        """Clear ImportError when skydiscover package is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "skydiscover":
                raise ImportError("No module named 'skydiscover'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pip install gepa\\[skydiscover\\]"):
            optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )


# ---------------------------------------------------------------------------
# Backend: skydiscover-adaevolve
# ---------------------------------------------------------------------------


class TestSkydiscoverAdaevolveBackend:
    def test_adaevolve_circle_packing(self):
        """'skydiscover-adaevolve' runs circle packing and returns a GEPAResult."""
        from skydiscover.api import DiscoveryResult as SDResult

        seed = _seed_score()
        better = seed + 0.10
        better_code = _SEED.replace("* 0.98", "* 0.995")

        mock_result = SDResult(
            best_program=None,
            best_score=better,
            best_solution=better_code,
            metrics={"combined_score": better},
            output_dir=None,
            initial_score=seed,
        )

        with patch("skydiscover.discover_solution", return_value=mock_result) as mock_ds:
            result = optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                objective=_OBJECTIVE,
                background=_BACKGROUND,
                config=GEPAConfig(
                    backend="skydiscover-adaevolve",
                    engine=EngineConfig(max_metric_calls=8),
                ),
            )

        kw = mock_ds.call_args[1]
        assert kw["search"] == "adaevolve"
        assert kw["iterations"] == 8

        assert result.best_candidate == better_code
        assert result.val_aggregate_scores == pytest.approx([seed, better])

    def test_adaevolve_rejects_valset(self):
        """skydiscover-adaevolve raises ValueError when valset is provided."""
        with pytest.raises(ValueError, match="single-task"):
            optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                dataset=[{"x": 1}],
                valset=[{"x": 2}],
                config=GEPAConfig(
                    backend="skydiscover-adaevolve",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )

    def test_adaevolve_no_improvement(self):
        """When best == seed, a single-candidate result is returned."""
        from skydiscover.api import DiscoveryResult as SDResult

        seed = _seed_score()

        with patch("skydiscover.discover_solution", return_value=SDResult(
            best_program=None,
            best_score=seed,
            best_solution=_SEED,
            metrics={"combined_score": seed},
            output_dir=None,
            initial_score=seed,
        )):
            result = optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                config=GEPAConfig(
                    backend="skydiscover-adaevolve",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )

        assert len(result.candidates) == 1
        assert result.val_aggregate_scores[0] == pytest.approx(seed)


# ---------------------------------------------------------------------------
# Evaluator adapter integration
# ---------------------------------------------------------------------------


class TestEvaluatorAdapter:
    def test_adapter_reads_program_file_and_scores(self):
        """The wrapped evaluator reads a .py file and correctly scores it."""
        from skydiscover.api import DiscoveryResult as SDResult

        captured: list[dict] = []

        def mock_ds(**kwargs):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(_SEED)
                tmp = f.name
            try:
                score_dict = kwargs["evaluator"](tmp)
                captured.append(score_dict)
            finally:
                os.unlink(tmp)
            seed = score_dict["combined_score"]
            return SDResult(
                best_program=None,
                best_score=seed,
                best_solution=_SEED,
                metrics=score_dict,
                output_dir=None,
                initial_score=seed,
            )

        with patch("skydiscover.discover_solution", side_effect=mock_ds):
            optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=5),
                ),
            )

        assert len(captured) == 1
        assert captured[0]["combined_score"] == pytest.approx(_seed_score())
