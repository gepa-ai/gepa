# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Real end-to-end smoke tests for the three optimize_anything backends.

All three backends actually run to completion (no mocks).  LLM calls are
intercepted at the lowest level:
  - GEPA: reflection_lm is a Python callable that returns the seed code.
  - skydiscover-evox / skydiscover-adaevolve: openai.OpenAI is patched so
    chat.completions.create returns the seed code in a ```python``` block.

We use a lightweight inline circle-packing evaluator (N=10, no subprocess)
so each iteration is fast.  Each backend is run for just 3 iterations.
"""

import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything


# ---------------------------------------------------------------------------
# Lightweight circle-packing evaluator
# ---------------------------------------------------------------------------

_N = 10


def _eval_circle_packing(candidate: str) -> tuple[float, dict]:
    """Inline evaluator: execute candidate Python, return sum-of-radii score."""
    ns: dict = {}
    try:
        exec(compile(candidate, "<candidate>", "exec"), ns)  # noqa: S102
        result = ns["main"]()
        circles = np.array(result["circles"])
        if circles.shape != (_N, 3):
            return 0.0, {"error": f"Expected ({_N},3), got {circles.shape}"}
        cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(r < 0):
            return 0.0, {"error": "negative radius"}
        if np.any(cx - r < -1e-6) or np.any(cx + r > 1 + 1e-6):
            return 0.0, {"error": "boundary violation x"}
        if np.any(cy - r < -1e-6) or np.any(cy + r > 1 + 1e-6):
            return 0.0, {"error": "boundary violation y"}
        for i in range(_N):
            for j in range(i + 1, _N):
                d = float(np.linalg.norm(circles[i, :2] - circles[j, :2]))
                if d < circles[i, 2] + circles[j, 2] - 1e-6:
                    return 0.0, {"error": f"overlap {i}-{j}"}
        score = float(r.sum())
        return score, {"sum_radii": score}
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
    "Candidate must define 'def main()' returning {'circles': ndarray(N,3)}."
)

_BACKGROUND = textwrap.dedent(f"""\
    N = {_N}.  All circles must stay inside [0,1]x[0,1].
    No two circles may overlap.  Return {{"circles": numpy array (N,3)}} from main().
    Goal: maximize sum of all radii.
""")


def _seed_score() -> float:
    score, _ = _eval_circle_packing(_SEED)
    return score


def _make_openai_mock(response_text: str):
    """Build a mock openai.OpenAI instance whose completions always return response_text."""
    choice = MagicMock()
    choice.message.content = response_text
    completion = MagicMock()
    completion.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    mock_openai_cls = MagicMock(return_value=client)
    return mock_openai_cls


# ---------------------------------------------------------------------------
# Backend: gepa
# ---------------------------------------------------------------------------


class TestGEPABackend:
    def test_gepa_runs_to_completion(self):
        """GEPA backend runs circle packing for 8 metric calls end-to-end."""
        # The reflection LM always returns the seed code — no real API calls.
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

        assert result is not None
        assert isinstance(result.best_candidate, str)
        assert len(result.val_aggregate_scores) >= 1
        # seed must have scored positively
        assert result.val_aggregate_scores[0] == pytest.approx(_seed_score(), abs=1e-4)


# ---------------------------------------------------------------------------
# Backend: skydiscover-evox
# ---------------------------------------------------------------------------


class TestSkydiscoverEvoxBackend:
    def test_evox_runs_to_completion(self):
        """skydiscover-evox backend runs circle packing for 3 iterations end-to-end."""
        llm_response = f"```python\n{_SEED}\n```"
        mock_openai_cls = _make_openai_mock(llm_response)

        with patch("skydiscover.llm.openai.openai.OpenAI", mock_openai_cls):
            result = optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                objective=_OBJECTIVE,
                background=_BACKGROUND,
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=3),
                    reflection=ReflectionConfig(reflection_lm="openai/gpt-4.1-mini"),
                ),
            )

        assert result is not None
        assert isinstance(result.best_candidate, str)
        assert len(result.val_aggregate_scores) >= 1
        # best score must be non-negative
        assert result.val_aggregate_scores[result.best_idx] >= 0.0

    def test_evox_rejects_dataset(self):
        """skydiscover-evox raises ValueError when dataset is provided."""
        with pytest.raises(ValueError, match="single-task"):
            optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                dataset=[{"x": 1}],
                config=GEPAConfig(
                    backend="skydiscover-evox",
                    engine=EngineConfig(max_metric_calls=3),
                ),
            )

    def test_evox_raises_without_skydiscover(self, monkeypatch):
        """Clear ImportError when skydiscover is not installed."""
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
                    engine=EngineConfig(max_metric_calls=3),
                ),
            )


# ---------------------------------------------------------------------------
# Backend: skydiscover-adaevolve
# ---------------------------------------------------------------------------


class TestSkydiscoverAdaevolveBackend:
    def test_adaevolve_runs_to_completion(self):
        """skydiscover-adaevolve backend runs circle packing for 3 iterations end-to-end."""
        llm_response = f"```python\n{_SEED}\n```"
        mock_openai_cls = _make_openai_mock(llm_response)

        with patch("skydiscover.llm.openai.openai.OpenAI", mock_openai_cls):
            result = optimize_anything(
                seed_candidate=_SEED,
                evaluator=_eval_circle_packing,
                objective=_OBJECTIVE,
                background=_BACKGROUND,
                config=GEPAConfig(
                    backend="skydiscover-adaevolve",
                    engine=EngineConfig(max_metric_calls=3),
                    reflection=ReflectionConfig(reflection_lm="openai/gpt-4.1-mini"),
                ),
            )

        assert result is not None
        assert isinstance(result.best_candidate, str)
        assert len(result.val_aggregate_scores) >= 1
        assert result.val_aggregate_scores[result.best_idx] >= 0.0

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
                    engine=EngineConfig(max_metric_calls=3),
                ),
            )
