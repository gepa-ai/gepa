# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the gepa engine's 1-to-1 GEPAConfig pass-through in optimize_anything.

Guards the config-forwarding contract: the gepa engine treats
``OptimizeAnythingConfig.engine_config`` as a ``GEPAConfig``-shaped dict and
builds a ``GEPAConfig`` from it directly. A field like ``tracking`` therefore
reaches GEPA untouched, and a typo fails fast with ``TypeError`` rather than
being silently ignored.
"""

import glob
import os

import pytest

from gepa.gepa_launcher import TrackingConfig
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engines.gepa import GepaEngine


class _FakeLM:
    """Minimal reflection LM: deterministic, network-free, cost-tracked."""

    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    def __call__(self, *args, **kwargs):
        return "IMPROVED CANDIDATE: a longer and better answer than before"


def _evaluator(candidate, example=None):
    return float(len(candidate)) / 100.0, {"diagnostics": "ok"}


def test_tracking_key_is_forwarded_not_dropped():
    """A ``tracking`` block in engine_config lands on the built GEPAConfig."""
    tracking = TrackingConfig(use_wandb=False, use_mlflow=False, key_prefix="gepa/")
    engine = GepaEngine(
        OptimizeAnythingConfig(engine="gepa", max_evals=4, engine_config={"tracking": tracking})
    )
    assert engine.gepa_config.tracking.key_prefix == "gepa/"


def test_unknown_key_fails_fast():
    """A misspelled engine_config key raises TypeError at construction."""
    with pytest.raises(TypeError):
        GepaEngine(
            OptimizeAnythingConfig(engine="gepa", max_evals=4, engine_config={"trackign": TrackingConfig()})
        )


def test_legacy_gepa_config_call_returns_gepa_result():
    """A v0.1.x-style call — legacy imports, ``GEPAConfig`` — still runs through
    the converted gepa-engine path but returns the launcher's ``GEPAResult``,
    so old examples using ``result.candidates`` / ``best_idx`` /
    ``val_aggregate_scores`` run unmodified."""
    from gepa.core.result import GEPAResult
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=4),
            reflection=ReflectionConfig(reflection_lm=_FakeLM()),
        ),
    )

    assert isinstance(result, GEPAResult)
    assert result.best_candidate
    assert result.best_idx >= 0
    assert len(result.candidates) >= 1
    assert result.val_aggregate_scores


def test_no_config_returns_gepa_result(monkeypatch):
    """``optimize_anything(seed, evaluator=fn)`` with config omitted/None must
    return ``GEPAResult`` so simple-API callers can use ``val_aggregate_scores``,
    ``best_idx``, ``candidates``, and ``to_dict()`` (issue #409 finding #1)."""
    from gepa.core.result import GEPAResult
    from gepa.oa.engine import Result
    from gepa.optimize_anything import optimize_anything

    stashed = GEPAResult(
        candidates=[{"current_candidate": "improved"}],
        parents=[[None]],
        val_aggregate_scores=[0.42],
        val_subscores=[{}],
        per_val_instance_best_candidates={},
        discovery_eval_counts=[1],
        total_metric_calls=1,
        _str_candidate_key="current_candidate",
    )

    def _fake_run_engine(server, engine, *, owns_server):
        return Result(
            best_candidate="improved",
            best_score=0.42,
            total_evals=1,
            metadata={"gepa_result": stashed},
        )

    monkeypatch.setattr("gepa.optimize_anything._run_engine", _fake_run_engine)

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
    )

    assert isinstance(result, GEPAResult)
    assert result is stashed
    assert result.best_candidate == "improved"
    assert result.best_idx == 0
    assert result.candidates == [{"current_candidate": "improved"}]
    assert result.val_aggregate_scores == [0.42]
    assert isinstance(result.to_dict(), dict)


def test_explicit_optimize_anything_config_still_returns_result():
    """An explicit ``OptimizeAnythingConfig`` keeps the new ``Result`` shape."""
    from gepa.oa.engine import Result
    from gepa.optimize_anything import optimize_anything

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
        config=OptimizeAnythingConfig(
            engine="gepa",
            max_evals=4,
            engine_config={"reflection": {"reflection_lm": _FakeLM()}},
        ),
    )

    assert isinstance(result, Result)
    assert result.best_score > 0.0
    assert not hasattr(result, "val_aggregate_scores")


def test_legacy_gepa_config_returns_gepa_result_even_when_budget_dies_early():
    """Budget smaller than the seed's valset pass: the engine has no GEPAResult
    to stash, so the legacy path synthesizes a single-candidate one — legacy
    callers must never see the omni Result."""
    from gepa.core.result import GEPAResult
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        dataset=[1, 2, 3],
        valset=[1, 2, 3, 4, 5],
        objective="maximize length",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=2),
            reflection=ReflectionConfig(reflection_lm=_FakeLM()),
        ),
    )

    assert isinstance(result, GEPAResult)
    assert result.best_candidate
    assert result.val_aggregate_scores[result.best_idx] == max(result.val_aggregate_scores)


def test_legacy_gepa_config_rejects_test_set():
    """``test_set`` needs an explicit OptimizeAnythingConfig; GEPAConfig returns
    ``GEPAResult``, which has no place for held-out test scores."""
    from gepa.optimize_anything import GEPAConfig, optimize_anything

    with pytest.raises(ValueError, match="test_set"):
        optimize_anything(
            seed_candidate="short",
            evaluator=_evaluator,
            test_set=["x"],
            config=GEPAConfig(),
        )


def test_no_config_rejects_test_set():
    """``test_set`` with config omitted also fails fast: that path returns
    ``GEPAResult``, which has no place for held-out test scores."""
    from gepa.optimize_anything import optimize_anything

    with pytest.raises(ValueError, match="test_set"):
        optimize_anything(
            seed_candidate="short",
            evaluator=_evaluator,
            test_set=["x"],
        )


def test_full_config_passthrough_and_reflective_dataset_persistence(tmp_path):
    """End-to-end: tracking flows through, reflective_dataset is persisted by core."""
    from gepa.optimize_anything import optimize_anything

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
        config=OptimizeAnythingConfig(
            engine="gepa",
            max_evals=6,
            run_dir=str(tmp_path),
            engine_config={
                "reflection": {"reflection_lm": _FakeLM()},
                "tracking": TrackingConfig(use_wandb=False, use_mlflow=False),
                "engine": {"write_agent_state": True},
            },
        ),
    )

    assert result.best_score > 0.0
    # reflective_dataset.json is now written by GEPA core under write_agent_state,
    # not by a private optimize_anything callback.
    rds = glob.glob(os.path.join(tmp_path, "iterations", "*", "reflective_dataset.json"))
    assert rds, "core should persist reflective_dataset.json when write_agent_state is on"
