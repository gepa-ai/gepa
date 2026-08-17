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
    engine = GepaEngine(OptimizeAnythingConfig(engine="gepa", max_evals=4, engine_config={"tracking": tracking}))
    assert engine.gepa_config.tracking.key_prefix == "gepa/"


def test_unknown_key_fails_fast():
    """A misspelled engine_config key raises TypeError at construction."""
    with pytest.raises(TypeError):
        GepaEngine(OptimizeAnythingConfig(engine="gepa", max_evals=4, engine_config={"trackign": TrackingConfig()}))


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


def test_legacy_max_workers_sizes_the_eval_server():
    """``EngineConfig.max_workers`` becomes the eval server's
    ``max_concurrency``; without the mapping every legacy run is silently
    gated at the server default regardless of the requested width."""
    from gepa.optimize_anything import EngineConfig, GEPAConfig, _from_legacy_config

    converted = _from_legacy_config(GEPAConfig(engine=EngineConfig(max_workers=512)))
    assert converted.max_concurrency == 512


def test_legacy_parallel_false_maps_to_sequential_eval_server():
    """``parallel=False`` meant sequential evaluation in the launcher."""
    from gepa.optimize_anything import EngineConfig, GEPAConfig, _from_legacy_config

    converted = _from_legacy_config(GEPAConfig(engine=EngineConfig(parallel=False, max_workers=512)))
    assert converted.max_concurrency == 1


def test_legacy_max_workers_none_falls_back_to_cpu_count():
    """``max_workers=None`` mirrors the field's own default factory."""
    import os

    from gepa.optimize_anything import EngineConfig, GEPAConfig, _from_legacy_config

    converted = _from_legacy_config(GEPAConfig(engine=EngineConfig(max_workers=None)))
    assert converted.max_concurrency == (os.cpu_count() or 32)


def test_legacy_gepa_config_accepts_test_set():
    """With the single unified GEPAResult, ``test_set`` works on every path —
    including a legacy ``GEPAConfig`` — because the result now carries a
    ``metadata`` dict for the held-out scores instead of rejecting them."""
    from gepa.core.result import GEPAResult
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
        test_set=["held-out-a", "held-out-b"],
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=4),
            reflection=ReflectionConfig(reflection_lm=_FakeLM()),
        ),
    )

    assert isinstance(result, GEPAResult)
    assert "test_score" in result.metadata
    assert "baseline_test_score" in result.metadata
    assert "gepa_result" not in result.metadata


def test_no_config_returns_unified_gepa_result(monkeypatch):
    """``optimize_anything(seed, evaluator=fn)`` with config omitted returns a
    GEPAResult exposing both the lean accessors and the pool surface
    (``candidates`` / ``val_aggregate_scores`` / ``to_dict()``) — issue #409."""
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
            total_evals=9,
            eval_log=[{"score": 0.42}],
            metadata={"gepa_result": stashed, "engine": "gepa"},
        )

    monkeypatch.setattr("gepa.optimize_anything._run_engine", _fake_run_engine)

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
    )

    assert isinstance(result, GEPAResult)
    assert result.best_candidate == "improved"
    assert result.best_idx == 0
    assert result.candidates == [{"current_candidate": "improved"}]
    assert result.val_aggregate_scores == [0.42]
    # Lean total_evals is the eval-server count, not GEPA core's counter.
    assert result.total_evals == 9
    assert result.total_metric_calls == 1
    assert result.eval_log == [{"score": 0.42}]
    assert result.metadata["engine"] == "gepa"
    assert "gepa_result" not in result.metadata
    serialized = result.to_dict()
    assert isinstance(serialized, dict)
    assert "eval_log" not in serialized
    assert "metadata" not in serialized
    assert "eval_server_calls" not in serialized
    assert serialized["total_metric_calls"] == 1


def test_no_config_accepts_test_set(monkeypatch):
    """``test_set`` with config omitted is accepted; held-out scores land in
    ``metadata`` on the unified GEPAResult."""
    from gepa.core.result import GEPAResult
    from gepa.oa.engine import Result
    from gepa.optimize_anything import optimize_anything

    def _fake_run_engine(server, engine, *, owns_server):
        return Result(
            best_candidate="improved",
            best_score=0.5,
            total_evals=2,
            metadata={"test_score": 0.4, "baseline_test_score": 0.2},
        )

    monkeypatch.setattr("gepa.optimize_anything._run_engine", _fake_run_engine)

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
        test_set=["held-out"],
    )

    assert isinstance(result, GEPAResult)
    assert result.metadata["test_score"] == 0.4
    assert result.metadata["baseline_test_score"] == 0.2


def test_explicit_optimize_anything_config_returns_unified_gepa_result():
    """An explicit ``OptimizeAnythingConfig`` returns the same unified
    GEPAResult as the no-config path: lean accessors plus the pool surface."""
    from gepa.core.result import GEPAResult
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

    assert isinstance(result, GEPAResult)
    assert result.best_candidate
    assert result.best_score > 0.0
    assert result.total_evals > 0
    assert isinstance(result.metadata, dict)
    assert "gepa_result" not in result.metadata
    assert result.best_idx >= 0
    assert len(result.candidates) >= 1
    assert result.val_aggregate_scores
    assert isinstance(result.to_dict(), dict)


def test_non_gepa_engine_returns_single_candidate_gepa_result(monkeypatch):
    """A generic engine (no candidate pool stashed) is projected onto a
    single-candidate GEPAResult: lean core populated from its Result, eval log
    and metadata attached, gepa-only pool fields left None/empty."""
    from gepa.core.result import GEPAResult
    from gepa.oa.engine import Result
    from gepa.optimize_anything import optimize_anything

    def _fake_run_engine(server, engine, *, owns_server):
        return Result(
            best_candidate="a much longer improved candidate",
            best_score=0.31,
            total_evals=3,
            eval_log=[{"score": 0.31}],
            metadata={"engine": "best_of_n", "wall_time": 0.01},
        )

    monkeypatch.setattr("gepa.optimize_anything._run_engine", _fake_run_engine)

    result = optimize_anything(
        seed_candidate="short",
        evaluator=_evaluator,
        objective="maximize length",
        config=OptimizeAnythingConfig(engine="best_of_n", max_evals=3),
    )

    assert isinstance(result, GEPAResult)
    assert result.best_candidate == "a much longer improved candidate"
    assert result.best_score == 0.31
    assert result.total_evals == 3
    assert result.eval_log == [{"score": 0.31}]
    assert result.metadata["engine"] == "best_of_n"
    assert len(result.candidates) == 1
    # gepa-only richness is absent on a generic run.
    assert result.per_val_instance_best_candidates == {}
    assert result.val_aggregate_subscores is None
    assert "gepa_result" not in result.metadata
    serialized = result.to_dict()
    assert isinstance(serialized, dict)
    assert "eval_log" not in serialized
    assert "metadata" not in serialized
    assert "eval_server_calls" not in serialized


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
