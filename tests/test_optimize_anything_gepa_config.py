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
