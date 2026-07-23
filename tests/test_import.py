import pytest


def test_package_import():
    """
    Ensures the 'gepa' package can be imported.
    """
    import gepa

    assert gepa is not None


def test_gepa_optimize_import():
    """
    Ensures the 'gepa.optimize' function can be imported.
    """
    from gepa import optimize

    assert optimize is not None


def test_optimize_anything_imports():
    """
    Ensures the public optimize_anything API and archived legacy API can be imported.
    """
    from gepa.gepa_launcher import optimize_anything as gepa_launcher
    from gepa.optimize_anything import Engine, OptimizeAnythingConfig, Task, optimize_anything

    assert optimize_anything is not gepa_launcher
    assert Engine is not None
    assert OptimizeAnythingConfig is not None
    assert Task is not None


def test_legacy_launcher_names_importable_from_optimize_anything():
    """Old (v0.1.x) examples import the launcher API from ``gepa.optimize_anything``;
    the module re-exports that surface so old imports keep working unmodified."""
    import gepa.gepa_launcher as launcher
    from gepa.optimize_anything import (
        Candidate,
        EngineConfig,
        Evaluator,
        GEPAConfig,
        GEPAResult,
        LogContext,
        MergeConfig,
        OptimizationState,
        RefinerConfig,
        ReflectionConfig,
        SideInfo,
        TrackingConfig,
        get_log_context,
        log,
        make_litellm_lm,
        set_log_context,
    )

    for obj in (
        Candidate,
        EngineConfig,
        Evaluator,
        GEPAConfig,
        GEPAResult,
        LogContext,
        MergeConfig,
        OptimizationState,
        RefinerConfig,
        ReflectionConfig,
        SideInfo,
        TrackingConfig,
        get_log_context,
        log,
        make_litellm_lm,
        set_log_context,
    ):
        assert obj is not None

    # Same objects, not copies — the module re-exports the launcher's surface.
    assert GEPAConfig is launcher.GEPAConfig
    assert log is launcher.log


def test_optimize_anything_old_kwargs_are_rejected():
    """optimize_anything mirrors the legacy seed_candidate/evaluator shape, so the
    intermediate names (evaluate, initial_candidate) and the now-config-only ``name``
    raise a natural TypeError from arg binding."""
    from gepa.optimize_anything import optimize_anything

    with pytest.raises(TypeError, match="evaluate"):
        optimize_anything(seed_candidate="seed", evaluate=lambda candidate: (1.0, {}))

    with pytest.raises(TypeError, match="initial_candidate"):
        optimize_anything(initial_candidate="seed", evaluator=lambda candidate: (1.0, {}))

    with pytest.raises(TypeError, match="name"):
        optimize_anything(seed_candidate="seed", evaluator=lambda candidate: (1.0, {}), name="t")


def test_optimize_anything_requires_evaluator():
    """evaluator (or the legacy batch_evaluator) is the only required arg;
    seed_candidate is an optional leading positional (None = seedless) and the
    run name lives on OptimizeAnythingConfig."""
    from gepa.optimize_anything import optimize_anything

    with pytest.raises(ValueError, match="evaluator"):
        optimize_anything("x")

    with pytest.raises(TypeError, match="name"):
        optimize_anything(name="t", evaluate=lambda candidate: (1.0, {}))
