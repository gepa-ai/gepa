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
    from gepa.legacy_optimize_anything import optimize_anything as legacy_optimize_anything
    from gepa.optimize_anything import Engine, OptimizeAnythingConfig, Task, optimize_anything

    assert optimize_anything is not legacy_optimize_anything
    assert Engine is not None
    assert OptimizeAnythingConfig is not None
    assert Task is not None


def test_optimize_anything_legacy_kwargs_are_rejected():
    """The legacy seed_candidate/evaluator route was dropped from the public entry point.

    optimize_anything now takes the task fields directly and builds the Task internally,
    so legacy-style kwargs raise a natural TypeError from arg binding. Callers that still
    need the old API import gepa.legacy_optimize_anything directly.
    """
    from gepa.optimize_anything import optimize_anything

    with pytest.raises(TypeError, match="seed_candidate"):
        optimize_anything(seed_candidate="seed", evaluator=lambda candidate: (1.0, {}))


def test_optimize_anything_missing_required_args_raise():
    """The new signature requires name, initial_candidate, and evaluate as keyword-only
    args; omitting one raises a natural TypeError from arg binding."""
    from gepa.optimize_anything import optimize_anything

    with pytest.raises(TypeError, match="evaluate"):
        optimize_anything(name="t", initial_candidate="x")

    with pytest.raises(TypeError, match="initial_candidate"):
        optimize_anything(name="t", evaluate=lambda candidate: (1.0, {}))
