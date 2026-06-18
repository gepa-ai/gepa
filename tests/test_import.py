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


def test_optimize_anything_legacy_call_falls_back(monkeypatch):
    """Legacy seed_candidate/evaluator calls transparently dispatch to the legacy API.

    Instead of raising, the public wrapper emits a DeprecationWarning and forwards
    the call verbatim to gepa.legacy_optimize_anything so existing code keeps working.
    """
    import gepa.legacy_optimize_anything as legacy_mod
    from gepa.optimize_anything import optimize_anything

    recorded = {}
    sentinel = object()

    def fake_legacy(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(legacy_mod, "optimize_anything", fake_legacy)

    def evaluator(candidate):
        return (1.0, {})

    with pytest.deprecated_call(match="legacy_optimize_anything"):
        result = optimize_anything(seed_candidate="seed", evaluator=evaluator)

    assert result is sentinel
    assert recorded["args"] == ()
    assert recorded["kwargs"] == {"seed_candidate": "seed", "evaluator": evaluator}


def test_legacy_api_args_match_legacy_signature():
    """_LEGACY_API_ARGS is hardcoded (to keep the legacy import lazy); guard it
    against drift by deriving the legacy parameters via inspect here."""
    import inspect

    from gepa.legacy_optimize_anything import optimize_anything as legacy_optimize_anything
    from gepa.optimize_anything import _LEGACY_API_ARGS

    legacy_params = set(inspect.signature(legacy_optimize_anything).parameters)
    # `config` is shared with the new API and intentionally excluded from the set.
    assert _LEGACY_API_ARGS == legacy_params - {"config"}


def test_optimize_anything_mixed_api_call_is_rejected():
    """Mixing new-API (task/evaluate) and legacy (seed_candidate/evaluator) args is rejected."""
    from gepa.optimize_anything import Task, optimize_anything

    with pytest.raises(TypeError, match="cannot mix"):
        optimize_anything(
            task=Task(name="t", initial_candidate="x", objective="o"),
            evaluator=lambda candidate: (1.0, {}),
        )


def test_optimize_anything_new_call_missing_evaluate_raises():
    """A new-API call missing evaluate raises a natural TypeError from arg binding."""
    from gepa.optimize_anything import optimize_anything

    with pytest.raises(TypeError, match="evaluate"):
        optimize_anything(task="not a Task")
