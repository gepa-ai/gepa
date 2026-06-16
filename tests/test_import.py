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


def test_optimize_anything_legacy_call_shows_migration(capsys):
    from gepa.optimize_anything import optimize_anything

    with pytest.deprecated_call(match="legacy_optimize_anything"):
        with pytest.raises(TypeError, match="legacy_optimize_anything"):
            optimize_anything(seed_candidate="seed", evaluator=lambda candidate: (1.0, {}))

    captured = capsys.readouterr()
    assert "from gepa.legacy_optimize_anything import" in captured.err
    assert "seed_candidate" in captured.err
    assert "Task(" in captured.err


def test_optimize_anything_invalid_new_call_shows_usage(capsys):
    from gepa.optimize_anything import optimize_anything

    with pytest.raises(TypeError, match="missing required argument: evaluate"):
        optimize_anything(task="not a Task")

    captured = capsys.readouterr()
    assert "from gepa.optimize_anything import OptimizeAnythingConfig, Task, optimize_anything" in captured.err
    assert "from gepa.legacy_optimize_anything import" in captured.err
