def test_package_import():
    """
    Ensures the 'gepa' package can be imported.
    """
    try:
        import gepa
    except ImportError as e:
        assert False, f"Failed to import the 'gepa' package: {e}"


def test_gepa_optimize_import():
    """
    Ensures the 'gepa.optimize' function can be imported.
    """
    try:
        from gepa import optimize
    except ImportError as e:
        assert False, f"Failed to import the 'gepa.optimize' function: {e}"
