"""
Adapter helpers for running TurboGEPA on common task setups.
"""

from .default_adapter import DefaultAdapter, DefaultDataInst  # noqa: F401

# DSPy adapter is optional - only import if dspy is available
try:
    from .dspy_adapter import DSpyAdapter, ScoreWithFeedback  # noqa: F401
except ImportError:
    # dspy not installed - adapter not available
    pass
