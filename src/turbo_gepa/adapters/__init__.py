"""
Adapter helpers for running TurboGEPA on common task setups.
"""

from .default_adapter import DefaultAdapter, DefaultDataInst

# DSPy adapter is optional - only import if dspy is available
try:
    from .dspy_adapter import DSpyAdapter, ScoreWithFeedback
except Exception:  # pragma: no cover - optional dependency failures
    # dspy not installed or failed to initialize - adapter not available
    pass
