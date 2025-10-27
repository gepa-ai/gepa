"""
Adapter helpers for running TurboGEPA on common task setups.
"""

from .default_adapter import DefaultAdapter, DefaultDataInst

# DSPy adapter is optional - only import if dspy is available
try:
    from .dspy_adapter import DSpyAdapter, ScoreWithFeedback
except ImportError:
    # dspy not installed - adapter not available
    pass
