"""
Utilities for GEPA optimization.

This module provides:
- Stop conditions for controlling optimization loops
- Code execution utilities for safely running generated code in fitness functions
"""

# Code execution utilities for fitness functions that evaluate generated code
from .code_execution import (
    CodeExecutionResult,
    ExecutionMode,
    TimeLimitError,
    execute_code,
    get_code_hash,
)
from .stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxMetricCallsStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)

__all__ = [
    # Stop conditions
    "CompositeStopper",
    "FileStopper",
    "MaxMetricCallsStopper",
    "NoImprovementStopper",
    "ScoreThresholdStopper",
    "SignalStopper",
    "StopperProtocol",
    "TimeoutStopCondition",
    # Code execution utilities
    "CodeExecutionResult",
    "ExecutionMode",
    "TimeLimitError",
    "execute_code",
    "get_code_hash",
]
