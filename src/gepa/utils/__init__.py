"""Utilities for ``optimize_anything`` evaluators and optimization control.

Re-exports:
    **Stop conditions** — control when optimization terminates:
    ``MaxMetricCallsStopper``, ``TimeoutStopCondition``, ``NoImprovementStopper``,
    ``ScoreThresholdStopper``, ``FileStopper``, ``SignalStopper``, ``CompositeStopper``.

    **Code execution** — safe sandboxed execution for code-evolution evaluators:
    ``execute_code``, ``CodeExecutionResult``, ``ExecutionMode``.

    **Stdio capture** — thread-safe stdout/stderr capture during evaluation:
    ``StreamCaptureManager``, ``ThreadLocalStreamCapture``.

    **Format validation** — evaluator decorators for hard-failing malformed outputs:
    ``require_json_output``, ``require_format``, ``require_regex_match``.
"""

# Code execution utilities for fitness functions that evaluate generated code
from .code_execution import (
    CodeExecutionResult,
    ExecutionMode,
    TimeLimitError,
    execute_code,
    get_code_hash,
)
from .format_validator import require_format, require_json_output, require_regex_match
from .stdio_capture import (
    StreamCaptureManager,
    ThreadLocalStreamCapture,
    stream_manager,
)
from .stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxCandidateProposalsStopper,
    MaxMetricCallsStopper,
    MaxReflectionCostStopper,
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
    "MaxCandidateProposalsStopper",
    "MaxMetricCallsStopper",
    "MaxReflectionCostStopper",
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
    # Format validation utilities
    "require_format",
    "require_json_output",
    "require_regex_match",
    # Stdio capture utilities
    "StreamCaptureManager",
    "ThreadLocalStreamCapture",
    "stream_manager",
]
