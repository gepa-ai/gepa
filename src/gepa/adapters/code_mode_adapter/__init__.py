# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Code Mode Adapter for GEPA.

This adapter optimizes Code Mode system text components while delegating
runtime-specific execution to a pluggable runner.

Exports:
    CodeModeAdapter: Main adapter class
    CodeModeDataInst: Dataset item type
    CodeModeOutput: Output type
    CodeModeTrajectory: Execution trace type
    CodeModeRunnerResult: Runner output contract
    StaticCodeModeRunner: In-memory deterministic runner
    HTTPCodeModeRunner: HTTP bridge runner for external runtimes
    MCPStreamableHTTPCodeModeRunner: Direct MCP streamable-http runner
    MCPStdioCodeModeRunner: Local MCP stdio runner
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode_adapter import (
        CodeModeAdapter,
        CodeModeDataInst,
        CodeModeOutput,
        CodeModeRunnerResult,
        CodeModeTrajectory,
    )
    from .runners import (
        HTTPCodeModeRunner,
        MCPStdioCodeModeRunner,
        MCPStreamableHTTPCodeModeRunner,
        StaticCodeModeRunner,
    )

__all__ = [
    "CodeModeAdapter",
    "CodeModeDataInst",
    "CodeModeOutput",
    "CodeModeRunnerResult",
    "CodeModeTrajectory",
    "HTTPCodeModeRunner",
    "MCPStreamableHTTPCodeModeRunner",
    "MCPStdioCodeModeRunner",
    "StaticCodeModeRunner",
]


def __getattr__(name: str):
    if name in {
        "CodeModeAdapter",
        "CodeModeDataInst",
        "CodeModeOutput",
        "CodeModeRunnerResult",
        "CodeModeTrajectory",
        "HTTPCodeModeRunner",
        "MCPStreamableHTTPCodeModeRunner",
        "MCPStdioCodeModeRunner",
        "StaticCodeModeRunner",
    }:
        from .code_mode_adapter import (
            CodeModeAdapter,
            CodeModeDataInst,
            CodeModeOutput,
            CodeModeRunnerResult,
            CodeModeTrajectory,
        )
        from .runners import (
            HTTPCodeModeRunner,
            MCPStdioCodeModeRunner,
            MCPStreamableHTTPCodeModeRunner,
            StaticCodeModeRunner,
        )

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
