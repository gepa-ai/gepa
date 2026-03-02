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
    from .runners import MCPStdioCodeModeRunner, MCPStreamableHTTPCodeModeRunner

__all__ = [
    "CodeModeAdapter",
    "CodeModeDataInst",
    "CodeModeOutput",
    "CodeModeRunnerResult",
    "CodeModeTrajectory",
    "MCPStreamableHTTPCodeModeRunner",
    "MCPStdioCodeModeRunner",
]


def __getattr__(name: str):
    if name in {
        "CodeModeAdapter",
        "CodeModeDataInst",
        "CodeModeOutput",
        "CodeModeRunnerResult",
        "CodeModeTrajectory",
        "MCPStreamableHTTPCodeModeRunner",
        "MCPStdioCodeModeRunner",
    }:
        from .code_mode_adapter import (
            CodeModeAdapter,
            CodeModeDataInst,
            CodeModeOutput,
            CodeModeRunnerResult,
            CodeModeTrajectory,
        )
        from .runners import MCPStdioCodeModeRunner, MCPStreamableHTTPCodeModeRunner

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
