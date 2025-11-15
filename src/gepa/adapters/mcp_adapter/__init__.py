# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
MCP Adapter for GEPA.

This adapter enables optimization of MCP tool descriptions and system prompts
using GEPA's iterative refinement approach.

Exports:
    MCPAdapter: Main adapter class
    MCPDataInst: Dataset item type
    MCPTrajectory: Execution trace type
    MCPOutput: Output type
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mcp_adapter import MCPAdapter, MCPDataInst, MCPOutput, MCPTrajectory

__all__ = [
    "MCPAdapter",
    "MCPDataInst",
    "MCPOutput",
    "MCPTrajectory",
]


def __getattr__(name: str):
    """Lazy import to handle missing MCP SDK gracefully."""
    if name in {"MCPAdapter", "MCPDataInst", "MCPOutput", "MCPTrajectory"}:
        from .mcp_adapter import MCPAdapter, MCPDataInst, MCPOutput, MCPTrajectory

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
