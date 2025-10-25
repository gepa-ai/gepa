# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""MCP Adapter for GEPA - Optimize Model Context Protocol tool usage."""

from .mcp_types import MCPDataInst, MCPOutput, MCPTrajectory

__all__ = [
    "MCPAdapter",
    "MCPDataInst",
    "MCPOutput",
    "MCPTrajectory",
]


def __getattr__(name):
    """Lazy import MCPAdapter to handle missing MCP SDK gracefully."""
    if name == "MCPAdapter":
        from .mcp_adapter import MCPAdapter

        return MCPAdapter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
