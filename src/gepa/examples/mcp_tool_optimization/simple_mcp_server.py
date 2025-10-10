#!/usr/bin/env python3
"""
Simple MCP server that provides a read_file tool.
This is a minimal working implementation for the example.
"""

import asyncio
import sys
from pathlib import Path

from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


def create_server(allowed_directory: str):
    """Create MCP server with read_file tool."""
    server = Server("simple-filesystem-server")
    allowed_path = Path(allowed_directory).resolve()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="read_file",
                description="Read the complete contents of a file from the allowed directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (relative or absolute within allowed directory)",
                        }
                    },
                    "required": ["path"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name != "read_file":
            raise ValueError(f"Unknown tool: {name}")

        path_str = arguments.get("path")
        if not path_str:
            raise ValueError("Missing 'path' argument")

        # Resolve the path
        requested_path = Path(path_str)
        if not requested_path.is_absolute():
            requested_path = allowed_path / requested_path

        requested_path = requested_path.resolve()

        # Security check: ensure path is within allowed directory
        try:
            requested_path.relative_to(allowed_path)
        except ValueError:
            raise ValueError(f"Access denied: {requested_path} is outside allowed directory")

        # Read the file
        if not requested_path.exists():
            raise FileNotFoundError(f"File not found: {requested_path}")

        if not requested_path.is_file():
            raise ValueError(f"Not a file: {requested_path}")

        content = requested_path.read_text()
        return [TextContent(type="text", text=content)]

    return server


async def main():
    """Run the MCP server."""
    if len(sys.argv) < 2:
        print("Usage: simple_mcp_server.py <allowed_directory>", file=sys.stderr)
        sys.exit(1)

    allowed_directory = sys.argv[1]
    server = create_server(allowed_directory)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="simple-filesystem-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
