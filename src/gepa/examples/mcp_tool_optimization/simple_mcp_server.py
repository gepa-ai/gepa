#!/usr/bin/env python3
"""
Simple MCP server that provides file system tools.
This is a minimal working implementation for the multi-tool example.
"""

import asyncio
import sys
from pathlib import Path

from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


def create_server(allowed_directory: str):
    """Create MCP server with file system tools."""
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
            ),
            Tool(
                name="write_file",
                description="Write content to a file in the allowed directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write (relative or absolute within allowed directory)",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        }
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                name="list_files",
                description="List files and directories in the allowed directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to list (relative or absolute within allowed directory)",
                        }
                    },
                    "required": ["path"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name not in ["read_file", "write_file", "list_files"]:
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

        if name == "read_file":
            # Read the file
            if not requested_path.exists():
                raise FileNotFoundError(f"File not found: {requested_path}")

            if not requested_path.is_file():
                raise ValueError(f"Not a file: {requested_path}")

            content = requested_path.read_text()
            return [TextContent(type="text", text=content)]

        elif name == "write_file":
            # Write to the file
            content = arguments.get("content", "")
            requested_path.parent.mkdir(parents=True, exist_ok=True)
            requested_path.write_text(content)
            return [TextContent(type="text", text=f"Successfully wrote {len(content)} characters to {requested_path.name}")]

        elif name == "list_files":
            # List files in directory
            if not requested_path.exists():
                raise FileNotFoundError(f"Directory not found: {requested_path}")

            if not requested_path.is_dir():
                raise ValueError(f"Not a directory: {requested_path}")

            files = []
            for item in requested_path.iterdir():
                if item.is_file():
                    files.append(f"📄 {item.name}")
                elif item.is_dir():
                    files.append(f"📁 {item.name}/")

            if not files:
                return [TextContent(type="text", text="Directory is empty")]
            else:
                return [TextContent(type="text", text="\n".join(sorted(files)))]

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
