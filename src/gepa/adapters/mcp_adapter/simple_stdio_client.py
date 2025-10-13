"""
Simple stdio MCP client that bypasses the broken SDK stdio_client.
"""

import asyncio
import json


class SimpleStdioMCPClient:
    """Minimal MCP client using direct subprocess communication."""

    def __init__(self, command: str, args: list[str]):
        self.command = command
        self.args = args
        self.process = None
        self.request_id = 0

    async def start(self):
        """Start the MCP server process."""
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        """Send JSON-RPC request and get response."""
        self.request_id += 1
        request = {"jsonrpc": "2.0", "method": method, "id": self.request_id}

        # Only include params if provided
        if params is not None:
            request["params"] = params

        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())

        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")

        return response.get("result", {})

    async def initialize(self) -> dict:
        """Initialize MCP session."""
        result = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2025-10-10",
                "capabilities": {},
                "clientInfo": {"name": "gepa-mcp-adapter", "version": "1.0"},
            },
        )

        # Send initialized notification (required by MCP protocol)
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        await self.process.stdin.drain()

        return result

    async def list_tools(self) -> list[dict]:
        """List available tools."""
        result = await self.send_request("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool."""
        result = await self.send_request("tools/call", {"name": name, "arguments": arguments})
        return result

    async def close(self):
        """Close the connection."""
        if self.process:
            self.process.stdin.close()
            await self.process.wait()
