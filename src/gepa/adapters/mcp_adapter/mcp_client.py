"""
Unified MCP Client - Supports stdio, SSE, and StreamableHTTP transports.

This utility provides a single abstraction for connecting to MCP servers
using different transport mechanisms.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseMCPClient(ABC):
    """Abstract base class for MCP clients."""

    def __init__(self):
        self.request_id = 0

    @abstractmethod
    async def start(self):
        """Start the MCP connection."""
        pass

    @abstractmethod
    async def send_request(self, method: str, params: dict | None = None) -> dict:
        """Send JSON-RPC request and get response."""
        pass

    @abstractmethod
    async def close(self):
        """Close the connection."""
        pass

    async def initialize(self) -> dict:
        """Initialize MCP session (common across all transports)."""
        result = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "gepa-mcp-adapter", "version": "1.0"},
            },
        )
        await self._send_initialized_notification()
        return result

    @abstractmethod
    async def _send_initialized_notification(self):
        """Send initialized notification (transport-specific)."""
        pass

    async def list_tools(self) -> list[dict]:
        """List available tools."""
        result = await self.send_request("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool."""
        return await self.send_request("tools/call", {"name": name, "arguments": arguments})


class StdioMCPClient(BaseMCPClient):
    """MCP client using stdio transport (subprocess-based)."""

    def __init__(self, command: str, args: list[str]):
        super().__init__()
        self.command = command
        self.args = args
        self.process = None

    async def start(self):
        """Start the MCP server process."""
        logger.info(f"Starting stdio MCP server: {self.command} {' '.join(self.args)}")
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        """Send JSON-RPC request via stdio."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Process not started or streams not available")

        self.request_id += 1
        request = {"jsonrpc": "2.0", "method": method, "id": self.request_id}

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

    async def _send_initialized_notification(self):
        """Send initialized notification via stdio."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not started or stdin not available")

        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        await self.process.stdin.drain()

    async def close(self):
        """Close the subprocess."""
        if self.process and self.process.stdin:
            self.process.stdin.close()
            await self.process.wait()
            logger.info("Stdio MCP connection closed")


class SSEMCPClient(BaseMCPClient):
    """MCP client using Server-Sent Events transport."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 30):
        super().__init__()
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.read_stream = None
        self.write_stream = None
        self._sse_context = None

    async def start(self):
        """Start the SSE connection."""
        from mcp.client.sse import sse_client  # type: ignore[import-untyped]

        logger.info(f"Connecting to SSE MCP server at {self.url}")

        self._sse_context = sse_client(
            url=self.url,
            headers=self.headers,
            timeout=self.timeout,
            sse_read_timeout=300,
        )

        streams = await self._sse_context.__aenter__()
        self.read_stream, self.write_stream = streams
        logger.info("SSE connection established")

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        """Send JSON-RPC request via SSE."""
        from mcp.shared.message import SessionMessage  # type: ignore[import-untyped]
        from mcp.types import JSONRPCMessage, JSONRPCRequest  # type: ignore[import-untyped]

        if not self.read_stream or not self.write_stream:
            raise RuntimeError("SSE streams not initialized")

        self.request_id += 1
        request_dict = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id,
        }

        if params is not None:
            request_dict["params"] = params

        logger.debug(f"Sending SSE request: {method} (id={self.request_id})")

        request = JSONRPCRequest(**request_dict)
        session_message = SessionMessage(message=JSONRPCMessage(request))
        await self.write_stream.send(session_message)

        # Read response
        response_message = await self.read_stream.receive()

        if hasattr(response_message.message.root, "error"):
            error = response_message.message.root.error
            raise Exception(f"MCP error: {error}")

        if hasattr(response_message.message.root, "result"):
            return response_message.message.root.result

        raise Exception(f"Unexpected response format: {response_message}")

    async def _send_initialized_notification(self):
        """Send initialized notification via SSE."""
        from mcp.shared.message import SessionMessage  # type: ignore[import-untyped]
        from mcp.types import JSONRPCMessage, JSONRPCNotification  # type: ignore[import-untyped]

        if not self.write_stream:
            raise RuntimeError("SSE write stream not initialized")

        notification = JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/initialized",
        )

        session_message = SessionMessage(message=JSONRPCMessage(notification))
        await self.write_stream.send(session_message)

    async def close(self):
        """Close the SSE connection."""
        if self._sse_context:
            try:
                await self._sse_context.__aexit__(None, None, None)
                logger.info("SSE connection closed")
            except Exception as e:
                logger.warning(f"Error closing SSE connection: {e}")


class StreamableHTTPMCPClient(BaseMCPClient):
    """MCP client using StreamableHTTP transport (production-grade)."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30,
        sse_read_timeout: float = 300,
    ):
        super().__init__()
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.read_stream = None
        self.write_stream = None
        self._transport_context = None

    async def start(self):
        """Start the StreamableHTTP connection."""
        from mcp.client.streamable_http import streamable_http_client  # type: ignore[import-untyped]

        logger.info(f"Connecting to StreamableHTTP MCP server at {self.url}")

        self._transport_context = streamable_http_client(
            url=self.url,
            headers=self.headers,
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
        )

        streams = await self._transport_context.__aenter__()
        self.read_stream, self.write_stream = streams
        logger.info("StreamableHTTP connection established")

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        """Send JSON-RPC request via StreamableHTTP."""
        from mcp.shared.message import SessionMessage  # type: ignore[import-untyped]
        from mcp.types import JSONRPCMessage, JSONRPCRequest  # type: ignore[import-untyped]

        if not self.read_stream or not self.write_stream:
            raise RuntimeError("StreamableHTTP streams not initialized")

        self.request_id += 1
        request_dict = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id,
        }

        if params is not None:
            request_dict["params"] = params

        logger.debug(f"Sending StreamableHTTP request: {method} (id={self.request_id})")

        request = JSONRPCRequest(**request_dict)
        session_message = SessionMessage(message=JSONRPCMessage(request))
        await self.write_stream.send(session_message)

        # Read response
        response_message = await self.read_stream.receive()

        if hasattr(response_message.message.root, "error"):
            error = response_message.message.root.error
            raise Exception(f"MCP error: {error}")

        if hasattr(response_message.message.root, "result"):
            return response_message.message.root.result

        raise Exception(f"Unexpected response format: {response_message}")

    async def _send_initialized_notification(self):
        """Send initialized notification via StreamableHTTP."""
        from mcp.shared.message import SessionMessage  # type: ignore[import-untyped]
        from mcp.types import JSONRPCMessage, JSONRPCNotification  # type: ignore[import-untyped]

        if not self.write_stream:
            raise RuntimeError("StreamableHTTP write stream not initialized")

        notification = JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/initialized",
        )

        session_message = SessionMessage(message=JSONRPCMessage(notification))
        await self.write_stream.send(session_message)

    async def close(self):
        """Close the StreamableHTTP connection."""
        if self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
                logger.info("StreamableHTTP connection closed")
            except Exception as e:
                logger.warning(f"Error closing StreamableHTTP connection: {e}")


def create_mcp_client(
    server_params: Any = None,
    remote_url: str | None = None,
    remote_transport: str = "sse",
    remote_headers: dict[str, str] | None = None,
    remote_timeout: float = 30,
    sse_read_timeout: float = 300,
) -> BaseMCPClient:
    """
    Factory function to create the appropriate MCP client.

    Args:
        server_params: StdioServerParameters for local server
        remote_url: URL for remote server
        remote_transport: "sse" or "streamable_http"
        remote_headers: HTTP headers for remote connections
        remote_timeout: Timeout for HTTP operations
        sse_read_timeout: Timeout for SSE streaming

    Returns:
        BaseMCPClient instance (Stdio, SSE, or StreamableHTTP)

    Raises:
        ValueError: If configuration is invalid
    """
    if server_params and remote_url:
        raise ValueError("Provide either server_params (local) or remote_url (remote), not both")
    if not server_params and not remote_url:
        raise ValueError("Must provide either server_params (local) or remote_url (remote)")

    if server_params:
        return StdioMCPClient(command=server_params.command, args=server_params.args)
    elif remote_url:  # Type guard ensures remote_url is not None
        if remote_transport == "sse":
            return SSEMCPClient(url=remote_url, headers=remote_headers, timeout=remote_timeout)
        elif remote_transport == "streamable_http":
            return StreamableHTTPMCPClient(
                url=remote_url,
                headers=remote_headers,
                timeout=remote_timeout,
                sse_read_timeout=sse_read_timeout,
            )
        else:
            raise ValueError(f"Unknown remote transport: {remote_transport}. Must be 'sse' or 'streamable_http'")
    else:
        # This should never happen due to earlier checks
        raise ValueError("Must provide either server_params (local) or remote_url (remote)")
