"""
SSE MCP Client for remote MCP servers using Server-Sent Events transport.
"""

import logging

logger = logging.getLogger(__name__)


class SSEMCPClient:
    """MCP client using SSE transport for remote servers."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 30):
        """
        Initialize SSE MCP client.

        Args:
            url: SSE endpoint URL
            headers: Optional HTTP headers (e.g., authentication)
            timeout: Timeout for HTTP operations
        """
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.request_id = 0
        self.read_stream = None
        self.write_stream = None
        self._sse_context = None

    async def start(self):
        """Start the SSE connection to remote MCP server."""
        from mcp.client.sse import sse_client

        logger.info(f"Connecting to SSE MCP server at {self.url}")

        # Create SSE connection
        self._sse_context = sse_client(
            url=self.url,
            headers=self.headers,
            timeout=self.timeout,
            sse_read_timeout=300,  # 5 minutes for long-running operations
        )

        # Enter context and get streams
        streams = await self._sse_context.__aenter__()
        self.read_stream, self.write_stream = streams

        logger.info("SSE connection established")

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        """
        Send JSON-RPC request and get response.

        Args:
            method: JSON-RPC method name
            params: Optional parameters

        Returns:
            Response result dict
        """
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCMessage, JSONRPCRequest

        self.request_id += 1
        request_dict = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id,
        }

        # Only include params if provided (per MCP spec)
        if params is not None:
            request_dict["params"] = params

        logger.debug(f"Sending request: {method} (id={self.request_id})")

        # Create proper JSONRPCRequest object
        request = JSONRPCRequest(**request_dict)

        # Send via write stream
        session_message = SessionMessage(message=JSONRPCMessage(request))
        await self.write_stream.send(session_message)

        # Read response from read stream
        response_message = await self.read_stream.receive()

        # Extract result
        if hasattr(response_message.message.root, "error"):
            error = response_message.message.root.error
            raise Exception(f"MCP error: {error}")

        if hasattr(response_message.message.root, "result"):
            return response_message.message.root.result

        raise Exception(f"Unexpected response format: {response_message}")

    async def initialize(self) -> dict:
        """Initialize MCP session."""
        result = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "gepa-mcp-adapter", "version": "1.0"},
            },
        )

        # Send initialized notification (required by MCP protocol)
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCMessage, JSONRPCNotification

        notification = JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/initialized",
        )

        session_message = SessionMessage(message=JSONRPCMessage(notification))
        await self.write_stream.send(session_message)

        logger.info(f"Session initialized: {result.get('serverInfo', {}).get('name', 'unknown')}")
        return result

    async def list_tools(self) -> list[dict]:
        """List available tools."""
        result = await self.send_request("tools/list")
        tools = result.get("tools", [])
        logger.debug(f"Listed {len(tools)} tools")
        return tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Call a tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result dict
        """
        logger.debug(f"Calling tool: {name} with args: {arguments}")
        result = await self.send_request("tools/call", {"name": name, "arguments": arguments})
        return result

    async def close(self):
        """Close the SSE connection."""
        if self._sse_context:
            try:
                await self._sse_context.__aexit__(None, None, None)
                logger.info("SSE connection closed")
            except Exception as e:
                logger.warning(f"Error closing SSE connection: {e}")
