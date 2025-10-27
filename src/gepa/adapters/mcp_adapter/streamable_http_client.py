"""
StreamableHTTP MCP Client for remote MCP servers with session management.
"""

import logging

logger = logging.getLogger(__name__)


class StreamableHTTPMCPClient:
    """MCP client using StreamableHTTP transport for production remote servers."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30,
        sse_read_timeout: float = 300,
    ):
        """
        Initialize StreamableHTTP MCP client.

        Args:
            url: MCP endpoint URL
            headers: Optional HTTP headers (e.g., authentication)
            timeout: Timeout for HTTP operations
            sse_read_timeout: Timeout for SSE streaming reads
        """
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.request_id = 0
        self.read_stream = None
        self.write_stream = None
        self._transport_context = None

    async def start(self):
        """Start the StreamableHTTP connection to remote MCP server."""
        from mcp.client.streamable_http import streamable_http_client

        logger.info(f"Connecting to StreamableHTTP MCP server at {self.url}")

        # Create StreamableHTTP connection
        self._transport_context = streamable_http_client(
            url=self.url,
            headers=self.headers,
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
        )

        # Enter context and get streams
        streams = await self._transport_context.__aenter__()
        self.read_stream, self.write_stream = streams

        logger.info("StreamableHTTP connection established")

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

        # Only include params if provided
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
        """Close the StreamableHTTP connection."""
        if self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
                logger.info("StreamableHTTP connection closed")
            except Exception as e:
                logger.warning(f"Error closing StreamableHTTP connection: {e}")
