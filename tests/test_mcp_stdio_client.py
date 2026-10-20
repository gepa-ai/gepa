import asyncio
import sys

import pytest

pytest.importorskip("mcp", reason="MCP SDK not installed")

from gepa.adapters.mcp_adapter.mcp_client import StdioMCPClient


@pytest.fixture
def notification_server(tmp_path):
    script = tmp_path / "notification_server.py"
    script.write_text(
        """import json
import sys

count = int(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    if request["method"] == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0", "id": request["id"], "result": {
                "protocolVersion": request["params"]["protocolVersion"],
                "capabilities": {"logging": {}, "tools": {}},
                "serverInfo": {"name": "notification-test-server", "version": "1.0"}
            }
        }), flush=True)
        continue
    for _ in range(count):
        print(json.dumps({
            "jsonrpc": "2.0", "method": "notifications/message",
            "params": {"level": "info", "data": "tool evaluation in progress"}
        }), flush=True)
    if request["method"] == "tools/call" and request["params"]["name"] == "unknown":
        response = {"error": {"code": -32602, "message": "invalid tool arguments"}}
    elif request["method"] == "tools/list":
        response = {"result": {"tools": [{
            "name": "read_file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}
        }]}}
    else:
        response = {"result": {"content": [{
            "type": "text", "text": request["params"]["arguments"]["path"]
        }]}}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], **response}), flush=True)
""",
        encoding="utf-8",
    )
    return script


@pytest.mark.parametrize("notification_count", [0, 1, 3])
def test_stdio_request_ignores_notifications_and_keeps_responses_in_order(notification_server, notification_count):
    async def exercise():
        client = StdioMCPClient(sys.executable, [str(notification_server), str(notification_count)])
        await client.start()
        try:
            await asyncio.wait_for(client.initialize(), timeout=5)
            first = await asyncio.wait_for(client.send_request("tools/list"), timeout=5)
            assert first["tools"][0]["name"] == "read_file"

            arguments = {"name": "read_file", "arguments": {"path": "notes.txt"}}
            second = await asyncio.wait_for(client.send_request("tools/call", arguments), timeout=5)
            assert second == {"content": [{"type": "text", "text": "notes.txt"}]}
        finally:
            await asyncio.wait_for(client.close(), timeout=5)
        assert client.process.returncode == 0

    asyncio.run(exercise())


def test_stdio_error_after_notification_propagates_and_does_not_poison_next_request(notification_server):
    async def exercise():
        client = StdioMCPClient(sys.executable, [str(notification_server), "1"])
        await client.start()
        try:
            await asyncio.wait_for(client.initialize(), timeout=5)
            with pytest.raises(Exception, match="MCP error.*invalid tool arguments"):
                await asyncio.wait_for(client.call_tool("unknown", {}), timeout=5)
            tools = await asyncio.wait_for(client.list_tools(), timeout=5)
            assert tools[0]["name"] == "read_file"
        finally:
            await asyncio.wait_for(client.close(), timeout=5)
        assert client.process.returncode == 0

    asyncio.run(exercise())
