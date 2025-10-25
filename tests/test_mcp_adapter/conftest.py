# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Pytest fixtures for MCP adapter tests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {
            "user_query": "What's in the file notes.txt?",
            "tool_arguments": {"path": "/tmp/notes.txt"},
            "reference_answer": "Meeting at 3pm",
            "additional_context": {},
        },
        {
            "user_query": "Show me config.json",
            "tool_arguments": {"path": "/tmp/config.json"},
            "reference_answer": "port: 8080",
            "additional_context": {},
        },
    ]


@pytest.fixture
def seed_candidate():
    """Sample seed candidate."""
    return {
        "tool_description": "Read the contents of a file from the filesystem.",
    }


@pytest.fixture
def simple_metric():
    """Simple metric function for testing."""

    def metric(item, output: str) -> float:
        if item["reference_answer"] and item["reference_answer"].lower() in output.lower():
            return 1.0
        return 0.0

    return metric


@pytest.fixture
def mock_mcp_tool():
    """Mock MCP Tool object."""
    from unittest.mock import MagicMock

    tool = MagicMock()
    tool.name = "read_file"
    tool.description = "Read file contents"
    tool.inputSchema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"],
    }
    return tool


@pytest.fixture
def mock_mcp_session():
    """Mock MCP ClientSession."""
    session = AsyncMock()

    # Mock initialize
    session.initialize = AsyncMock()

    # Mock list_tools
    from unittest.mock import MagicMock

    tools_result = MagicMock()
    tool = MagicMock()
    tool.name = "read_file"
    tool.description = "Read file contents"
    tool.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}
    tools_result.tools = [tool]
    session.list_tools = AsyncMock(return_value=tools_result)

    # Mock call_tool
    tool_result = MagicMock()
    tool_result.content = []

    # Add text content
    text_content = MagicMock()
    text_content.text = "File contents: Meeting at 3pm"
    tool_result.content.append(text_content)

    session.call_tool = AsyncMock(return_value=tool_result)

    return session


@pytest.fixture
def mock_stdio_client(mock_mcp_session):
    """Mock stdio_client context manager."""
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    @asynccontextmanager
    async def mock_client(params):
        # Mock read/write streams
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        yield read_stream, write_stream

    return mock_client


@pytest.fixture
def mock_litellm_completion():
    """Mock litellm completion response."""

    def make_response(text: str):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = text
        return response

    return make_response


@pytest.fixture
def mock_model_callable():
    """Mock model callable for testing."""

    def model(messages):
        # Simple mock that returns JSON tool call
        user_msg = messages[-1]["content"]
        if "file" in user_msg.lower():
            return '{"action": "call_tool", "arguments": {"path": "/tmp/test.txt"}}'
        return '{"action": "answer", "text": "Direct answer"}'

    return model


@pytest.fixture
def server_params():
    """Sample MCP server parameters."""
    try:
        from mcp import StdioServerParameters

        return StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
    except ImportError:
        # If MCP not installed, return a mock
        from unittest.mock import MagicMock

        params = MagicMock()
        params.command = "npx"
        params.args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        return params
