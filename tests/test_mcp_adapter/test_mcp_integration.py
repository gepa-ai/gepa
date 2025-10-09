# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Integration tests for MCP adapter.

These tests require MCP SDK to be installed but use mocked models
to avoid external API calls.
"""

import pytest

# Skip all tests if MCP not installed
pytest.importorskip("mcp", reason="MCP SDK not installed")

from gepa.adapters.mcp_adapter import MCPAdapter, MCPDataInst


@pytest.mark.integration
class TestMCPAdapterIntegration:
    """Integration tests for MCP adapter."""

    def test_adapter_imports_successfully(self):
        """Test that adapter can be imported without errors."""
        from gepa.adapters.mcp_adapter import MCPAdapter

        assert MCPAdapter is not None

    def test_adapter_initialization_with_real_types(self):
        """Test adapter initialization with real MCP types."""
        from mcp import StdioServerParameters

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )

        def simple_metric(item, output):
            return 1.0 if item["reference_answer"] in output else 0.0

        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        assert adapter.server_params == server_params
        assert adapter.tool_name == "read_file"

    def test_dataset_type_compatibility(self):
        """Test that MCPDataInst is compatible with adapter."""
        dataset = [
            MCPDataInst(
                user_query="Test query",
                tool_arguments={"path": "/tmp/test.txt"},
                reference_answer="Expected answer",
                additional_context={},
            )
        ]

        assert len(dataset) == 1
        assert dataset[0]["user_query"] == "Test query"

    def test_candidate_structure(self):
        """Test that candidate structure is correct."""
        candidate = {
            "tool_description": "Read file contents from filesystem",
        }

        # This should be the correct structure for MCP adapter
        assert "tool_description" in candidate
        assert isinstance(candidate["tool_description"], str)

    def test_multiple_component_candidate(self):
        """Test candidate with multiple components."""
        candidate = {
            "tool_description": "Read file contents",
            "system_prompt": "You are a helpful file assistant.",
        }

        assert len(candidate) == 2
        assert "tool_description" in candidate
        assert "system_prompt" in candidate


@pytest.mark.integration
@pytest.mark.skipif(
    True,  # Always skip by default, enable manually for real testing
    reason="Requires MCP server to be running - enable manually",
)
class TestMCPAdapterWithRealServer:
    """
    Tests that require a real MCP server.

    These are skipped by default. To run them:
    1. Ensure npx and node are installed
    2. Run: pytest tests/test_mcp_adapter/test_mcp_integration.py -v -m integration --deselect="TestMCPAdapterWithRealServer"
    """

    def test_real_evaluation_flow(self):
        """
        Test real evaluation with actual MCP server.

        This test is skipped by default to avoid requiring MCP server setup.
        """
        import tempfile
        from pathlib import Path

        from mcp import StdioServerParameters

        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_dir],
        )

        def mock_model(messages):
            # Simple mock that calls the tool
            return '{"action": "call_tool", "arguments": {"path": "test.txt"}}'

        def simple_metric(item, output):
            return 1.0 if "Test content" in output else 0.0

        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model=mock_model,
            metric_fn=simple_metric,
        )

        dataset = [
            MCPDataInst(
                user_query="What's in test.txt?",
                tool_arguments={"path": "test.txt"},
                reference_answer="Test content",
                additional_context={},
            )
        ]

        candidate = {
            "tool_description": "Read file contents",
        }

        result = adapter.evaluate(
            batch=dataset,
            candidate=candidate,
            capture_traces=True,
        )

        assert len(result.outputs) == 1
        assert len(result.scores) == 1


def test_readme_example_structure():
    """Test that README example structure is valid."""
    # This validates the example from README without running it
    from mcp import StdioServerParameters

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    dataset = [
        {
            "user_query": "What's in the file notes.txt?",
            "tool_arguments": {"path": "/tmp/notes.txt"},
            "reference_answer": "Meeting at 3pm",
            "additional_context": {},
        },
    ]

    def metric_fn(item, output):
        return 1.0 if item["reference_answer"] in output else 0.0

    # Just test that we can create the adapter
    adapter = MCPAdapter(
        server_params=server_params,
        tool_name="read_file",
        task_model="ollama/llama3.2:1b",
        metric_fn=metric_fn,
    )

    assert adapter is not None
    assert adapter.tool_name == "read_file"
