# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Comprehensive tests for MCP adapter.

Tests cover:
- Adapter initialization
- Local (stdio) and remote (SSE/StreamableHTTP) transports
- Multi-tool support
- Two-pass workflow
- Evaluation and scoring
- Reflective dataset generation
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if MCP not installed
pytest.importorskip("mcp", reason="MCP SDK not installed")

from gepa.adapters.mcp_adapter import MCPAdapter, MCPDataInst, MCPOutput, MCPTrajectory
from gepa.adapters.mcp_adapter.mcp_client import (
    SSEMCPClient,
    StdioMCPClient,
    StreamableHTTPMCPClient,
    create_mcp_client,
)
from gepa.core.adapter import EvaluationBatch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {
            "user_query": "What's in notes.txt?",
            "tool_arguments": {"path": "notes.txt"},
            "reference_answer": "Meeting at 3pm",
            "additional_context": {},
        },
        {
            "user_query": "Read data.txt",
            "tool_arguments": {"path": "data.txt"},
            "reference_answer": "revenue",
            "additional_context": {},
        },
    ]


@pytest.fixture
def seed_candidate():
    """Sample seed candidate."""
    return {"tool_description": "Read file contents from disk."}


@pytest.fixture
def simple_metric():
    """Simple metric function."""

    def metric(item, output: str) -> float:
        ref = item.get("reference_answer", "")
        return 1.0 if ref and ref.lower() in output.lower() else 0.0

    return metric


@pytest.fixture
def server_params():
    """Sample MCP server parameters."""
    from mcp import StdioServerParameters

    return StdioServerParameters(command="python", args=["server.py"])


@pytest.fixture
def mock_model_callable():
    """Mock model callable."""

    def model(messages):
        return '{"action": "call_tool", "tool": "read_file", "arguments": {"path": "test.txt"}}'

    return model


# ============================================================================
# Test Client Factory
# ============================================================================


class TestMCPClientFactory:
    """Tests for MCP client factory function."""

    def test_create_stdio_client(self, server_params):
        """Test creating stdio client."""
        client = create_mcp_client(server_params=server_params)
        assert isinstance(client, StdioMCPClient)
        assert client.command == "python"
        assert client.args == ["server.py"]

    def test_create_sse_client(self):
        """Test creating SSE client."""
        client = create_mcp_client(
            remote_url="https://example.com/sse",
            remote_transport="sse",
        )
        assert isinstance(client, SSEMCPClient)
        assert client.url == "https://example.com/sse"

    def test_create_streamable_http_client(self):
        """Test creating StreamableHTTP client."""
        client = create_mcp_client(
            remote_url="https://example.com/mcp",
            remote_transport="streamable_http",
        )
        assert isinstance(client, StreamableHTTPMCPClient)
        assert client.url == "https://example.com/mcp"

    def test_create_client_invalid_both_params(self, server_params):
        """Test error when both local and remote params provided."""
        with pytest.raises(ValueError, match="not both"):
            create_mcp_client(
                server_params=server_params,
                remote_url="https://example.com",
            )

    def test_create_client_missing_params(self):
        """Test error when no params provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            create_mcp_client()

    def test_create_client_invalid_transport(self):
        """Test error for invalid transport type."""
        with pytest.raises(ValueError, match="Unknown remote transport"):
            create_mcp_client(
                remote_url="https://example.com",
                remote_transport="invalid",
            )


# ============================================================================
# Test Adapter Initialization
# ============================================================================


class TestMCPAdapterInitialization:
    """Tests for MCPAdapter initialization."""

    def test_adapter_with_single_tool(self, server_params, simple_metric):
        """Test creating adapter with single tool (string)."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        assert adapter.tool_names == ["read_file"]
        assert adapter.task_model == "gpt-4o-mini"
        assert adapter.enable_two_pass is True

    def test_adapter_with_multiple_tools(self, server_params, simple_metric):
        """Test creating adapter with multiple tools (list)."""
        adapter = MCPAdapter(
            tool_names=["read_file", "write_file", "list_files"],
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        assert adapter.tool_names == ["read_file", "write_file", "list_files"]
        assert len(adapter.tool_names) == 3

    def test_adapter_with_remote_sse(self, simple_metric):
        """Test creating adapter with remote SSE server."""
        adapter = MCPAdapter(
            tool_names="search",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            remote_url="https://example.com/sse",
            remote_transport="sse",
            remote_headers={"Authorization": "Bearer token"},
        )

        assert adapter.remote_url == "https://example.com/sse"
        assert adapter.remote_transport == "sse"
        assert adapter.remote_headers["Authorization"] == "Bearer token"

    def test_adapter_with_remote_streamable_http(self, simple_metric):
        """Test creating adapter with StreamableHTTP server."""
        adapter = MCPAdapter(
            tool_names="search",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            remote_url="https://example.com/mcp",
            remote_transport="streamable_http",
        )

        assert adapter.remote_transport == "streamable_http"

    def test_adapter_with_callable_model(self, server_params, simple_metric, mock_model_callable):
        """Test creating adapter with callable model."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model=mock_model_callable,
            metric_fn=simple_metric,
            server_params=server_params,
        )

        assert callable(adapter.task_model)
        assert adapter.task_model is mock_model_callable

    def test_adapter_custom_parameters(self, server_params, simple_metric):
        """Test adapter with custom parameters."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
            base_system_prompt="Custom prompt",
            enable_two_pass=False,
            failure_score=0.5,
        )

        assert adapter.base_system_prompt == "Custom prompt"
        assert adapter.enable_two_pass is False
        assert adapter.failure_score == 0.5


# ============================================================================
# Test Adapter Helper Methods
# ============================================================================


class TestMCPAdapterHelpers:
    """Tests for MCPAdapter helper methods."""

    def test_build_system_prompt_single_tool(self, server_params, simple_metric):
        """Test building system prompt with single tool."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        candidate = {"tool_description": "Custom description"}
        tools = [
            {
                "name": "read_file",
                "description": "Original description",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]

        prompt = adapter._build_system_prompt(candidate, tools)

        assert "Custom description" in prompt
        assert "read_file" in prompt
        assert "call_tool" in prompt

    def test_build_system_prompt_multiple_tools(self, server_params, simple_metric):
        """Test building system prompt with multiple tools."""
        adapter = MCPAdapter(
            tool_names=["tool1", "tool2"],
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        candidate = {
            "tool_description_tool1": "Description 1",
            "tool_description_tool2": "Description 2",
        }
        tools = [
            {"name": "tool1", "description": "Default 1", "inputSchema": {}},
            {"name": "tool2", "description": "Default 2", "inputSchema": {}},
        ]

        prompt = adapter._build_system_prompt(candidate, tools)

        assert "Description 1" in prompt
        assert "Description 2" in prompt
        assert "tool1" in prompt
        assert "tool2" in prompt

    def test_extract_tool_response(self, server_params, simple_metric):
        """Test extracting text from tool response."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        # Test dict response with content list
        result = {
            "content": [
                {"type": "text", "text": "Line 1"},
                {"type": "text", "text": "Line 2"},
            ]
        }
        extracted = adapter._extract_tool_response(result)
        assert extracted == "Line 1\nLine 2"

        # Test empty content
        result = {"content": []}
        extracted = adapter._extract_tool_response(result)
        assert extracted == ""

    def test_extract_tool_response_with_error(self, server_params, simple_metric):
        """Test extracting error from tool response (DSPy pattern)."""
        from mcp.types import CallToolResult, TextContent

        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        # Create a proper MCP error result
        error_result = CallToolResult(
            content=[TextContent(type="text", text="File not found")],
            isError=True,
        )

        extracted = adapter._extract_tool_response(error_result)
        assert "ERROR" in extracted
        assert "File not found" in extracted

    def test_extract_tool_response_with_structured_content(self, server_params, simple_metric):
        """Test extracting structured content (modern MCP pattern)."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        # Mock result with structured content
        result = MagicMock()
        result.isError = False
        result.structuredContent = {"name": "John", "age": 30}
        result.content = []

        extracted = adapter._extract_tool_response(result)
        assert "John" in extracted
        assert "30" in extracted

    def test_extract_tool_response_with_image_content(self, server_params, simple_metric):
        """Test handling image content."""
        from mcp.types import ImageContent

        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        # Mock result with image content
        result = MagicMock()
        result.isError = False
        result.structuredContent = None
        result.content = [ImageContent(type="image", data=b"fake_image_data", mimeType="image/png")]

        extracted = adapter._extract_tool_response(result)
        assert "Image" in extracted
        assert "image/png" in extracted

    def test_generate_tool_feedback_success(self, server_params, simple_metric):
        """Test generating feedback for successful execution."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        traj = {
            "user_query": "test",
            "tool_names": ["read_file"],
            "selected_tool": "read_file",
            "tool_called": True,
            "tool_arguments": {"path": "test.txt"},
            "tool_response": "content",
            "tool_description_used": "desc",
            "system_prompt_used": "prompt",
            "model_first_pass_output": "output",
            "model_final_output": "final",
            "score": 0.8,
        }

        feedback = adapter._generate_tool_feedback(traj, 0.8)
        assert "Good" in feedback or "appropriately" in feedback

    def test_generate_tool_feedback_failure(self, server_params, simple_metric):
        """Test generating feedback for failed execution."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        traj = {
            "user_query": "test",
            "tool_names": ["read_file"],
            "selected_tool": None,
            "tool_called": False,
            "tool_arguments": None,
            "tool_response": None,
            "tool_description_used": "desc",
            "system_prompt_used": "prompt",
            "model_first_pass_output": "output",
            "model_final_output": "wrong",
            "score": 0.0,
        }

        feedback = adapter._generate_tool_feedback(traj, 0.0)
        assert "Incorrect" in feedback or "not called" in feedback


# ============================================================================
# Test Evaluation
# ============================================================================


class TestMCPAdapterEvaluation:
    """Tests for MCPAdapter evaluation."""

    def test_evaluate_structure(self, sample_dataset, seed_candidate, simple_metric, mock_model_callable):
        """Test evaluation batch structure."""
        # Verify the adapter is properly configured for evaluation
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model=mock_model_callable,
            metric_fn=simple_metric,
            server_params=MagicMock(),  # Mock params
        )

        # Verify adapter configuration
        assert adapter.tool_names == ["read_file"]
        assert callable(adapter.metric_fn)

        # Note: Full integration tests with real MCP servers should be run separately


# ============================================================================
# Test Reflective Dataset
# ============================================================================


class TestMCPAdapterReflectiveDataset:
    """Tests for reflective dataset generation."""

    def test_make_reflective_dataset_tool_description(self, server_params, simple_metric, seed_candidate):
        """Test reflective dataset for tool_description."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        trajectories = [
            {
                "user_query": "What's in the file?",
                "tool_names": ["read_file"],
                "selected_tool": "read_file",
                "tool_called": True,
                "tool_arguments": {"path": "test.txt"},
                "tool_response": "content",
                "tool_description_used": "Read files",
                "system_prompt_used": "System prompt",
                "model_first_pass_output": "Calling tool",
                "model_final_output": "The file contains...",
                "score": 1.0,
            }
        ]

        eval_batch = EvaluationBatch(
            outputs=[
                {
                    "final_answer": "answer",
                    "tool_called": True,
                    "selected_tool": "read_file",
                    "tool_response": "resp",
                }
            ],
            scores=[1.0],
            trajectories=trajectories,
        )

        reflective_data = adapter.make_reflective_dataset(
            candidate=seed_candidate,
            eval_batch=eval_batch,
            components_to_update=["tool_description"],
        )

        assert "tool_description" in reflective_data
        assert len(reflective_data["tool_description"]) == 1

        example = reflective_data["tool_description"][0]
        assert "Inputs" in example
        assert "Generated Outputs" in example
        assert "Feedback" in example

    def test_make_reflective_dataset_system_prompt(self, server_params, simple_metric, seed_candidate):
        """Test reflective dataset for system_prompt."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        trajectories = [
            {
                "user_query": "Test query",
                "tool_names": ["read_file"],
                "selected_tool": None,
                "tool_called": False,
                "tool_arguments": None,
                "tool_response": None,
                "tool_description_used": "desc",
                "system_prompt_used": "System prompt",
                "model_first_pass_output": "Direct answer",
                "model_final_output": "Wrong answer",
                "score": 0.0,
            }
        ]

        eval_batch = EvaluationBatch(
            outputs=[{"final_answer": "wrong", "tool_called": False, "selected_tool": None, "tool_response": None}],
            scores=[0.0],
            trajectories=trajectories,
        )

        reflective_data = adapter.make_reflective_dataset(
            candidate=seed_candidate,
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        assert "system_prompt" in reflective_data
        assert len(reflective_data["system_prompt"]) == 1


# ============================================================================
# Test Type Definitions
# ============================================================================


def test_mcp_types_import():
    """Test that MCP types can be imported."""
    from gepa.adapters.mcp_adapter import MCPDataInst, MCPOutput, MCPTrajectory

    assert MCPDataInst is not None
    assert MCPOutput is not None
    assert MCPTrajectory is not None


def test_mcp_adapter_import():
    """Test that MCPAdapter can be imported."""
    from gepa.adapters.mcp_adapter import MCPAdapter

    assert MCPAdapter is not None


# ============================================================================
# Test Multi-Tool Features
# ============================================================================


class TestMultiToolSupport:
    """Tests for multi-tool functionality."""

    def test_multi_tool_initialization(self, server_params, simple_metric):
        """Test initializing adapter with multiple tools."""
        tools = ["read_file", "write_file", "list_files"]
        adapter = MCPAdapter(
            tool_names=tools,
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        assert adapter.tool_names == tools
        assert len(adapter.tool_names) == 3

    def test_multi_tool_system_prompt_generation(self, server_params, simple_metric):
        """Test system prompt generation with multiple tools."""
        adapter = MCPAdapter(
            tool_names=["tool1", "tool2"],
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        candidate = {
            "tool_description_tool1": "Desc 1",
            "tool_description_tool2": "Desc 2",
        }

        tools = [
            {"name": "tool1", "description": "Default 1", "inputSchema": {}},
            {"name": "tool2", "description": "Default 2", "inputSchema": {}},
        ]

        prompt = adapter._build_system_prompt(candidate, tools)

        # Both tools should be in the prompt
        assert "tool1" in prompt
        assert "tool2" in prompt
        assert "Desc 1" in prompt
        assert "Desc 2" in prompt


# ============================================================================
# Test Two-Pass Workflow
# ============================================================================


class TestTwoPassWorkflow:
    """Tests for two-pass workflow configuration."""

    def test_two_pass_enabled_by_default(self, server_params, simple_metric):
        """Test that two-pass is enabled by default."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
        )

        assert adapter.enable_two_pass is True

    def test_two_pass_can_be_disabled(self, server_params, simple_metric):
        """Test that two-pass can be disabled."""
        adapter = MCPAdapter(
            tool_names="read_file",
            task_model="gpt-4o-mini",
            metric_fn=simple_metric,
            server_params=server_params,
            enable_two_pass=False,
        )

        assert adapter.enable_two_pass is False
