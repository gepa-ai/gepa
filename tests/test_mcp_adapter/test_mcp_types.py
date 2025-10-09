# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Tests for MCP adapter type definitions.
"""

from gepa.adapters.mcp_adapter.mcp_types import (
    MCPDataInst,
    MCPOutput,
    MCPTrajectory,
)


def test_mcp_data_inst_creation():
    """Test creating an MCPDataInst instance."""
    data_inst = MCPDataInst(
        user_query="What's in the file?",
        tool_arguments={"path": "/tmp/file.txt"},
        reference_answer="File contents",
        additional_context={"source": "test"},
    )

    assert data_inst["user_query"] == "What's in the file?"
    assert data_inst["tool_arguments"] == {"path": "/tmp/file.txt"}
    assert data_inst["reference_answer"] == "File contents"
    assert data_inst["additional_context"] == {"source": "test"}


def test_mcp_data_inst_required_fields():
    """Test that MCPDataInst has all required fields."""
    data_inst = MCPDataInst(
        user_query="Test query",
        tool_arguments={},
        reference_answer=None,
        additional_context={},
    )

    # All fields should be accessible
    assert "user_query" in data_inst
    assert "tool_arguments" in data_inst
    assert "reference_answer" in data_inst
    assert "additional_context" in data_inst
    assert isinstance(data_inst["tool_arguments"], dict)
    assert isinstance(data_inst["additional_context"], dict)


def test_mcp_data_inst_with_complex_arguments():
    """Test MCPDataInst with complex tool arguments."""
    data_inst = MCPDataInst(
        user_query="Search docs",
        tool_arguments={
            "query": "test query",
            "limit": 10,
            "filters": {"category": "API", "version": "v1"},
        },
        reference_answer="Search results",
        additional_context={},
    )

    assert data_inst["tool_arguments"]["query"] == "test query"
    assert data_inst["tool_arguments"]["limit"] == 10
    assert data_inst["tool_arguments"]["filters"]["category"] == "API"


def test_mcp_output_creation():
    """Test creating an MCPOutput instance."""
    output = MCPOutput(
        final_answer="The answer is 42",
        tool_called=True,
        tool_response="Tool returned 42",
    )

    assert output["final_answer"] == "The answer is 42"
    assert output["tool_called"] is True
    assert output["tool_response"] == "Tool returned 42"


def test_mcp_output_without_tool_call():
    """Test MCPOutput when tool was not called."""
    output = MCPOutput(
        final_answer="Direct answer",
        tool_called=False,
        tool_response=None,
    )

    assert output["final_answer"] == "Direct answer"
    assert output["tool_called"] is False
    assert output["tool_response"] is None


def test_mcp_trajectory_creation():
    """Test creating an MCPTrajectory instance."""
    trajectory = MCPTrajectory(
        user_query="What's the status?",
        tool_name="check_status",
        tool_called=True,
        tool_arguments={"id": "123"},
        tool_response="Status: active",
        tool_description_used="Check system status",
        system_prompt_used="You are a helpful assistant.",
        model_first_pass_output='{"action": "call_tool"}',
        model_final_output="The status is active",
        score=1.0,
    )

    assert trajectory["user_query"] == "What's the status?"
    assert trajectory["tool_name"] == "check_status"
    assert trajectory["tool_called"] is True
    assert trajectory["tool_arguments"] == {"id": "123"}
    assert trajectory["tool_response"] == "Status: active"
    assert trajectory["tool_description_used"] == "Check system status"
    assert trajectory["system_prompt_used"] == "You are a helpful assistant."
    assert trajectory["score"] == 1.0


def test_mcp_trajectory_all_fields_present():
    """Test that MCPTrajectory has all expected fields."""
    trajectory = MCPTrajectory(
        user_query="test",
        tool_name="test_tool",
        tool_called=False,
        tool_arguments=None,
        tool_response=None,
        tool_description_used="desc",
        system_prompt_used="prompt",
        model_first_pass_output="output1",
        model_final_output="output2",
        score=0.5,
    )

    required_fields = [
        "user_query",
        "tool_name",
        "tool_called",
        "tool_arguments",
        "tool_response",
        "tool_description_used",
        "system_prompt_used",
        "model_first_pass_output",
        "model_final_output",
        "score",
    ]

    for field in required_fields:
        assert field in trajectory


def test_mcp_trajectory_with_failed_call():
    """Test MCPTrajectory when tool call failed."""
    trajectory = MCPTrajectory(
        user_query="Get data",
        tool_name="fetch_data",
        tool_called=True,
        tool_arguments={"id": "invalid"},
        tool_response=None,  # Failed to get response
        tool_description_used="Fetch data by ID",
        system_prompt_used="System prompt",
        model_first_pass_output="Attempting to call tool",
        model_final_output="ERROR: Tool call failed",
        score=0.0,
    )

    assert trajectory["tool_called"] is True
    assert trajectory["tool_response"] is None
    assert trajectory["score"] == 0.0
    assert "ERROR" in trajectory["model_final_output"]
