# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Tests for MCP adapter.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Try to import MCP adapter, skip tests if dependencies not installed
pytest.importorskip("mcp", reason="MCP SDK not installed")

from gepa.adapters.mcp_adapter import MCPAdapter, MCPDataInst
from gepa.core.adapter import EvaluationBatch


class TestMCPAdapterInitialization:
    """Tests for MCPAdapter initialization."""

    def test_adapter_creation_with_string_model(self, server_params, simple_metric):
        """Test creating adapter with model string."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        assert adapter.tool_name == "read_file"
        assert adapter.task_model == "openai/gpt-4o-mini"
        assert adapter.enable_two_pass is True
        assert adapter.failure_score == 0.0

    def test_adapter_creation_with_callable_model(self, server_params, simple_metric, mock_model_callable):
        """Test creating adapter with callable model."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model=mock_model_callable,
            metric_fn=simple_metric,
        )

        assert adapter.task_model is mock_model_callable
        assert callable(adapter.task_model)

    def test_adapter_with_custom_parameters(self, server_params, simple_metric):
        """Test creating adapter with custom parameters."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
            base_system_prompt="Custom prompt",
            enable_two_pass=False,
            failure_score=0.5,
        )

        assert adapter.base_system_prompt == "Custom prompt"
        assert adapter.enable_two_pass is False
        assert adapter.failure_score == 0.5


class TestMCPAdapterHelperMethods:
    """Tests for MCPAdapter helper methods."""

    def test_build_system_prompt(self, server_params, simple_metric, mock_mcp_tool):
        """Test system prompt building."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        candidate = {"tool_description": "Custom tool description"}
        prompt = adapter._build_system_prompt(candidate, mock_mcp_tool)

        assert "Custom tool description" in prompt
        assert "read_file" in prompt
        assert "call_tool" in prompt

    def test_find_tool_success(self, server_params, simple_metric, mock_mcp_tool):
        """Test finding a tool by name."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        tools = [mock_mcp_tool]
        found_tool = adapter._find_tool(tools, "read_file")

        assert found_tool.name == "read_file"

    def test_find_tool_not_found(self, server_params, simple_metric, mock_mcp_tool):
        """Test error when tool not found."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="nonexistent_tool",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        tools = [mock_mcp_tool]
        with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not found"):
            adapter._find_tool(tools, "nonexistent_tool")

    def test_extract_tool_response(self, server_params, simple_metric):
        """Test extracting text from tool response."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        # Mock result with text content
        result = MagicMock()
        text_content = MagicMock()
        text_content.text = "File content here"
        result.content = [text_content]

        extracted = adapter._extract_tool_response(result)
        assert extracted == "File content here"

    def test_generate_tool_feedback_success(self, server_params, simple_metric):
        """Test generating feedback for successful tool usage."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        trajectory = {
            "user_query": "test",
            "tool_name": "read_file",
            "tool_called": True,
            "tool_arguments": {"path": "/tmp/test.txt"},
            "tool_response": "content",
            "tool_description_used": "desc",
            "system_prompt_used": "prompt",
            "model_first_pass_output": "output",
            "model_final_output": "final",
            "score": 0.8,
        }

        feedback = adapter._generate_tool_feedback(trajectory, 0.8)
        assert "Good" in feedback or "correct" in feedback.lower()

    def test_generate_tool_feedback_failure(self, server_params, simple_metric):
        """Test generating feedback for failed tool usage."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        trajectory = {
            "user_query": "test",
            "tool_name": "read_file",
            "tool_called": False,
            "tool_arguments": None,
            "tool_response": None,
            "tool_description_used": "desc",
            "system_prompt_used": "prompt",
            "model_first_pass_output": "output",
            "model_final_output": "wrong answer",
            "score": 0.0,
        }

        feedback = adapter._generate_tool_feedback(trajectory, 0.0)
        assert "incorrect" in feedback.lower() or "not called" in feedback.lower()


class TestMCPAdapterEvaluation:
    """Tests for MCPAdapter evaluation."""

    @patch("gepa.adapters.mcp_adapter.mcp_adapter.stdio_client")
    @patch("gepa.adapters.mcp_adapter.mcp_adapter.ClientSession")
    def test_evaluate_basic(
        self,
        mock_client_session_class,
        mock_stdio_client,
        server_params,
        simple_metric,
        sample_dataset,
        seed_candidate,
        mock_mcp_session,
    ):
        """Test basic evaluation flow."""
        # Setup mocks
        mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        mock_client_session_class.return_value = mock_mcp_session

        # Create adapter with callable model to avoid litellm
        def mock_model(messages):
            return '{"action": "call_tool", "arguments": {"path": "/tmp/test.txt"}}'

        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model=mock_model,
            metric_fn=simple_metric,
        )

        # Run evaluation
        result = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=False,
        )

        # Verify result structure
        assert isinstance(result, EvaluationBatch)
        assert len(result.outputs) == len(sample_dataset)
        assert len(result.scores) == len(sample_dataset)
        assert result.trajectories is None

    @patch("gepa.adapters.mcp_adapter.mcp_adapter.stdio_client")
    @patch("gepa.adapters.mcp_adapter.mcp_adapter.ClientSession")
    def test_evaluate_with_traces(
        self,
        mock_client_session_class,
        mock_stdio_client,
        server_params,
        simple_metric,
        sample_dataset,
        seed_candidate,
        mock_mcp_session,
    ):
        """Test evaluation with trajectory capture."""
        # Setup mocks
        mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        mock_client_session_class.return_value = mock_mcp_session

        def mock_model(messages):
            return '{"action": "call_tool", "arguments": {"path": "/tmp/test.txt"}}'

        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model=mock_model,
            metric_fn=simple_metric,
        )

        # Run evaluation with traces
        result = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=True,
        )

        # Verify trajectories captured
        assert result.trajectories is not None
        assert len(result.trajectories) == len(sample_dataset)

        # Check trajectory structure
        traj = result.trajectories[0]
        assert "user_query" in traj
        assert "tool_name" in traj
        assert "tool_called" in traj

    @patch("gepa.adapters.mcp_adapter.mcp_adapter.stdio_client")
    @patch("gepa.adapters.mcp_adapter.mcp_adapter.ClientSession")
    def test_evaluate_handles_errors_gracefully(
        self,
        mock_client_session_class,
        mock_stdio_client,
        server_params,
        simple_metric,
        sample_dataset,
        seed_candidate,
    ):
        """Test that evaluation handles errors gracefully."""
        # Setup mocks to raise error
        mock_stdio_client.return_value.__aenter__.side_effect = Exception("Connection failed")

        def mock_model(messages):
            return '{"action": "answer", "text": "test"}'

        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model=mock_model,
            metric_fn=simple_metric,
            failure_score=0.0,
        )

        # Run evaluation
        result = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=False,
        )

        # Should return failure scores, not raise
        assert len(result.outputs) == len(sample_dataset)
        assert all(score == 0.0 for score in result.scores)


class TestMCPAdapterReflectiveDataset:
    """Tests for reflective dataset generation."""

    def test_make_reflective_dataset_tool_description(self, server_params, simple_metric, seed_candidate):
        """Test reflective dataset generation for tool_description component."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        # Create mock evaluation batch
        trajectories = [
            {
                "user_query": "What's in the file?",
                "tool_name": "read_file",
                "tool_called": True,
                "tool_arguments": {"path": "/tmp/test.txt"},
                "tool_response": "File content",
                "tool_description_used": "Read file contents",
                "system_prompt_used": "System prompt",
                "model_first_pass_output": "Calling tool",
                "model_final_output": "The file contains...",
                "score": 1.0,
            }
        ]

        eval_batch = EvaluationBatch(
            outputs=[{"final_answer": "answer", "tool_called": True, "tool_response": "resp"}],
            scores=[1.0],
            trajectories=trajectories,
        )

        # Generate reflective dataset
        reflective_data = adapter.make_reflective_dataset(
            candidate=seed_candidate,
            eval_batch=eval_batch,
            components_to_update=["tool_description"],
        )

        # Verify structure
        assert "tool_description" in reflective_data
        assert len(reflective_data["tool_description"]) == 1

        example = reflective_data["tool_description"][0]
        assert "Inputs" in example
        assert "Generated Outputs" in example
        assert "Feedback" in example

    def test_make_reflective_dataset_system_prompt(self, server_params, simple_metric, seed_candidate):
        """Test reflective dataset generation for system_prompt component."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        trajectories = [
            {
                "user_query": "What's in the file?",
                "tool_name": "read_file",
                "tool_called": False,
                "tool_arguments": None,
                "tool_response": None,
                "tool_description_used": "Read file contents",
                "system_prompt_used": "System prompt",
                "model_first_pass_output": "Direct answer",
                "model_final_output": "Wrong answer",
                "score": 0.0,
            }
        ]

        eval_batch = EvaluationBatch(
            outputs=[{"final_answer": "wrong", "tool_called": False, "tool_response": None}],
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

    def test_make_reflective_dataset_multiple_components(self, server_params, simple_metric, seed_candidate):
        """Test reflective dataset generation for multiple components."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        trajectories = [
            {
                "user_query": "Test",
                "tool_name": "read_file",
                "tool_called": True,
                "tool_arguments": {},
                "tool_response": "resp",
                "tool_description_used": "desc",
                "system_prompt_used": "prompt",
                "model_first_pass_output": "out1",
                "model_final_output": "out2",
                "score": 0.5,
            }
        ]

        eval_batch = EvaluationBatch(
            outputs=[{"final_answer": "ans", "tool_called": True, "tool_response": "r"}],
            scores=[0.5],
            trajectories=trajectories,
        )

        reflective_data = adapter.make_reflective_dataset(
            candidate=seed_candidate,
            eval_batch=eval_batch,
            components_to_update=["tool_description", "system_prompt"],
        )

        assert "tool_description" in reflective_data
        assert "system_prompt" in reflective_data


class TestMCPAdapterTwoPassWorkflow:
    """Tests for two-pass workflow."""

    def test_two_pass_disabled(self, server_params, simple_metric):
        """Test that two-pass can be disabled."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
            enable_two_pass=False,
        )

        assert adapter.enable_two_pass is False

    def test_two_pass_enabled_by_default(self, server_params, simple_metric):
        """Test that two-pass is enabled by default."""
        adapter = MCPAdapter(
            server_params=server_params,
            tool_name="read_file",
            task_model="openai/gpt-4o-mini",
            metric_fn=simple_metric,
        )

        assert adapter.enable_two_pass is True


def test_mcp_adapter_import():
    """Test that MCPAdapter can be imported."""
    from gepa.adapters.mcp_adapter import MCPAdapter

    assert MCPAdapter is not None


def test_mcp_types_import():
    """Test that MCP types can be imported."""
    from gepa.adapters.mcp_adapter import MCPOutput, MCPTrajectory

    assert MCPOutput is not None
    assert MCPTrajectory is not None
