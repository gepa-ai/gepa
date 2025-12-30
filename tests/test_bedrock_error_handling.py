# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for Bedrock inference profile error handling."""

from unittest.mock import Mock, patch

from gepa import optimize


def test_bedrock_inference_profile_error_handling(capsys):
    """Test that Bedrock inference profile errors are caught and provide helpful guidance."""
    mock_data = [
        {
            "input": "test_input",
            "answer": "test_answer",
            "additional_context": {"context": "test_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    # Mock litellm.completion to raise a Bedrock inference profile error
    with patch("litellm.completion") as mock_completion:
        # Simulate the Bedrock error
        bedrock_error_msg = (
            "BedrockException - {'message':'Invocation of model ID anthropic.claude-sonnet-4-5-20250929-v1:0 "
            "with on-demand throughput isn't supported. Retry your request with the ID or ARN of an inference "
            "profile that contains this model.'}"
        )
        mock_completion.side_effect = Exception(bedrock_error_msg)

        # The optimize function should handle this error gracefully and not crash
        optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm="bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )

        # Check that the error message contains helpful guidance in the logs
        captured = capsys.readouterr()
        log_output = captured.out
        assert "Bedrock Inference Profile Error" in log_output or "Bedrock configuration error" in log_output
        assert "inference profile" in log_output.lower()
        assert "cross-region inference profile" in log_output or "ARN" in log_output
        assert "bedrock/us.anthropic.claude-sonnet" in log_output


def test_bedrock_model_name_without_error():
    """Test that using a Bedrock model name without errors works normally."""
    mock_data = [
        {
            "input": "test_input",
            "answer": "test_answer",
            "additional_context": {"context": "test_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    # Mock litellm.completion to work successfully
    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "improved instructions"
        mock_completion.return_value = mock_response

        result = optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )

        # Should complete successfully without raising an exception
        assert result is not None


def test_non_bedrock_error_passes_through(capsys):
    """Test that non-Bedrock errors are not modified."""
    mock_data = [
        {
            "input": "test_input",
            "answer": "test_answer",
            "additional_context": {"context": "test_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    # Mock litellm.completion to raise a non-Bedrock error
    with patch("litellm.completion") as mock_completion:
        original_error_msg = "Some unrelated API error"
        mock_completion.side_effect = Exception(original_error_msg)

        # The optimize function should handle this error gracefully and not crash
        optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm="openai/gpt-4",
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )

        # Check that the error message is unchanged (no Bedrock guidance added)
        captured = capsys.readouterr()
        log_output = captured.out
        assert "Bedrock Inference Profile Error" not in log_output
        assert original_error_msg in log_output
