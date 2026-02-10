import re
from unittest.mock import Mock

import pytest

from gepa import optimize


def test_reflection_prompt_template():
    """Test that reflection_prompt_template works with optimize()."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    # Mock the reflection LM to return improved instructions and track calls
    reflection_calls = []

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        reflection_calls.append(prompt)
        return "```\nimproved instructions\n```"

    custom_template = """Current instructions:
<curr_param>
Inputs, outputs, and feedback:
<side_info>
Please improve the instructions."""

    optimize(
        seed_candidate={"instructions": "initial instructions"},
        trainset=mock_data,
        task_lm=task_lm,
        reflection_lm=mock_reflection_lm,
        reflection_prompt_template=custom_template,
        max_metric_calls=2,
        reflection_minibatch_size=1,
    )

    # Check that the reflection_lm was called with our custom template
    assert len(reflection_calls) > 0
    reflection_prompt = reflection_calls[0]
    assert "initial instructions" in reflection_prompt
    assert "my_input" in reflection_prompt
    assert "Please improve the instructions." in reflection_prompt


def test_reflection_prompt_template_missing_placeholders():
    """Test that reflection_prompt_template fails when placeholders are missing."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    # Mock the reflection LM to return improved instructions and track calls
    reflection_calls = []

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        reflection_calls.append(prompt)
        return "```\nimproved instructions\n```"

    custom_template = "Missing both placeholders."

    with pytest.raises(
        ValueError,
        match=re.escape("Missing placeholder(s) in prompt template: <curr_param>, <side_info>"),
    ):
        optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            reflection_prompt_template=custom_template,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )


def test_reflection_prompt_template_dict():
    """Test that reflection_prompt_template works with a dict mapping parameter names to templates."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    # Track which parameter each reflection call was for
    reflection_calls = {}

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        # Store the prompt to check later
        if "Instructions template:" in prompt:
            reflection_calls["instructions"] = prompt
        elif "Context template:" in prompt:
            reflection_calls["context"] = prompt
        return "```\nimproved text\n```"

    # Create parameter-specific templates
    custom_templates = {
        "instructions": """Instructions template:
<curr_param>
Data:
<side_info>
Make it better.""",
        "context": """Context template:
<curr_param>
Feedback:
<side_info>
Improve context.""",
    }

    optimize(
        seed_candidate={"instructions": "initial instructions", "context": "initial context"},
        trainset=mock_data,
        task_lm=task_lm,
        reflection_lm=mock_reflection_lm,
        reflection_prompt_template=custom_templates,
        max_metric_calls=4,
        reflection_minibatch_size=1,
        module_selector="round_robin",  # Round robin to update each component in turn
    )

    # Check that at least one reflection call was made
    assert len(reflection_calls) > 0

    # Verify that custom templates were used correctly for the parameters that were reflected on
    if "instructions" in reflection_calls:
        instructions_call = reflection_calls["instructions"]
        assert "Instructions template:" in instructions_call
        assert "Make it better." in instructions_call

    if "context" in reflection_calls:
        context_call = reflection_calls["context"]
        assert "Context template:" in context_call
        assert "Improve context." in context_call


def test_empty_seed_candidate():
    """Test that optimize() fails gracefully with empty seed_candidate."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        return "```\nimproved instructions\n```"

    # Test with empty dict
    with pytest.raises(ValueError, match=r"seed_candidate must contain at least one component text\."):
        optimize(
            seed_candidate={},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )


def test_none_seed_candidate():
    """Test that optimize() fails gracefully with None seed_candidate."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        return "```\nimproved instructions\n```"

    # Test with None - Note: this will be caught by type checker, but we test runtime behavior
    with pytest.raises(ValueError, match=r"seed_candidate must contain at least one component text\."):
        optimize(
            seed_candidate=None,  # type: ignore
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )
