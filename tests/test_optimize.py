from unittest.mock import Mock
import re
from gepa.core.adapter import EvaluationBatch
from gepa import optimize
import pytest


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
<curr_instructions>
Inputs, outputs, and feedback:
<inputs_outputs_feedback>
Please improve the instructions."""

    result = optimize(
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

    with pytest.raises(ValueError, match=re.escape("Missing placeholder(s) in prompt template: <curr_instructions>, <inputs_outputs_feedback>")):
        result = optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            reflection_prompt_template=custom_template,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )
