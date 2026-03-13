# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.strategies.instruction_proposal import InstructionCrossoverSignature, InstructionProposalSignature


class TestInstructionProposalSignature:
    """Test InstructionProposalSignature functions."""

    @pytest.mark.parametrize(
        "lm_output,expected_instruction",
        [
            # Test with language specifier
            (
                """Here's the improved instruction:
```markdown
This is the actual instruction content.
It should not include the word 'markdown'.
```
""",
                "This is the actual instruction content.\nIt should not include the word 'markdown'.",
            ),
            # Test without language specifier (original behavior)
            (
                """Here's the instruction:
```
This is the instruction without language specifier.
```
Done.""",
                "This is the instruction without language specifier.",
            ),
            (
                """```markdown
Don't get confused by these backticks: ```
```""",
                "Don't get confused by these backticks: ```",
            ),
            # Test stripping the output string
            (
                """```

Here are the instructions.

```""",
                "Here are the instructions.",
            ),
            # Test multiple sets of backticks (should take the "outermost" block)
            (
                """Begin text
```plaintext
Begin instructions

```
Internal block 1
```

```python
Internal block 2
```

End instructions
```
End text
""",
                "Begin instructions\n\n```\nInternal block 1\n```\n\n```python\nInternal block 2\n```\n\nEnd instructions",
            ),
            # Test when the output starts with ``` but doesn't end with it
            (
                """```text
Here are the instructions.""",
                "Here are the instructions.",
            ),
            # Test when the output ends with ``` but doesn't start with it
            (
                """Here are the instructions.
```""",
                "Here are the instructions.",
            ),
            # Test only backticks in the middle
            (
                """
Here are some backticks:
```
I hope you didn't get confused.
                """,
                "Here are some backticks:\n```\nI hope you didn't get confused.",
            ),
            # Test when there are no backticks at all, also strip whitespace
            (
                """
                Here are the instructions.
                """,
                "Here are the instructions.",
            ),
        ],
    )
    def test_extract_code_blocks(self, lm_output, expected_instruction):
        """Test extraction of instructions from various code block formats."""
        result = InstructionProposalSignature.output_extractor(lm_output)
        assert result["new_instruction"] == expected_instruction


class TestInstructionCrossoverSignature:
    """Test InstructionCrossoverSignature."""

    def test_prompt_includes_both_candidates(self):
        """Test that the prompt renderer includes both primary and donor candidates."""
        input_dict = {
            "current_instruction_doc": "Primary instruction text",
            "donor_instruction_doc": "Donor instruction text",
            "dataset_with_feedback": [{"input": "test", "output": "result", "feedback": "good"}],
            "prompt_template": None,
        }
        prompt = InstructionCrossoverSignature.prompt_renderer(input_dict)
        assert isinstance(prompt, str)
        assert "Primary instruction text" in prompt
        assert "Donor instruction text" in prompt

    def test_run_produces_merged_output(self):
        """Test full run() with a mock LM."""
        calls = []

        def mock_lm(prompt):
            calls.append(prompt)
            return "```\nMerged instruction combining the best of both.\n```"

        result = InstructionCrossoverSignature.run(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": "Candidate A text",
                "donor_instruction_doc": "Candidate B text",
                "dataset_with_feedback": [{"input": "x", "output": "y", "feedback": "combine"}],
                "prompt_template": None,
            },
        )
        assert result["new_instruction"] == "Merged instruction combining the best of both."
        assert len(calls) == 1
        assert "Candidate A" in calls[0]
        assert "Candidate B" in calls[0]

    def test_validate_prompt_template_requires_donor(self):
        """Test that custom templates must include <donor_param>."""
        with pytest.raises(ValueError, match="donor_param"):
            InstructionCrossoverSignature.validate_prompt_template(
                "Only has <curr_param> and <side_info>"
            )

    def test_validate_prompt_template_none_is_ok(self):
        """Test that None template (use default) passes validation."""
        InstructionCrossoverSignature.validate_prompt_template(None)
