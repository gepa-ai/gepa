# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.strategies.instruction_proposal import InstructionProposalError, InstructionProposalSignature


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

    @pytest.mark.parametrize(
        "lm_output,expected_instruction",
        [
            ("Preamble\n```\nNew instruction\n```\nDone.", "New instruction"),
            ("<think>Reasoning</think>\n```markdown\nNew instruction\n```", "New instruction"),
            ("```\nUse this nested block:\n```python\npass\n```\n```", "Use this nested block:\n```python\npass\n```"),
        ],
    )
    def test_fenced_extractor_accepts_complete_proposals_with_surrounding_text(self, lm_output, expected_instruction):
        result = InstructionProposalSignature.fenced_output_extractor(lm_output)
        assert result["new_instruction"] == expected_instruction

    @pytest.mark.parametrize(
        "lm_output",
        [
            "<think>The generation stopped mid-reasoning",
            "The provider stripped the reasoning tags before truncation",
            "```text\nThe instruction was cut off",
            "The model emitted only a closing fence\n```",
            "```\n```",
        ],
    )
    def test_fenced_extractor_rejects_outputs_without_a_complete_nonempty_proposal(self, lm_output):
        with pytest.raises(InstructionProposalError):
            InstructionProposalSignature.fenced_output_extractor(lm_output)

    def test_legacy_extractor_remains_permissive(self):
        assert InstructionProposalSignature.output_extractor("unfenced instruction") == {
            "new_instruction": "unfenced instruction"
        }
