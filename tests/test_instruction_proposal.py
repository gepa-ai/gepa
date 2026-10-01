# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.lm import LMOutput
from gepa.strategies.instruction_proposal import (
    InstructionProposalError,
    InstructionProposalSignature,
    parse_proposal,
)


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
            ("```text\nHere are the instructions.", "Here are the instructions."),
            ("Here are the instructions.\n```", "Here are the instructions."),
            (
                "Here are some backticks:\n```\nI hope you didn't get confused.",
                "Here are some backticks:\n```\nI hope you didn't get confused.",
            ),
            ("  Here are the unfenced instructions.  ", "Here are the unfenced instructions."),
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
            ("Do not emit <think> tags in your output.", "Do not emit <think> tags in your output."),
            (LMOutput("```\nComplete proposal\n```\ntrailing", finish_reason="length"), "Complete proposal"),
        ],
    )
    def test_extractor_accepts_complete_proposals_with_surrounding_text(self, lm_output, expected_instruction):
        result = InstructionProposalSignature.output_extractor(lm_output)
        assert result["new_instruction"] == expected_instruction

    @pytest.mark.parametrize(
        "lm_output",
        [
            "<think>The generation stopped mid-reasoning",
            "<think>first</think>\n<think>The second thought was cut off",
            LMOutput("The provider stripped reasoning tags", finish_reason="length"),
            LMOutput("```text\nThe instruction was cut off", finish_reason="max_tokens"),
        ],
    )
    def test_parser_returns_a_typed_miss_for_incomplete_proposals(self, lm_output):
        result = parse_proposal(InstructionProposalSignature, lm_output)

        assert result.text is None
        assert result.error is not None

    def test_legacy_extractor_raises_only_at_its_hard_fail_boundary(self):
        with pytest.raises(InstructionProposalError, match="incomplete"):
            InstructionProposalSignature.output_extractor("<think>unfinished")

    def test_reasoning_envelopes_are_extensible_without_changing_the_adapter(self):
        class ReasoningSignature(InstructionProposalSignature):
            reasoning_tags = ("think", "reasoning")

        rejected = parse_proposal(ReasoningSignature, "<reasoning>unfinished")
        assert rejected.text is None
        assert rejected.error is not None and "unterminated reasoning" in rejected.error

        assert InstructionProposalSignature.output_extractor("<reasoning>valid unfenced instruction") == {
            "new_instruction": "<reasoning>valid unfenced instruction"
        }
