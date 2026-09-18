# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.strategies.instruction_proposal import InstructionProposalSignature


class TestInstructionProposalSignature:
    """Test InstructionProposalSignature functions."""

    @pytest.mark.parametrize(
        "lm_output,expected_instruction",
        [
            # Test with language specifier
            (
                """```markdown
This is the actual instruction content.
It should not include the word 'markdown'.
```
""",
                "This is the actual instruction content.\nIt should not include the word 'markdown'.",
            ),
            # Test without language specifier (original behavior)
            (
                """```
This is the instruction without language specifier.
```""",
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
                """```plaintext
Begin instructions

```
Internal block 1
```

```python
Internal block 2
```

End instructions
```""",
                "Begin instructions\n\n```\nInternal block 1\n```\n\n```python\nInternal block 2\n```\n\nEnd instructions",
            ),
        ],
    )
    def test_extract_code_blocks(self, lm_output, expected_instruction):
        """Test extraction of instructions from various code block formats."""
        result = InstructionProposalSignature.output_extractor(lm_output)
        assert result["new_instruction"] == expected_instruction

    @pytest.mark.parametrize(
        "lm_output",
        [
            "```text\nHere are the instructions.",
            "Here are the instructions.\n```",
            "Here are some backticks:\n```\nBut no outer fence.",
            "Here are the instructions.",
            "Analysis before the fence.\n```\nHere are the instructions.\n```",
            "```\nHere are the instructions.\n```\nCommentary after the fence.",
            "```\n```",
        ],
    )
    def test_rejects_incomplete_or_nonconforming_fences(self, lm_output):
        with pytest.raises(ValueError, match="complete outer code fence"):
            InstructionProposalSignature.output_extractor(lm_output)
