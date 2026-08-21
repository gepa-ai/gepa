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


class TestTruncatedOutputDetection:
    """A reflection LM cut off mid-thought carries no proposal (#390)."""

    @pytest.mark.parametrize(
        "lm_out",
        [
            # The shape reported in #390: the budget ran out mid-monologue, so
            # neither the reasoning block nor a fence was ever closed.
            "<think>\nOkay, let me try to figure out how to improve the parameter",
            # More opens than closes: one thought was left hanging.
            "<think>done</think>\n<think>and then it stopped",
        ],
    )
    def test_unterminated_reasoning_block_is_truncated(self, lm_out):
        assert InstructionProposalSignature.is_truncated_output(lm_out)

    @pytest.mark.parametrize(
        "lm_out",
        [
            # A complete reasoning block followed by a fenced instruction.
            "<think>\nI reasoned.\n</think>\n```\nThe new instruction\n```",
            # No reasoning block at all, fenced.
            "Here it is:\n```\nThe new instruction\n```",
            # No fence either. The extractor salvages this deliberately, so it
            # must not be reclassified as truncated by this predicate.
            "Here are the instructions.",
            # An unterminated fence is likewise left to the extractor.
            "```text\nHere are the instructions.",
            # The instruction may legitimately talk about the tag, in either
            # direction: a fenced proposal is extractable whatever its text says.
            "```\nNever emit a </think> tag.\n```",
            "```\nAlways open a <think> block before answering.\n```",
            # An unclosed reasoning block that still reached a fenced
            # instruction: the extractor takes the fence, so nothing is lost.
            "<think>\nStill thinking\n```\nAn instruction\n```",
        ],
    )
    def test_well_formed_output_is_not_truncated(self, lm_out):
        assert not InstructionProposalSignature.is_truncated_output(lm_out)

    def test_reasoning_tags_are_extensible_by_subclass(self):
        class QwenLikeSignature(InstructionProposalSignature):
            reasoning_tags = ("think", "reasoning")

        cut_off = "<reasoning>\nHalf a thought"
        assert QwenLikeSignature.is_truncated_output(cut_off)
        assert not InstructionProposalSignature.is_truncated_output(cut_off)
