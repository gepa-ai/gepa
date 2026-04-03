# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.strategies.instruction_proposal import (
    InstructionEditSignature,
    InstructionProposalSignature,
    _fuzzy_find_and_replace,
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


class TestFuzzyFindAndReplace:
    """Test the fuzzy matching cascade used by InstructionEditSignature."""

    def test_exact_match(self):
        text = "Hello world\nFoo bar\nBaz qux"
        result, matched = _fuzzy_find_and_replace(text, "Foo bar", "REPLACED")
        assert matched
        assert result == "Hello world\nREPLACED\nBaz qux"

    def test_whitespace_normalized_match(self):
        text = "Hello world\n  Foo   bar  \nBaz qux"
        result, matched = _fuzzy_find_and_replace(text, "Foo bar", "REPLACED")
        assert matched
        assert "REPLACED" in result
        # Original "  Foo   bar  " line should be gone
        assert "  Foo   bar  " not in result

    def test_indentation_mismatch(self):
        text = "def foo():\n    if True:\n        return 1\n    return 0"
        # Search text has wrong indentation
        search = "if True:\n    return 1"
        replace = "if True:\n    return 42"
        result, matched = _fuzzy_find_and_replace(text, search, replace)
        assert matched
        assert "42" in result

    def test_fuzzy_line_match(self):
        text = "Step 1: Do the thing\nStep 2: Do another thing\nStep 3: Finish up"
        # Slightly different wording
        search = "Step 2: Do another thingg"  # typo
        replace = "Step 2: Do something better"
        result, matched = _fuzzy_find_and_replace(text, search, replace)
        assert matched
        assert "Do something better" in result

    def test_no_match(self):
        text = "Hello world"
        result, matched = _fuzzy_find_and_replace(text, "completely unrelated text that is very different", "X")
        assert not matched
        assert result == "Hello world"

    def test_empty_search(self):
        text = "Hello world"
        result, matched = _fuzzy_find_and_replace(text, "", "X")
        assert not matched
        assert result == "Hello world"

    def test_multiline_exact(self):
        text = "line1\nline2\nline3\nline4"
        result, matched = _fuzzy_find_and_replace(text, "line2\nline3", "replaced2\nreplaced3")
        assert matched
        assert result == "line1\nreplaced2\nreplaced3\nline4"


class TestInstructionEditSignature:
    """Test InstructionEditSignature with search/replace blocks."""

    def test_apply_edits_exact(self):
        original = "Hello world. Foo bar."
        lm_out = (
            "<<<<<<< SEARCH\nFoo bar\n=======\nBaz qux\n>>>>>>> REPLACE"
        )
        edited, failed = InstructionEditSignature._apply_edits(original, lm_out)
        assert edited == "Hello world. Baz qux."
        assert failed == []

    def test_apply_edits_multiple_blocks(self):
        original = "AAA\nBBB\nCCC"
        lm_out = (
            "<<<<<<< SEARCH\nAAA\n=======\nXXX\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nCCC\n=======\nZZZ\n>>>>>>> REPLACE"
        )
        edited, failed = InstructionEditSignature._apply_edits(original, lm_out)
        assert edited == "XXX\nBBB\nZZZ"
        assert failed == []

    def test_apply_edits_no_blocks(self):
        original = "Hello"
        lm_out = "Here is my suggestion: just change it."
        edited, failed = InstructionEditSignature._apply_edits(original, lm_out)
        assert edited is None
        assert failed == []

    def test_apply_edits_fuzzy_fallback(self):
        original = "  Hello   world  \nFoo"
        lm_out = "<<<<<<< SEARCH\nHello world\n=======\nGoodbye world\n>>>>>>> REPLACE"
        edited, failed = InstructionEditSignature._apply_edits(original, lm_out)
        assert edited is not None
        assert "Goodbye world" in edited
        assert failed == []

    def test_apply_edits_partial_failure(self):
        original = "AAA\nBBB"
        lm_out = (
            "<<<<<<< SEARCH\nAAA\n=======\nXXX\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nNOT_IN_TEXT_AT_ALL_VERY_DIFFERENT\n=======\nZZZ\n>>>>>>> REPLACE"
        )
        edited, failed = InstructionEditSignature._apply_edits(original, lm_out)
        # First block should succeed, second should fail
        assert edited is not None
        assert "XXX" in edited
        assert len(failed) == 1
        assert "NOT_IN_TEXT_AT_ALL_VERY_DIFFERENT" in failed[0]

    def test_apply_edits_delete(self):
        original = "Keep this\nDelete this\nKeep this too"
        lm_out = "<<<<<<< SEARCH\nDelete this\n=======\n\n>>>>>>> REPLACE"
        edited, failed = InstructionEditSignature._apply_edits(original, lm_out)
        assert edited is not None
        assert "Delete this" not in edited
        assert failed == []

    def test_run_with_exact_edits(self):
        """Test the full run() flow with a mock LM that returns valid edit blocks."""
        original = "You are a helpful assistant.\nBe concise."
        calls = []

        def mock_lm(prompt):
            calls.append(prompt)
            return (
                "<<<<<<< SEARCH\nBe concise.\n=======\n"
                "Be concise and accurate.\n>>>>>>> REPLACE"
            )

        result = InstructionEditSignature.run(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": original,
                "dataset_with_feedback": [{"input": "test", "output": "test", "feedback": "be more accurate"}],
                "prompt_template": None,
            },
        )
        assert result["new_instruction"] == "You are a helpful assistant.\nBe concise and accurate."
        assert len(calls) == 1  # No retry needed

    def test_run_with_retry(self):
        """Test that failed edits trigger a retry with error feedback."""
        original = "You are a helpful assistant.\nBe concise."
        calls = []

        def mock_lm(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                # First attempt: bad SEARCH text
                return (
                    "<<<<<<< SEARCH\nThis text is not in the original at all and is very different\n=======\n"
                    "Be accurate.\n>>>>>>> REPLACE"
                )
            else:
                # Retry: correct SEARCH text
                return (
                    "<<<<<<< SEARCH\nBe concise.\n=======\n"
                    "Be accurate.\n>>>>>>> REPLACE"
                )

        result = InstructionEditSignature.run(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": original,
                "dataset_with_feedback": [{"input": "test", "output": "test", "feedback": "be accurate"}],
                "prompt_template": None,
            },
            max_retries=1,
        )
        assert result["new_instruction"] == "You are a helpful assistant.\nBe accurate."
        assert len(calls) == 2  # Initial + 1 retry

    def test_run_fallback_to_full_rewrite(self):
        """Test fallback to ``` block extraction when no edit blocks found."""
        original = "Old instruction."

        def mock_lm(prompt):
            # LM returns a full rewrite instead of edit blocks
            return "Here's the new instruction:\n```\nNew instruction.\n```"

        result = InstructionEditSignature.run(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": original,
                "dataset_with_feedback": [{"input": "test", "output": "test", "feedback": "rewrite"}],
                "prompt_template": None,
            },
        )
        assert result["new_instruction"] == "New instruction."

    def test_run_no_retry_when_all_succeed(self):
        """Test that no retry happens when all edits succeed."""
        original = "AAA\nBBB\nCCC"
        calls = []

        def mock_lm(prompt):
            calls.append(prompt)
            return (
                "<<<<<<< SEARCH\nBBB\n=======\nXXX\n>>>>>>> REPLACE"
            )

        result = InstructionEditSignature.run(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": original,
                "dataset_with_feedback": [{"input": "t", "output": "t", "feedback": "f"}],
                "prompt_template": None,
            },
            max_retries=3,
        )
        assert result["new_instruction"] == "AAA\nXXX\nCCC"
        assert len(calls) == 1
