# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa.oa.proposers.claude_code_agent import _extract_fenced


def test_extract_fenced_uses_last_complete_block():
    body = "Explanation:\n```text\nexample\n```\nFinal proposal:\n```markdown\nactual proposal\n```"

    assert _extract_fenced(body) == "actual proposal"


def test_extract_fenced_preserves_unfenced_file_compatibility():
    assert _extract_fenced("  direct proposal  ") == "direct proposal"


def test_extract_fenced_rejects_visibly_truncated_file():
    assert _extract_fenced("<think>unfinished reasoning") == ""
