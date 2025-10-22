"""Simple test for seed initialization (no real LLM calls)."""

import asyncio
from turbo_gepa.seed_initializer import _format_examples_for_induction, _build_induction_prompt, _build_optimization_prompt, _parse_generated_specs


def test_format_examples():
    """Test example formatting."""
    examples = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 5*3?", "output": "15"},
    ]

    formatted = _format_examples_for_induction(examples)

    assert "Example 1:" in formatted
    assert "What is 2+2?" in formatted
    assert "4" in formatted
    assert "Example 2:" in formatted
    assert "What is 5*3?" in formatted
    assert "15" in formatted

    print("✓ Example formatting works")


def test_build_prompts():
    """Test prompt building."""
    examples_text = "Example 1: foo\nExample 2: bar"

    # Test induction prompt
    induction_prompt = _build_induction_prompt(examples_text, num_seeds=3)
    assert "3 different" in induction_prompt.lower()
    assert "TASK:" in induction_prompt
    assert "OUTPUT_FORMAT:" in induction_prompt
    assert "POLICY_RULES:" in induction_prompt

    # Test optimization prompt
    opt_prompt = _build_optimization_prompt(examples_text, "You are a tutor", num_seeds=3)
    assert "You are a tutor" in opt_prompt
    assert "3 different" in opt_prompt.lower() or "3" in opt_prompt
    assert "TASK:" in opt_prompt

    print("✓ Prompt building works")


def test_parse_specs():
    """Test spec parsing."""
    # Test with separator - use longer text to pass minimum length filter (50 chars)
    # Avoid starting with "This is" or "Here is" which triggers preamble removal
    content = """
TASK: Solve the math problem step-by-step.
OUTPUT_FORMAT: Provide detailed solution with final answer.
POLICY_RULES: Show all work and verify calculations.

---SPEC---

TASK: Analyze the question carefully before answering.
OUTPUT_FORMAT: Give concise response with reasoning.
POLICY_RULES: State assumptions and check edge cases.

---SPEC---

TASK: Apply systematic approach to problem solving.
OUTPUT_FORMAT: Structure answer with clear sections.
POLICY_RULES: Validate results and explain methodology.
"""

    specs = _parse_generated_specs(content, expected_count=3)
    assert len(specs) == 3
    assert "TASK:" in specs[0]
    assert "TASK:" in specs[1]
    assert "TASK:" in specs[2]

    print(f"✓ Spec parsing works: {len(specs)} specs extracted")


def test_fallback():
    """Test fallback spec generation."""
    from turbo_gepa.seed_initializer import _fallback_spec

    fallback = _fallback_spec()
    assert "TASK:" in fallback
    assert "OUTPUT_FORMAT:" in fallback
    assert "POLICY_RULES:" in fallback

    print("✓ Fallback spec generation works")


if __name__ == "__main__":
    print("Testing seed initialization components...\n")

    test_format_examples()
    test_build_prompts()
    test_parse_specs()
    test_fallback()

    print("\n🎉 All seed initialization component tests passed!")
