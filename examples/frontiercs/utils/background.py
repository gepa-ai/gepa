"""Per-problem background builder for Frontier-CS optimization."""

from __future__ import annotations

from typing import Any

BASE_BACKGROUND = """Domain: Competitive programming (Frontier-CS benchmark).
The candidate is a system prompt for an LLM to generate C++ solutions.
Solutions are compiled (C++17) and evaluated against hidden test cases.
Statuses: AC (accepted), WA (wrong answer), TLE (time limit exceeded), RE (runtime error).
Goal: maximize accepted test cases."""


def build_background(problem: dict[str, Any]) -> str:
    """Build a per-problem background string with enriched metadata."""
    parts = [BASE_BACKGROUND]
    parts.append(f"Problem category: {problem.get('tag', 'unknown')}")
    parts.append(f"Time limit: {problem.get('time_limit', 'N/A')} ms")
    parts.append(f"Memory limit: {problem.get('memory_limit', 'N/A')} KB")
    if problem.get("sample_input"):
        parts.append(f"Sample input:\n{problem['sample_input']}")
        parts.append(f"Expected output:\n{problem['sample_output']}")
    return "\n".join(parts)
