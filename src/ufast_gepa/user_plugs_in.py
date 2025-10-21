"""User-supplied LLM integration hooks with safe defaults.

Projects embedding uFast-GEPA should replace ``task_lm_call`` and
``reflect_lm_call`` with calls into their production LLM stack. The default
implementations below provide deterministic heuristics so local demos and tests
run without external dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def _heuristic_quality(prompt: str, example: Dict[str, Any]) -> float:
    difficulty = float(example.get("difficulty", 0.5))
    coverage = 0.1 if "example" in prompt.lower() else 0.0
    reasoning = 0.15 if "think through" in prompt.lower() else 0.0
    formatting = 0.1 if "format" in prompt.lower() else 0.0
    base = 0.45 + (1.0 - difficulty) * 0.35
    quality = base + coverage + reasoning + formatting
    tokens_penalty = max(len(prompt.split()) - 160, 0) * 0.002
    return max(0.0, min(1.0, quality - tokens_penalty))


async def task_lm_call(prompt: str, example: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the task model against ``example`` and return metrics.

    Replace this function with a call into your LLM provider for production
    deployments. The default heuristic depends on optional ``difficulty``
    metadata in ``example`` to simulate varying hardness.
    """

    tokens = float(len(prompt.split()))
    quality = _heuristic_quality(prompt, example)
    return {
        "quality": quality,
        "neg_cost": -tokens,
        "tokens": tokens,
    }


async def reflect_lm_call(traces: List[Dict[str, Any]], parent_prompt: str, parent_meta: Dict[str, Any] | None = None) -> List[str]:
    """Produce mutated prompt variants based on failure ``traces``.

    The default implementation inspects recent trace messages and adds
    corrective clauses that commonly boost reasoning quality. Swap this out for
    a true reflection prompt when integrating with a live LLM.

    Args:
        traces: List of execution traces from failed examples
        parent_prompt: The current prompt text
        parent_meta: Optional metadata (e.g., temperature) for context
    """

    if not traces:
        return []

    suggestions: Set[str] = set()
    lower_prompt = parent_prompt.lower()
    if "step" not in lower_prompt:
        suggestions.add("Add step-by-step reasoning before final answers.")
    if "format" not in lower_prompt:
        suggestions.add("Clarify the expected output format explicitly.")
    if "avoid" not in lower_prompt:
        suggestions.add("Add an 'Avoid common mistakes' clause.")

    patched_prompt = parent_prompt
    for suggestion in suggestions:
        patched_prompt = f"{patched_prompt}\n\n{suggestion}"
    return [patched_prompt]
