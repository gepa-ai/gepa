"""User-supplied LLM integration hooks.

Projects embedding TurboGEPA must provide implementations for task_lm_call,
batch_reflect_lm_call, and spec_induction_lm_call that call their LLM stack.

These stub implementations raise NotImplementedError to ensure users provide
real LLM integrations.
"""

from __future__ import annotations

from typing import Any, Dict, List


async def task_lm_call(prompt: str, example: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the task model against ``example`` and return metrics.

    This function MUST be replaced with a real LLM call in production.

    Args:
        prompt: The candidate prompt text
        example: The task example containing "input", "answer", etc.

    Returns:
        Dict with keys:
            - "quality": Float 0-1 indicating correctness
            - "neg_cost": Negative cost (e.g., -tokens/1000)
            - "tokens": Token count
            - "trace": Optional execution trace for reflection

    Raises:
        NotImplementedError: If not overridden with real LLM implementation
    """
    raise NotImplementedError(
        "task_lm_call must be implemented with a real LLM call. "
        "See examples in src/turbo_gepa/adapters/default_adapter.py for reference."
    )


async def batch_reflect_lm_call(
    parent_contexts: List[Dict[str, Any]],
    num_mutations: int,
) -> List[str]:
    """Produce mutated prompt variants from MULTIPLE parent prompts.

    This is more efficient than calling reflect_lm_call separately for each parent,
    and allows the reflection LLM to synthesize ideas across multiple successful prompts.

    Args:
        parent_contexts: List of dicts, each containing:
            - "prompt": The parent prompt text
            - "traces": List of failure traces for this parent
            - "meta": Metadata (quality, cost, temperature, etc.)
        num_mutations: Number of new prompt variants to generate

    Returns:
        List of new prompt variants (length <= num_mutations)

    Raises:
        NotImplementedError: If not overridden with real LLM implementation
    """
    raise NotImplementedError(
        "batch_reflect_lm_call must be implemented with a real LLM call. "
        "See examples in src/turbo_gepa/adapters/default_adapter.py for reference."
    )


async def spec_induction_lm_call(
    task_examples: List[Dict[str, Any]],
    num_specs: int,
) -> List[str]:
    """Generate fresh prompt specifications from task I/O examples.

    This implements the PROMPT-MII concept: instead of editing existing prompts,
    induce a new specification from scratch by looking at what the task requires.

    Args:
        task_examples: List of task examples, each containing:
            - "input": The task input
            - "answer": The expected output
            - (optional) other fields like "difficulty", "additional_context"
        num_specs: Number of fresh specs to generate

    Returns:
        List of new prompt specifications (length <= num_specs)

    Raises:
        NotImplementedError: If not overridden with real LLM implementation
    """
    raise NotImplementedError(
        "spec_induction_lm_call must be implemented with a real LLM call. "
        "See examples in src/turbo_gepa/adapters/default_adapter.py for reference."
    )
