"""
Seed prompt initialization using PROMPT-MII-style spec induction.

Instead of starting with generic seeds, analyze task examples to generate
structured, task-specific starting prompts.
"""

from __future__ import annotations

import re
from typing import Any

from .interfaces import Candidate


async def initialize_seeds_from_examples(
    examples: list[dict[str, Any]],
    *,
    num_seeds: int = 3,
    reflection_lm: str,
    user_seed: str | None = None,
    reflection_lm_temperature: float | None = None,
) -> list[Candidate]:
    """Generate optimized seed prompts from task examples using PROMPT-MII approach.

    Instead of starting with generic prompts like "You are a helpful assistant",
    this analyzes the task examples and generates structured specifications that:
    1. Define the task clearly
    2. Specify output format
    3. Include policy rules
    4. Provide examples
    5. Define validation checks

    Args:
        examples: List of task input/output examples (3-5 recommended)
        num_seeds: Number of seed prompts to generate (default: 3)
        reflection_lm: Model to use for spec induction
        user_seed: Optional user-provided seed to optimize/expand
        reflection_lm_temperature: Optional temperature for reflection LLM

    Returns:
        List of Candidate objects with generated seed prompts

    Example:
        >>> examples = [
        ...     {"input": "What is 2+2?", "output": "4"},
        ...     {"input": "What is 5*3?", "output": "15"},
        ... ]
        >>> seeds = await initialize_seeds_from_examples(
        ...     examples,
        ...     num_seeds=3,
        ...     reflection_lm="openrouter/google/gemini-2.0-flash-001"
        ... )
        >>> print(seeds[0].text)
        # Structured prompt with TASK, OUTPUT_FORMAT, etc.
    """
    try:
        from litellm import acompletion
    except ImportError:
        raise ImportError("litellm required for seed initialization. Install with: pip install litellm")

    # Build examples text
    examples_text = _format_examples_for_induction(examples[:5])  # Use up to 5 examples

    # Build the spec induction meta-prompt
    if user_seed:
        # User provided a seed - optimize/expand it
        induction_prompt = _build_optimization_prompt(examples_text, user_seed, num_seeds)
    else:
        # No user seed - generate from scratch
        induction_prompt = _build_induction_prompt(examples_text, num_seeds)

    # Call reflection LLM to generate specs
    completion_kwargs = {
        "model": reflection_lm,
        "messages": [{"role": "user", "content": induction_prompt}],
    }
    if reflection_lm_temperature is not None:
        completion_kwargs["temperature"] = reflection_lm_temperature

    response = await acompletion(**completion_kwargs)
    content = response.choices[0].message.content

    # Parse the generated specs
    specs = _parse_generated_specs(content, num_seeds)

    # Convert to Candidates with metadata
    candidates = []
    for idx, spec in enumerate(specs):
        candidates.append(
            Candidate(
                text=spec,
                meta={
                    "generation_method": "prompt_mii_seed_initialization",
                    "seed_index": idx,
                    "from_user_seed": user_seed is not None,
                },
            )
        )

    return candidates


def _format_examples_for_induction(examples: list[dict[str, Any]]) -> str:
    """Format task examples for the meta-prompt."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        input_text = ex.get("input", "")
        output_text = ex.get("output", ex.get("answer", ""))
        formatted.append(f"""Example {i}:
Input: {input_text}
Output: {output_text}
---""")
    return "\n".join(formatted)


def _build_induction_prompt(examples_text: str, num_seeds: int) -> str:
    """Build the PROMPT-MII-style spec induction meta-prompt."""
    return f"""You are an expert at designing prompts for language models. Your task is to analyze the given input/output examples and generate {num_seeds} different high-quality prompt specifications.

Each specification should be a complete, structured prompt that instructs a model to perform the task shown in the examples.

**REQUIRED STRUCTURE** for each specification:

1. TASK: One concise sentence describing what the model must do
2. OUTPUT_FORMAT: Exact structure/schema of the output (format, delimiters, etc.)
3. POLICY_RULES: Behavioral constraints (reasoning style, citations, brevity, etc.)
4. EXAMPLES: 2-3 concise input/output examples inline
5. VALIDATION_CHECKS: 2-3 automatic checks to verify correct output

**Constraints:**
- Each spec should be 200-400 tokens
- Use clear, enforceable wording
- Make it model-agnostic (works with any LLM)
- Each of the {num_seeds} specs should use a DIFFERENT approach or style
- No meta-commentary - just the prompt text

**Task Examples to Analyze:**

{examples_text}

**Generate {num_seeds} different specifications:**

Separate each specification with "---SPEC---" on its own line.
"""


def _build_optimization_prompt(examples_text: str, user_seed: str, num_seeds: int) -> str:
    """Build a prompt to optimize/expand a user-provided seed."""
    return f"""You are an expert at optimizing prompts for language models. The user has provided a basic prompt, and your task is to generate {num_seeds} improved, structured versions based on the task examples.

**User's Original Prompt:**
{user_seed}

**Task Examples (showing what this prompt should accomplish):**

{examples_text}

**Your Task:**
Generate {num_seeds} different optimized versions of the user's prompt. Each should:

1. Preserve the user's intent and core ideas
2. Add structure using these sections:
   - TASK: Clear one-sentence goal
   - OUTPUT_FORMAT: Exact output structure
   - POLICY_RULES: Key behavioral constraints
   - EXAMPLES: 2-3 inline examples
   - VALIDATION_CHECKS: How to verify correctness

3. Each of the {num_seeds} versions should emphasize different aspects:
   - Version 1: Maximize clarity and precision
   - Version 2: Add reasoning/chain-of-thought elements
   - Version 3: Focus on format adherence and validation
   (etc. for additional versions)

**Constraints:**
- 200-400 tokens per spec
- Keep the user's voice/style where appropriate
- Make concrete and actionable
- No meta-commentary

**Generate {num_seeds} optimized specifications:**

Separate each with "---SPEC---" on its own line.
"""


def _parse_generated_specs(content: str, expected_count: int) -> list[str]:
    """Parse the generated specifications from LLM output."""
    # Split by separator
    specs = [s.strip() for s in content.split("---SPEC---") if s.strip()]

    # If LLM didn't use separator, try other common patterns
    if len(specs) < expected_count:
        # Try numbered sections
        numbered = re.split(r"\n(?:Specification|Version|Prompt)\s+\d+:?\s*\n", content)
        if len(numbered) > len(specs):
            specs = [s.strip() for s in numbered if s.strip()]

    # Clean up each spec
    cleaned = []
    for spec in specs:
        # Remove common preambles
        spec = re.sub(
            r"^(?:Here is|This is|Specification|Version|Prompt)\s+(?:\d+|[A-Z]):?\s*", "", spec, flags=re.IGNORECASE
        )
        spec = spec.strip()
        if len(spec) > 50:  # Reasonable minimum length
            cleaned.append(spec)

    # If we got more than expected, take the first N
    if len(cleaned) > expected_count:
        cleaned = cleaned[:expected_count]

    # If we got fewer, warn but return what we have
    if len(cleaned) < expected_count:
        import warnings

        warnings.warn(f"Requested {expected_count} specs but only generated {len(cleaned)}", stacklevel=2)

    return cleaned if cleaned else [_fallback_spec()]


def _fallback_spec() -> str:
    """Fallback specification if generation fails."""
    return """TASK: Analyze the input and provide the correct output following the examples.

OUTPUT_FORMAT: Provide your answer clearly and concisely.

POLICY_RULES:
- Think step-by-step
- Verify your answer
- Use the same format as the examples

EXAMPLES:
[See the input for examples of the task]

VALIDATION_CHECKS:
- Output format matches examples
- Answer is complete and clear
"""


# Convenience function for DefaultAdapter integration
async def maybe_initialize_seeds(
    dataset: list[Any],
    user_seeds: list[str] | None,
    *,
    enable_seed_initialization: bool = False,
    num_generated_seeds: int = 3,
    reflection_lm: str | None = None,
    reflection_lm_temperature: float | None = None,
) -> list[Candidate]:
    """Optionally generate smart seeds from task examples, or use user-provided seeds.

    Args:
        dataset: Task dataset (must have .to_payload() method or be dicts)
        user_seeds: User-provided seed prompts (strings)
        enable_seed_initialization: Whether to use PROMPT-MII-style initialization
        num_generated_seeds: How many seeds to generate if initializing
        reflection_lm: Model for spec induction (required if enable_seed_initialization=True)
        reflection_lm_temperature: Optional temperature for reflection LLM

    Returns:
        List of Candidate objects (either user seeds or generated seeds)

    Example:
        >>> # User provides basic seeds, but wants them optimized
        >>> seeds = await maybe_initialize_seeds(
        ...     dataset=train_data,
        ...     user_seeds=["You are a math tutor"],
        ...     enable_seed_initialization=True,
        ...     reflection_lm="openrouter/google/gemini-2.0-flash-001"
        ... )
        # Returns 3 optimized versions of the user's seed

        >>> # Generate seeds from scratch
        >>> seeds = await maybe_initialize_seeds(
        ...     dataset=train_data,
        ...     user_seeds=None,
        ...     enable_seed_initialization=True,
        ...     reflection_lm="openrouter/google/gemini-2.0-flash-001"
        ... )
        # Returns 3 task-specific generated seeds
    """
    # If initialization disabled, just return user seeds as-is
    if not enable_seed_initialization:
        if user_seeds:
            return [Candidate(text=seed, meta={}) for seed in user_seeds]
        else:
            # Return default generic seed
            return [
                Candidate(
                    text="You are a helpful assistant. Follow the instructions carefully.",
                    meta={"generation_method": "default"},
                )
            ]

    # Seed initialization enabled - need reflection_lm
    if not reflection_lm:
        raise ValueError(
            "enable_seed_initialization=True requires reflection_lm to be set. "
            "Provide a model like 'openrouter/google/gemini-2.0-flash-001'"
        )

    # Sample a few examples from dataset for spec induction
    import random

    examples = []
    for item in random.sample(dataset, min(5, len(dataset))):
        if hasattr(item, "to_payload"):
            payload = item.to_payload()
        elif isinstance(item, dict):
            payload = item
        else:
            continue

        examples.append(
            {
                "input": payload.get("input", ""),
                "output": payload.get("output", payload.get("answer", "")),
            }
        )

    if not examples:
        raise ValueError("Could not extract examples from dataset for seed initialization")

    # If user provided seed(s), optimize the first one and generate variants
    user_seed = user_seeds[0] if user_seeds else None

    # Generate specs
    generated = await initialize_seeds_from_examples(
        examples,
        num_seeds=num_generated_seeds,
        reflection_lm=reflection_lm,
        user_seed=user_seed,
        reflection_lm_temperature=reflection_lm_temperature,
    )

    return generated
