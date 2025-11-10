"""
Strategy definitions for reflection/spec-induction prompts.

Each strategy controls:
    - The system prompt instructions sent to the reflection LLM
    - How we build the user message (using parent contexts/examples)
    - How to parse the raw response into candidate prompt strings
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

StrategyPromptBuilder = Callable[
    [Sequence[dict[str, object]], Sequence[dict[str, object]], Sequence[dict[str, object]], int],
    str,
]
StrategyResponseParser = Callable[[str], list[str]]


@dataclass(slots=True)
class ReflectionStrategy:
    """Configuration for a reflection/spec-induction strategy."""

    name: str
    system_prompt: str
    user_prompt_builder: StrategyPromptBuilder
    response_parser: StrategyResponseParser
    requires_examples: bool = False  # True if builder expects task examples


def _format_parent_summaries(parent_contexts: Sequence[dict[str, object]]) -> str:
    """Render parent prompt summaries similar to the original reflection runner."""

    summaries: list[str] = []
    for i, ctx in enumerate(parent_contexts[:5]):  # Limit for token safety
        prompt_text = ctx.get("prompt", "")
        meta = ctx.get("meta", {}) or {}
        traces = ctx.get("traces", []) or []

        parent_objectives = meta.get("parent_objectives")
        if isinstance(parent_objectives, dict):
            quality = parent_objectives.get("quality", 0.0)
        else:
            quality = meta.get("quality", 0.0)

        shard_fraction = meta.get("quality_shard_fraction", 0.0)
        shard_info = f", shard={shard_fraction * 100:.0f}%" if shard_fraction else ""
        avg_quality = 0.0
        if traces:
            values = [t.get("quality", 0.0) for t in traces[:3]]
            if values:
                avg_quality = sum(values) / len(values)

        perf_summary = f"Recent avg: {avg_quality:.1%}" if traces else f"Quality: {quality:.1%}"
        summaries.append(
            f"""PROMPT {chr(65 + i)} ({perf_summary}{shard_info}):
"{prompt_text.strip()}"
"""
        )

    return "\n".join(summaries)


def _format_reflection_examples(reflection_examples: Sequence[dict[str, object]]) -> str:
    """Summarize previous reflections/examples for few-shot guidance."""

    if not reflection_examples:
        return ""
    summaries: list[str] = []
    for idx, ex in enumerate(reflection_examples[:5]):
        example_input = ex.get("input", "").strip()
        answer = (ex.get("expected_answer") or ex.get("answer") or "").strip()
        assistant_output = ex.get("assistant_output", "").strip()
        feedback = ex.get("feedback", "").strip()
        additional = ex.get("additional_context") or {}
        solution = ""
        if isinstance(additional, dict):
            solution = (additional.get("solution") or additional.get("Solution") or "").strip()

        block = [f"Example {idx + 1} Input: {example_input}"]
        if answer:
            block.append(f"Expected answer: {answer}")
        if assistant_output:
            block.append(f"Assistant output: {assistant_output}")
        if feedback:
            block.append(f"Feedback: {feedback}")
        if solution:
            block.append(f"Solution: {solution}")
        summaries.append("\n".join(block))

    return "\n\n".join(summaries)


def build_incremental_reflection_prompt(
    parent_contexts: Sequence[dict[str, object]],
    reflection_examples: Sequence[dict[str, object]],
    _task_examples: Sequence[dict[str, object]],
    num_mutations: int,
) -> str:
    """Default user message for incremental reflection mutations."""

    parent_section = _format_parent_summaries(parent_contexts)
    examples_section = _format_reflection_examples(reflection_examples)
    instruction = f"""You are TurboGEPA's reflection model. Generate {num_mutations} high-quality prompt mutations.
- Analyze the successful parent prompts and their quality metrics.
- Blend the strongest instructions while keeping structured formatting.
- Each mutation MUST be wrapped inside <PROMPT>...</PROMPT> tags.
- Avoid copying answers (e.g., "### 242") or giving final answers yourself.
- Ensure each prompt ends with guidance to answer using '### <final answer>'.\n"""

    sections = [instruction]
    if parent_section:
        sections.append("=== PARENT PROMPT SUMMARIES ===")
        sections.append(parent_section)
    if examples_section:
        sections.append("=== TASK EXAMPLES & SOLUTIONS ===")
        sections.append(examples_section)

    sections.append("Generate improved prompts now.")
    return "\n".join(sections)


def _format_task_examples(task_examples: Sequence[dict[str, object]]) -> str:
    formatted: list[str] = []
    for i, ex in enumerate(task_examples, 1):
        input_text = ex.get("input", "")
        answer = ex.get("answer", "")
        additional = ex.get("additional_context") or {}
        block = [f"Example {i}:", f"Input: {input_text}", f"Expected Output: {answer}"]
        if isinstance(additional, dict):
            for key, value in additional.items():
                block.append(f"{key.title()}: {value}")
        formatted.append("\n".join(block))
    return "\n\n".join(formatted)


def build_spec_induction_prompt(
    parent_contexts: Sequence[dict[str, object]],
    _reflection_examples: Sequence[dict[str, object]],
    task_examples: Sequence[dict[str, object]],
    num_mutations: int,
) -> str:
    """PROMPT-MII style spec induction prompt."""

    examples_section = _format_task_examples(task_examples[:3])
    parent_section = _format_parent_summaries(parent_contexts)
    instruction = f"""You are designing new instruction prompts for the same task shown below.
Generate {num_mutations} distinct instruction variants that would help an AI assistant solve NEW problems in this domain.
Each instruction MUST be wrapped in <PROMPT>...</PROMPT> tags.
Focus on clarity, domain-specific guidance, and enforcing the '### <final answer>' output format.
"""
    sections = [instruction]
    sections.append("=== TASK EXAMPLES ===")
    sections.append(examples_section)
    if parent_section:
        sections.append("\n=== CURRENT HIGH-PERFORMING PROMPTS ===")
        sections.append(parent_section)
    sections.append("\nWrite the new instructions now.")
    return "\n".join(sections)


def parse_prompts_from_tags(content: str) -> list[str]:
    """Extract prompt blocks from LLM output."""

    matches = re.findall(r"<PROMPT>\s*(.*?)\s*</PROMPT>", content, re.DOTALL | re.IGNORECASE)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    # Fallback: split on line breaks or --- separators
    fallback = [segment.strip() for segment in content.split("---") if segment.strip()]
    if fallback:
        return fallback

    return [content.strip()] if content.strip() else []


BASE_REFLECTION_SYSTEM_PROMPT = (
    "You are a prompt-evolution strategist. Given successful parent prompts and evaluation traces, "
    "generate improved instructions that stay faithful to the task requirements."
)

SPEC_INDUCTION_SYSTEM_PROMPT = (
    "You are a specification engineer. Study the provided task examples and craft new instructions "
    "that would let a model solve similar problems."
)

INTERLEAVED_THINKING_SYSTEM_PROMPT = (
    "You are a cognitive coach specializing in interleaved reasoning. "
    "Rewrite or enhance task instructions so the student alternates between private <think> steps and public "
    "<answer> outputs, each covering one verifiable chunk of reasoning. Preserve the original task intent and "
    "constraints while guiding the student to finish with a single final <answer> that contains only the final solution."
)

def default_reflection_strategies() -> tuple[ReflectionStrategy, ...]:
    """Return the built-in strategy list (incremental reflection + spec induction + interleaved thinking)."""

    return (
        ReflectionStrategy(
            name="incremental_reflection",
            system_prompt=BASE_REFLECTION_SYSTEM_PROMPT,
            user_prompt_builder=build_incremental_reflection_prompt,
            response_parser=parse_prompts_from_tags,
            requires_examples=False,
        ),
        ReflectionStrategy(
            name="spec_induction",
            system_prompt=SPEC_INDUCTION_SYSTEM_PROMPT,
            user_prompt_builder=build_spec_induction_prompt,
            response_parser=parse_prompts_from_tags,
            requires_examples=True,
        ),
        ReflectionStrategy(
            name="interleaved_thinking",
            system_prompt=INTERLEAVED_THINKING_SYSTEM_PROMPT,
            user_prompt_builder=build_interleaved_prompt,
            response_parser=parse_prompts_from_tags,
            requires_examples=False,
        ),
    )


def available_reflection_strategy_names() -> tuple[str, ...]:
    """Return the canonical order of built-in reflection strategy names."""

    return tuple(strategy.name for strategy in default_reflection_strategies())


def resolve_reflection_strategy_names(names: Sequence[str] | None) -> tuple[ReflectionStrategy, ...]:
    """
    Resolve a list of strategy names to ReflectionStrategy objects.

    Args:
        names: Iterable of strategy names (order preserved). If None, returns all defaults.
               Pass an empty tuple/list to disable built-ins (use custom strategies only).
    """

    if names is None:
        return default_reflection_strategies()

    registry = {strategy.name: strategy for strategy in default_reflection_strategies()}
    resolved: list[ReflectionStrategy] = []
    for name in names:
        strategy = registry.get(name)
        if strategy is None:
            available = ", ".join(registry)
            raise ValueError(f"Unknown reflection strategy '{name}'. Available strategies: {available}")
        resolved.append(strategy)
    return tuple(resolved)
def build_interleaved_prompt(
    parent_contexts: Sequence[dict[str, object]],
    reflection_examples: Sequence[dict[str, object]],
    _task_examples: Sequence[dict[str, object]],
    num_mutations: int,
) -> str:
    """Prompt that enforces interleaved <think>/<answer> reasoning."""

    parent_section = _format_parent_summaries(parent_contexts)
    examples_section = _format_reflection_examples(reflection_examples)
    instruction = f"""You are improving prompts to enforce interleaved reasoning.
Generate {num_mutations} new prompt variants that explicitly teach the student to alternate between <think> (private) and <answer> (public) blocks.
Rules:
- Each <think> block must handle one short reasoning step and stay hidden.
- Each <answer> block must summarize only that step’s conclusion for the user.
- The process continues step-by-step until a final <answer> presents ONLY the final solution.
- Keep the original task intent, formatting requirements, and answer style intact.
- Ensure the rewritten prompt clearly explains this alternating pattern and how to end with the final answer.
Wrap every candidate prompt in <PROMPT>...</PROMPT> tags."""

    sections = [instruction]
    if parent_section:
        sections.append("\n=== CURRENT PROMPT SUMMARIES ===")
        sections.append(parent_section)
    if examples_section:
        sections.append("\n=== RECENT TASK TRACES ===")
        sections.append(examples_section)
    sections.append("\nGenerate the improved prompts now.")
    return "\n".join(sections)
