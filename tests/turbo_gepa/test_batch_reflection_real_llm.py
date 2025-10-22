"""Test batch reflection with real LLM calls to verify end-to-end integration.

This test requires OPENROUTER_API_KEY to be set.
Run with: OPENROUTER_API_KEY=your_key python tests/turbo_gepa/test_batch_reflection_real_llm.py
"""

import asyncio
import os

from turbo_gepa.interfaces import Candidate
from turbo_gepa.mutator import MutationConfig, Mutator


def create_real_batch_reflection_runner(model: str):
    """Create a real batch reflection runner using litellm."""
    async def batch_reflection_runner(parent_contexts, num_mutations):
        try:
            from litellm import acompletion
        except ImportError:
            raise ImportError("litellm is required for real LLM tests. Install with: pip install litellm")

        # Build reflection prompt
        parent_summaries = []
        for i, ctx in enumerate(parent_contexts[:5]):
            prompt_text = ctx.get("prompt", "")
            meta = ctx.get("meta", {})
            quality = meta.get("quality", 0.0)
            parent_summaries.append(f"""PROMPT {chr(65+i)} (Quality: {quality:.1%}):
"{prompt_text}"
""")

        all_parents_text = "\n".join(parent_summaries)

        reflection_prompt = f"""You are optimizing prompts for a challenging task. Below are {len(parent_contexts)} successful prompts:

{all_parents_text}

Generate {num_mutations} NEW prompt variants that:
1. Synthesize the best ideas from multiple successful prompts above
2. Are substantially different from each other

Output format: Return each new prompt separated by "---" (exactly {num_mutations} prompts)."""

        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": reflection_prompt}],
        )

        content = response.choices[0].message.content
        mutations = [m.strip() for m in content.split("---") if m.strip()]
        return mutations[:num_mutations]

    return batch_reflection_runner


def create_real_spec_induction_runner(model: str):
    """Create a real spec induction runner using litellm."""
    async def spec_induction_runner(task_examples, num_specs):
        try:
            from litellm import acompletion
        except ImportError:
            raise ImportError("litellm is required for real LLM tests. Install with: pip install litellm")

        # Build examples
        examples_text = "\n".join([
            f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}"
            for ex in task_examples[:5]
        ])

        spec_prompt = f"""Based on these input/output examples:

{examples_text}

Generate {num_specs} different system prompts that could solve this task.
Each prompt should use a different approach or strategy.

Output format: Return each prompt separated by "---" (exactly {num_specs} prompts)."""

        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": spec_prompt}],
        )

        content = response.choices[0].message.content
        specs = [s.strip() for s in content.split("---") if s.strip()]
        return specs[:num_specs]

    return spec_induction_runner


def test_real_batch_reflection():
    """Test with real LLM calls to verify the entire pipeline works."""
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not set. Skipping real LLM test.")
        print("   Set it to run this test: export OPENROUTER_API_KEY=your_key")
        return

    print("🔑 API key found, testing with real LLM calls...\n")

    # Use a fast, cheap model
    model = "openrouter/google/gemini-flash-1.5"

    config = MutationConfig(
        reflection_batch_size=3,
        max_mutations=4,
        max_tokens=500,
    )

    mutator = Mutator(
        config,
        batch_reflection_runner=create_real_batch_reflection_runner(model),
        spec_induction_runner=create_real_spec_induction_runner(model),
        temperature_mutations_enabled=False,
    )

    # Create diverse parent contexts
    parent_contexts = [
        {
            "candidate": Candidate(
                text="You are a helpful math tutor. Break down problems step by step.",
                meta={"quality": 0.65}
            ),
            "failures": [
                ("ex1", [{"quality": 0.5, "trace": "Skipped showing work"}]),
            ],
        },
        {
            "candidate": Candidate(
                text="Solve the problem by identifying the key concepts first.",
                meta={"quality": 0.70}
            ),
            "failures": [
                ("ex2", [{"quality": 0.6, "trace": "Didn't explain the reasoning"}]),
            ],
        },
    ]

    task_examples = [
        {"input": "What is 15% of 80?", "output": "12"},
        {"input": "Solve: 2x + 5 = 13", "output": "x = 4"},
    ]

    print(f"Testing with model: {model}")
    print(f"Parent contexts: {len(parent_contexts)}")
    print(f"Task examples: {len(task_examples)}")
    print(f"Requesting: {config.max_mutations} mutations\n")

    # Run the mutation
    proposals = asyncio.run(mutator.propose(parent_contexts, config.max_mutations, task_examples))

    print(f"\n✅ Generated {len(proposals)} mutations\n")

    # Analyze results
    incremental_count = 0
    spec_count = 0

    for i, proposal in enumerate(proposals):
        gen_method = proposal.meta.get("generation_method", "unknown")
        if gen_method == "incremental_reflection":
            incremental_count += 1
        elif gen_method == "spec_induction":
            spec_count += 1

        print(f"Mutation {i+1} ({gen_method}):")
        print(f"  {proposal.text[:100]}...")
        print(f"  Meta: parent={proposal.meta.get('parent', 'N/A')[:16]}, "
              f"num_parents_seen={proposal.meta.get('num_parents_seen', 'N/A')}")
        print()

    print(f"Summary:")
    print(f"  Incremental reflection: {incremental_count}")
    print(f"  Spec induction: {spec_count}")
    print(f"  Total: {len(proposals)}")

    # Verify we got both types
    assert incremental_count > 0, "Should have incremental reflection mutations"
    assert spec_count > 0, "Should have spec induction mutations"

    # Verify metadata
    for proposal in proposals:
        assert "generation_method" in proposal.meta, "Should have generation_method"
        if proposal.meta["generation_method"] == "incremental_reflection":
            assert "num_parents_seen" in proposal.meta, "Incremental should track parents seen"
            assert proposal.meta["num_parents_seen"] == len(parent_contexts), \
                f"Should see all {len(parent_contexts)} parents"

    print("\n🎉 Real LLM batch reflection test passed!")


def test_real_batch_reflection_output_quality():
    """Test that batch reflection actually produces reasonable prompts."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not set. Skipping quality test.")
        return

    print("\n🔍 Testing output quality...\n")

    model = "openrouter/google/gemini-flash-1.5"

    # Create just the batch reflection runner (test it in isolation)
    runner = create_real_batch_reflection_runner(model)

    # Create parent contexts with different approaches
    parent_contexts = [
        {
            "prompt": "Think step by step and show your work.",
            "traces": [{"quality": 0.7, "trace": "Good but verbose"}],
            "meta": {"quality": 0.7},
        },
        {
            "prompt": "Use clear reasoning and verify your answer.",
            "traces": [{"quality": 0.65, "trace": "Sometimes skips verification"}],
            "meta": {"quality": 0.65},
        },
    ]

    # Request 3 mutations
    mutations = asyncio.run(runner(parent_contexts, 3))

    print(f"Requested: 3 mutations")
    print(f"Received: {len(mutations)} mutations\n")

    assert len(mutations) > 0, "Should generate at least one mutation"
    assert len(mutations) <= 3, "Should not exceed requested count"

    for i, mutation in enumerate(mutations):
        print(f"Mutation {i+1}:")
        print(f"  {mutation}")
        print(f"  Length: {len(mutation)} chars, {len(mutation.split())} words")

        # Basic quality checks
        assert len(mutation) > 10, "Mutation should be non-trivial"
        assert len(mutation) < 5000, "Mutation should not be excessively long"

        # Should be different from parents
        assert mutation.strip() != parent_contexts[0]["prompt"].strip(), \
            "Mutation should differ from parent 1"
        assert mutation.strip() != parent_contexts[1]["prompt"].strip(), \
            "Mutation should differ from parent 2"

        print()

    print("✅ Output quality check passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Batch Reflection with Real LLM Calls")
    print("=" * 60)
    print()

    test_real_batch_reflection()
    test_real_batch_reflection_output_quality()

    print("\n" + "=" * 60)
    print("All real LLM tests completed!")
    print("=" * 60)
