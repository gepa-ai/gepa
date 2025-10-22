"""
Demo: PROMPT-MII-style seed initialization.

Shows three usage modes:
1. Generate seeds from scratch (no user input)
2. Optimize a user-provided seed
3. Traditional optimization (no initialization)

Requires:
    pip install datasets
    export OPENROUTER_API_KEY=your_key
"""

import os
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst


def load_sample_dataset():
    """Load a small AIME dataset for demo."""
    from datasets import load_dataset

    print("📥 Loading AIME dataset...")
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")

    dataset = []
    for idx, sample in enumerate(ds.select(range(10))):  # Just 10 examples
        dataset.append(
            DefaultDataInst(
                input=sample["problem"],
                answer="### " + str(sample["answer"]),
                additional_context={"solution": sample.get("solution", "")},
                id=f"train-{idx}",
                difficulty=0.5,
            )
        )

    print(f"✓ Loaded {len(dataset)} AIME problems\n")
    return dataset


def demo_1_generate_from_scratch():
    """Demo 1: Generate seeds from task examples (no user input)."""
    print("=" * 70)
    print("DEMO 1: Generate Seeds from Task Examples")
    print("=" * 70)

    dataset = load_sample_dataset()

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="openrouter/google/gemini-2.0-flash-001",
        reflection_lm="openrouter/google/gemini-2.0-flash-001",
    )

    print("🌱 Generating 3 task-specific seeds using PROMPT-MII approach...")
    print("   (Analyzing task examples to create structured specifications)\n")

    result = adapter.optimize(
        seeds=None,  # No user seeds - generate from scratch
        enable_seed_initialization=True,
        num_generated_seeds=3,
        max_rounds=3,
        max_evaluations=50,
        display_progress=True,
    )

    print("\n📊 Generated Seeds:")
    for i, entry in enumerate(result["pareto_entries"][:3], 1):
        if entry.candidate.meta.get("generation_method") == "prompt_mii_seed_initialization":
            print(f"\n   Seed {i}:")
            print(f"   {entry.candidate.text[:200]}...")
            print(f"   Quality: {entry.result.objectives.get('quality', 0):.2%}")

    return result


def demo_2_optimize_user_seed():
    """Demo 2: User provides a basic seed, PROMPT-MII optimizes it."""
    print("\n" + "=" * 70)
    print("DEMO 2: Optimize User-Provided Seed")
    print("=" * 70)

    dataset = load_sample_dataset()

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="openrouter/google/gemini-2.0-flash-001",
        reflection_lm="openrouter/google/gemini-2.0-flash-001",
    )

    user_seed = "You are a math tutor helping students solve AIME problems."

    print(f"\n🧑 User's original seed:")
    print(f'   "{user_seed}"')
    print(f"\n🌱 Generating 3 optimized variants using PROMPT-MII...\n")

    result = adapter.optimize(
        seeds=[user_seed],
        enable_seed_initialization=True,
        num_generated_seeds=3,
        max_rounds=3,
        max_evaluations=50,
        display_progress=True,
    )

    print("\n📊 Optimized Variants:")
    for i, entry in enumerate(result["pareto_entries"][:3], 1):
        if entry.candidate.meta.get("from_user_seed"):
            print(f"\n   Variant {i}:")
            print(f"   {entry.candidate.text[:200]}...")
            print(f"   Quality: {entry.result.objectives.get('quality', 0):.2%}")

    return result


def demo_3_traditional_no_initialization():
    """Demo 3: Traditional optimization (for comparison)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Traditional Optimization (No Initialization)")
    print("=" * 70)

    dataset = load_sample_dataset()

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="openrouter/google/gemini-2.0-flash-001",
        reflection_lm="openrouter/google/gemini-2.0-flash-001",
    )

    traditional_seed = "You are a helpful assistant. Solve the math problem."

    print(f"\n📝 Using traditional generic seed:")
    print(f'   "{traditional_seed}"')
    print(f"\n🔄 Running traditional optimization...\n")

    result = adapter.optimize(
        seeds=[traditional_seed],
        enable_seed_initialization=False,  # Disabled
        max_rounds=3,
        max_evaluations=50,
        display_progress=True,
    )

    print("\n📊 Results:")
    best = max(result["pareto_entries"], key=lambda e: e.result.objectives.get("quality", 0))
    print(f"\n   Best quality: {best.result.objectives.get('quality', 0):.2%}")
    print(f"   Prompt: {best.candidate.text[:200]}...")

    return result


def compare_results():
    """Compare all three approaches."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n❌ OPENROUTER_API_KEY not set. Set it to run the comparison:")
        print("   export OPENROUTER_API_KEY=your_key")
        return

    print("\nRunning all three demos...\n")

    result1 = demo_1_generate_from_scratch()
    result2 = demo_2_optimize_user_seed()
    result3 = demo_3_traditional_no_initialization()

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    def best_quality(result):
        return max(e.result.objectives.get("quality", 0) for e in result["pareto_entries"])

    print(f"\n1. Generated from scratch:  {best_quality(result1):.2%}")
    print(f"2. Optimized user seed:     {best_quality(result2):.2%}")
    print(f"3. Traditional (no init):   {best_quality(result3):.2%}")

    print("\n💡 Key Insights:")
    print("   • PROMPT-MII initialization creates structured, task-specific prompts")
    print("   • Can optimize user seeds OR generate from scratch")
    print("   • Typically achieves better results faster than generic seeds")
    print("   • Especially valuable for complex tasks (math, code, reasoning)")


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("=" * 70)
        print("❌ ERROR: OPENROUTER_API_KEY not set")
        print("=" * 70)
        print("\nThis demo requires an OpenRouter API key.")
        print("Get one at: https://openrouter.ai/keys")
        print("\nThen run:")
        print("   export OPENROUTER_API_KEY=your_key")
        print("   python examples/demo_seed_initialization.py")
        print("=" * 70)
    else:
        compare_results()
