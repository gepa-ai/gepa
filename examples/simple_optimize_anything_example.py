#!/usr/bin/env python3
"""
Simple example: Evolve a prompt using GEPA's optimize_anything API.

This minimal example shows how to use optimize_anything to improve
a system prompt for a math Q&A task.
"""


# Set your API key
# os.environ["OPENAI_API_KEY"] = "your-key-here"


def main():
    """Run prompt evolution example."""
    from gepa import optimize_anything

    # ========================================================================
    # 1. Define the initial prompt to evolve
    # ========================================================================
    seed_candidate = {"system_prompt": "You are a helpful math tutor. Answer concisely."}

    # ========================================================================
    # 2. Define training data (workload instances)
    # ========================================================================
    trainset = [
        {"problem": "What is 2 + 2?", "answer": "4"},
        {"problem": "What is 10 - 3?", "answer": "7"},
        {"problem": "What is 5 × 6?", "answer": "30"},
        {"problem": "What is 20 ÷ 4?", "answer": "5"},
        {"problem": "What is 3²?", "answer": "9"},
        {"problem": "What is √16?", "answer": "4"},
        {"problem": "What is 2 + 3 × 4?", "answer": "14"},
        {"problem": "What is (2 + 3) × 4?", "answer": "20"},
    ]

    # ========================================================================
    # 3. Define evaluation function
    # ========================================================================
    def evaluate(candidate: dict[str, str], batch: list[dict]) -> list[tuple[float, dict]]:
        """
        Evaluate the prompt on a batch of math problems.

        Args:
            candidate: Dict containing "system_prompt" key with prompt text
            batch: List of problem dicts with "problem" and "answer" keys

        Returns:
            List of (score, context) tuples where:
            - score: 1.0 if correct, 0.0 if wrong
            - context: Dict with feedback for reflection
        """
        results = []

        for item in batch:
            # In production, replace with actual LLM call:
            # import openai
            # response = openai.chat.completions.create(
            #     model="gpt-4",
            #     messages=[
            #         {"role": "system", "content": candidate["system_prompt"]},
            #         {"role": "user", "content": item["problem"]}
            #     ]
            # )
            # answer = response.choices[0].message.content.strip()

            # For demo: mock LLM response
            answer = mock_llm(candidate["system_prompt"], item["problem"])

            # Score the response
            correct = answer == item["answer"]
            score = 1.0 if correct else 0.0

            # Build context for reflection - helps GEPA improve the prompt
            context = {
                "problem": item["problem"],
                "expected": item["answer"],
                "got": answer,
                "feedback": (
                    f"✓ Correct answer for '{item['problem']}'"
                    if correct
                    else f"✗ Wrong answer. Problem: '{item['problem']}', Expected: '{item['answer']}', Got: '{answer}'"
                ),
            }

            results.append((score, context))

        return results

    # ========================================================================
    # 4. Run optimization
    # ========================================================================
    print("Starting prompt evolution...")
    print(f"Initial prompt: {seed_candidate['system_prompt']}")
    print()

    result = optimize_anything(
        seed_candidate=seed_candidate,
        trainset=trainset,
        evaluate=evaluate,
        reflection_lm="gpt-4",  # or "gpt-4o-mini", "claude-3-sonnet", etc.
        max_metric_calls=30,  # Stop after 30 evaluations
        reflection_minibatch_size=3,  # Use 3 examples per evolution step
        display_progress_bar=True,
        run_dir="./prompt_evolution_run",  # Save results here
    )

    # ========================================================================
    # 5. Show results
    # ========================================================================
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(f"\nInitial score: {result.history[0]['best_score']:.2%}")
    print(f"Final score:   {result.best_score:.2%}")
    print(f"Improvement:   {(result.best_score - result.history[0]['best_score']):.2%}")

    print(f"\n{'=' * 80}")
    print("BEST PROMPT FOUND:")
    print("=" * 80)
    print(result.best_candidate["system_prompt"])

    print(f"\n{'=' * 80}")
    print("EVOLUTION HISTORY:")
    print("=" * 80)
    for i, h in enumerate(result.history[:10]):  # Show first 10 iterations
        print(f"Iteration {i:2d}: Score = {h['best_score']:.2%}, Metric calls = {h['metric_calls']:3d}")

    if len(result.pareto_frontier) > 1:
        print(f"\n{'=' * 80}")
        print(f"PARETO FRONTIER: {len(result.pareto_frontier)} diverse candidates found")
        print("=" * 80)
        for i, candidate in enumerate(result.pareto_frontier[:3]):
            print(f"\nCandidate {i + 1}:")
            print(f"  Score: {candidate['score']:.2%}")
            print(f"  Prompt: {candidate['candidate']['system_prompt'][:100]}...")


def mock_llm(system_prompt: str, user_message: str) -> str:
    """
    Mock LLM for demonstration.

    In production, replace with actual LLM API call.
    """
    # Simple keyword-based responses for demo
    msg = user_message.lower()

    if "2 + 2" in msg or "2+2" in msg:
        return "4"
    elif "10 - 3" in msg or "10-3" in msg:
        return "7"
    elif "5 × 6" in msg or "5*6" in msg or "5 × 6" in msg:
        return "30"
    elif "20 ÷ 4" in msg or "20/4" in msg or "20 ÷ 4" in msg:
        return "5"
    elif "3²" in msg or "3^2" in msg:
        return "9"
    elif "√16" in msg or "sqrt(16)" in msg:
        return "4"
    elif "2 + 3 × 4" in msg and "(" not in msg:
        return "14"
    elif "(2 + 3) × 4" in msg or "(2 + 3) * 4" in msg:
        return "20"
    else:
        # Sometimes wrong to give GEPA something to optimize
        import random

        return random.choice(["42", "0", "Wrong", "I don't know"])


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GEPA: Evolve Anything - Simple Example                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

This example demonstrates GEPA's optimize_anything API by evolving a system
prompt for a math Q&A task.

NOTE: This demo uses mock LLM responses. To use real LLMs:
  1. Set your API key: export OPENAI_API_KEY="your-key"
  2. Uncomment the openai.chat.completions.create() call in evaluate()
  3. Remove the mock_llm() function

""")

    main()
