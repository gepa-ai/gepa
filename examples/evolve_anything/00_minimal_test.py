#!/usr/bin/env python3
"""
Minimal test to verify the evolve() API works correctly.
This is a toy example that evolves a greeting message.
"""

from typing import Any


def evaluate_greeting(candidate: dict[str, str], batch: list[dict[str, Any]]) -> list[dict]:
    """Evaluate a greeting on different scenarios."""
    results = []
    greeting = candidate["greeting"]

    for scenario in batch:
        name = scenario["name"]
        context = scenario["context"]
        expected_words = scenario.get("expected_words", [])

        # Simple scoring: check if greeting includes expected elements
        greeting_lower = greeting.lower()
        score = sum(1 for word in expected_words if word.lower() in greeting_lower) / max(len(expected_words), 1)

        # Feedback
        missing = [w for w in expected_words if w.lower() not in greeting_lower]
        if missing:
            feedback = f"Missing important elements: {', '.join(missing)}"
        else:
            feedback = "Good! Includes all expected elements."

        results.append(
            {
                "score": score,
                "context_and_feedback": {
                    "inputs": f"Name: {name}, Context: {context}",
                    "outputs": greeting,
                    "feedback": feedback,
                },
            }
        )

    return results


def main():
    """Test the evolve API with a simple greeting evolution."""
    try:
        from gepa import evolve
    except ImportError:
        print("Error: Could not import gepa.evolve()")
        print("Make sure GEPA is installed: pip install -e .")
        return

    # Training scenarios
    scenarios = [
        {"name": "Alice", "context": "professional email", "expected_words": ["professional", "regards"]},
        {"name": "Bob", "context": "casual chat", "expected_words": ["friendly", "casual"]},
        {"name": "Dr. Smith", "context": "academic", "expected_words": ["respectful", "formal"]},
    ]

    # Initial greeting (intentionally generic)
    seed_greeting = "Hello!"

    print("=" * 60)
    print("GEPA Evolve API - Minimal Test")
    print("=" * 60)
    print(f"\nInitial greeting: {seed_greeting}")
    print("Evolving greeting for different contexts...")
    print()

    try:
        # Run evolution (very few iterations for quick test)
        result = evolve(
            seed_candidate={"greeting": seed_greeting},
            trainset=scenarios,
            evaluate=evaluate_greeting,
            num_iterations=5,  # Very short for testing
            minibatch_size=2,
            teacher_lm="openai/gpt-4o-mini",
            random_seed=42,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)
        print(f"\nBest Score: {result['best_score']:.2f}")
        print(f"Best Greeting: {result['best_candidate']['greeting']}")
        print("\n✓ API test passed!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
