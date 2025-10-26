#!/usr/bin/env python3
"""
Example 4: Evolving Tutorial Content for Maximum Clarity

This example demonstrates using GEPA to evolve educational content. Starting from
a basic tutorial, GEPA discovers improvements that make explanations clearer,
more engaging, and more effective at teaching concepts.

This showcases evolving "soft" text content (documentation, tutorials, explanations)
as opposed to executable code or structured data.
"""

import os
from typing import Any

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_llm_response(system_prompt: str, user_prompt: str) -> str:
    """Get response from LLM."""
    if OPENAI_API_KEY:
        try:
            import litellm

            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM call failed: {e}")

    return "Mock answer for testing"


# Student learning scenarios - different skill levels and learning goals
LEARNING_SCENARIOS = [
    {
        "student_profile": "Beginner programmer, familiar with basic Python",
        "question": "What is recursion and when should I use it?",
        "learning_goal": "Understand recursion concept and recognize use cases",
        "evaluation_criteria": [
            "Explains recursion clearly with simple examples",
            "Contrasts with iterative approaches",
            "Mentions base case and recursive case",
            "Provides practical use cases",
            "Avoids overwhelming technical jargon",
        ],
    },
    {
        "student_profile": "Intermediate programmer learning algorithms",
        "question": "How does recursion help in solving tree traversal problems?",
        "learning_goal": "Connect recursion to tree data structures",
        "evaluation_criteria": [
            "Explains tree structure briefly",
            "Shows clear example of recursive tree traversal",
            "Explains why recursion is natural for trees",
            "Provides working code example",
            "Mentions different traversal orders",
        ],
    },
    {
        "student_profile": "New to programming, understands variables and loops",
        "question": "Can you explain recursion with a very simple example?",
        "learning_goal": "First introduction to recursion concept",
        "evaluation_criteria": [
            "Uses extremely simple, relatable example",
            "Avoids complex terminology",
            "Shows step-by-step execution",
            "Connects to familiar concepts",
            "Encourages experimentation",
        ],
    },
    {
        "student_profile": "Experienced programmer debugging recursive code",
        "question": "My recursive function causes stack overflow. What's wrong?",
        "learning_goal": "Debug and fix recursive code issues",
        "evaluation_criteria": [
            "Identifies common causes of stack overflow in recursion",
            "Explains base case importance",
            "Shows how to add recursion depth limits",
            "Suggests conversion to iteration if appropriate",
            "Provides debugging strategies",
        ],
    },
    {
        "student_profile": "Computer science student studying algorithms",
        "question": "What's the time and space complexity of recursive algorithms?",
        "learning_goal": "Analyze computational complexity of recursion",
        "evaluation_criteria": [
            "Explains call stack space complexity",
            "Shows how to analyze recursive time complexity",
            "Provides examples with complexity analysis",
            "Mentions tail recursion optimization",
            "Compares to iterative complexity",
        ],
    },
]


def evaluate_tutorial(candidate: dict[str, str], batch: list[dict[str, Any]]) -> list[dict]:
    """
    Evaluate tutorial content by simulating student learning.

    We use an LLM to simulate a student reading the tutorial and then
    answering questions to verify understanding.
    """
    results = []
    tutorial_content = candidate["tutorial"]

    for scenario in batch:
        student_profile = scenario["student_profile"]
        question = scenario["question"]
        criteria = scenario["evaluation_criteria"]

        # Simulate student learning: present tutorial, then ask question
        learning_prompt = f"""You are simulating a student with this profile: {student_profile}

You've just read this tutorial on recursion:

---
{tutorial_content}
---

Now answer this question based on what you learned: {question}

Answer naturally as a student would, demonstrating your understanding (or confusion if the tutorial wasn't clear)."""

        student_answer = get_llm_response("You are simulating a student learning from a tutorial.", learning_prompt)

        # Evaluate the student's answer against learning criteria
        eval_prompt = f"""Evaluate if this student answer demonstrates understanding based on the learning criteria.

Student's answer:
{student_answer}

Learning criteria (each worth equal points):
{chr(10).join(f"- {c}" for c in criteria)}

For each criterion:
- Award 1 point if clearly demonstrated
- Award 0.5 points if partially demonstrated
- Award 0 points if not demonstrated or shows confusion

Respond with ONLY a JSON object:
{{"total_points": <number>, "max_points": {len(criteria)}, "analysis": "<brief explanation>"}}"""

        eval_response = get_llm_response("You are an education evaluator assessing student understanding.", eval_prompt)

        # Parse evaluation
        try:
            import json
            import re

            json_match = re.search(r"\{.*\}", eval_response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group(0))
                total = eval_data.get("total_points", 0)
                max_pts = eval_data.get("max_points", len(criteria))
                analysis = eval_data.get("analysis", "")
                score = total / max_pts if max_pts > 0 else 0.0
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            # Fallback: simple keyword matching
            score = sum(
                1
                for criterion in criteria
                if any(word.lower() in student_answer.lower() for word in criterion.split()[:3])
            ) / len(criteria)
            analysis = f"Fallback scoring (parsing failed: {e})"

        # Prepare feedback
        if score >= 0.8:
            feedback = f"✓ Excellent! Student demonstrated strong understanding.\n{analysis}"
        elif score >= 0.5:
            feedback = f"◐ Partial understanding. Some concepts were grasped.\n{analysis}"
        else:
            feedback = f"✗ Poor understanding. Tutorial needs clearer explanation.\n{analysis}"

        feedback += "\n\nMissing coverage: " + ", ".join(
            c for c in criteria if not any(word.lower() in student_answer.lower() for word in c.split()[:2])
        )

        results.append(
            {
                "score": score,
                "context_and_feedback": {
                    "inputs": f"Student: {student_profile}\nQuestion: {question}",
                    "outputs": f"Student answer: {student_answer[:150]}...",
                    "feedback": feedback,
                    "criteria_not_met": [
                        c for c in criteria if not any(word.lower() in student_answer.lower() for word in c.split()[:2])
                    ],
                },
            }
        )

    return results


# Initial basic tutorial (intentionally too technical)
SEED_TUTORIAL = """
# Recursion

Recursion is a programming technique where a function calls itself. 

A recursive function must have:
1. Base case - condition to stop recursion
2. Recursive case - function calls itself with modified input

Example:
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
```

Recursion is useful for tree traversal, divide-and-conquer algorithms, and problems with recursive structure.
"""


def main():
    """Run tutorial evolution using GEPA."""

    from gepa import evolve

    print("=" * 80)
    print("GEPA Evolve-Anything Example: Tutorial Content Evolution")
    print("=" * 80)
    print("\nTopic: Teaching Recursion")
    print(f"Evaluating on {len(LEARNING_SCENARIOS)} diverse student scenarios")
    print("\nInitial Tutorial:")
    print("-" * 80)
    print(SEED_TUTORIAL)
    print("-" * 80)
    print("\nStarting evolution...\n")

    # Run GEPA evolution
    result = evolve(
        seed_candidate={"tutorial": SEED_TUTORIAL},
        trainset=LEARNING_SCENARIOS,
        evaluate=evaluate_tutorial,
        reflection_prompt="""You are evolving a tutorial on recursion to maximize student learning.

A great tutorial should:
1. **Adapt to audience**: Match student's skill level (beginner to advanced)
2. **Clear explanations**: Use simple language, avoid jargon unless necessary
3. **Strong examples**: Provide concrete, relatable examples with step-by-step execution
4. **Build intuition**: Connect to familiar concepts before introducing complexity
5. **Practical guidance**: Answer "when" and "why" questions, not just "what"
6. **Progressive complexity**: Start simple, gradually introduce advanced concepts
7. **Address common issues**: Mention pitfalls and debugging strategies

Based on the feedback, identify gaps:
- Which student profiles struggled to understand?
- Which learning criteria were not met?
- What concepts need clearer explanation?
- Are examples appropriate for the audience?
- Does it address practical debugging concerns?

Improve the tutorial by:
- Adding clearer explanations for struggling students
- Providing better examples for different skill levels
- Including visual/step-by-step breakdowns
- Adding practical tips and common pitfalls
- Structuring content for progressive learning
- Using analogies and relatable examples

Provide only the improved tutorial content in markdown format.""",
        num_iterations=20,
        minibatch_size=3,
        teacher_lm="openai/gpt-4o",
        random_seed=42,
        output_dir="./tutorial_evolution_output",
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"\nBest Learning Score: {result['best_score']:.3f}")
    print("\nEvolved Tutorial:")
    print("-" * 80)
    print(result["best_candidate"]["tutorial"])
    print("-" * 80)

    # Test on all scenarios
    print("\n" + "=" * 80)
    print("Testing Evolved Tutorial on All Student Scenarios:")
    print("=" * 80)

    eval_results = evaluate_tutorial(result["best_candidate"], LEARNING_SCENARIOS)

    avg_score = sum(r["score"] for r in eval_results) / len(eval_results)
    print(f"\nOverall Learning Effectiveness: {avg_score:.2%}")

    for scenario, eval_result in zip(LEARNING_SCENARIOS, eval_results, strict=False):
        status = "✓" if eval_result["score"] > 0.75 else "◐" if eval_result["score"] > 0.4 else "✗"
        print(f"\n{status} {scenario['student_profile']}: {eval_result['score']:.2%} understanding")
        print(f"   Question: {scenario['question']}")
        # print(f"   {eval_result['context_and_feedback']['feedback'][:80]}...")


if __name__ == "__main__":
    main()
