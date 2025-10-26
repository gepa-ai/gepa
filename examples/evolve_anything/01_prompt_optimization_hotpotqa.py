#!/usr/bin/env python3
"""
Example 1: Prompt Optimization for HotpotQA

This example demonstrates using GEPA's evolve() API to optimize a prompt for
question-answering on the HotpotQA dataset. We evolve a simple prompt to improve
accuracy on multi-hop reasoning questions.

Inspired by: openevolve/examples/llm_prompt_optimization/
"""

import os
from typing import Any

# Set up OpenAI API key (or use any other LLM provider)
# For demo purposes, we'll use a mock LLM if no key is provided
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_llm_response(prompt: str, question: str) -> str:
    """Get response from LLM. Uses OpenAI if available, else mocked."""
    if OPENAI_API_KEY:
        try:
            import litellm

            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM call failed: {e}, using mock response")

    # Mock response for testing
    return f"Mock answer based on: {question[:30]}..."


# Sample HotpotQA-style questions (normally you'd load from dataset)
hotpotqa_samples = [
    {
        "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "answer": "Chief of Protocol",
        "context": "Kiss and Tell (1945 film): Shirley Temple plays Corliss Archer. Shirley Temple: Later became Chief of Protocol of the United States.",
    },
    {
        "question": "What is the code name for the German offensive that started on 16 December 1944 across the densely forested Ardennes region?",
        "answer": "Operation Watch on the Rhine",
        "context": "Battle of the Bulge: Started 16 December 1944 in Ardennes. Code name was Operation Watch on the Rhine.",
    },
    {
        "question": "Who is older, Annie Morton or Terry Richardson?",
        "answer": "Terry Richardson",
        "context": "Annie Morton: Born October 8, 1970. Terry Richardson: Born August 14, 1965.",
    },
    {
        "question": "Which author was born earlier, Alistair MacLean or Jack Higgins?",
        "answer": "Alistair MacLean",
        "context": "Alistair MacLean: Born 21 April 1922. Jack Higgins: Born 27 July 1929.",
    },
    {
        "question": "In which country did this Australian who was a member of the fifth season of 'The X Factor Australia' live when she was born?",
        "answer": "Kenya",
        "context": "Fifth season of The X Factor Australia had Samantha Jade. Samantha Jade was born in Perth but lived in Kenya as a child.",
    },
]


def evaluate_prompt(candidate: dict[str, str], batch: list[dict[str, Any]]) -> list[dict]:
    """
    Evaluate the prompt on a batch of HotpotQA questions.

    Returns list of WorkloadResult dicts with score and context_and_feedback.
    """
    results = []
    prompt = candidate["qa_prompt"]

    for item in batch:
        question = item["question"]
        expected_answer = item["answer"]
        context = item["context"]

        # Create the full question with context
        full_question = f"Context: {context}\n\nQuestion: {question}\n\nProvide a brief, direct answer."

        # Get LLM response
        response = get_llm_response(prompt, full_question)

        # Simple scoring: 1.0 if answer substring is in response (case-insensitive)
        score = 1.0 if expected_answer.lower() in response.lower() else 0.0

        # Prepare feedback for reflection
        if score > 0:
            feedback = f"✓ Correct! The response correctly identified '{expected_answer}'."
        else:
            feedback = f"✗ Incorrect. Expected '{expected_answer}' but got '{response[:100]}...'. The context provided was: {context}"

        results.append(
            {
                "score": score,
                "context_and_feedback": {
                    "inputs": full_question,
                    "outputs": response,
                    "feedback": feedback,
                    "expected_answer": expected_answer,
                    "got_correct": score > 0,
                },
            }
        )

    return results


def main():
    """Run prompt optimization using GEPA's evolve() API."""

    # Import GEPA
    from gepa import evolve

    # Initial seed prompt (intentionally simple/suboptimal)
    seed_prompt = "Answer the question based on the given context."

    print("=" * 80)
    print("GEPA Evolve-Anything Example: Prompt Optimization for HotpotQA")
    print("=" * 80)
    print(f"\nInitial prompt: {seed_prompt}")
    print(f"Training on {len(hotpotqa_samples)} HotpotQA examples")
    print("\nStarting evolution...\n")

    # Run GEPA evolution
    result = evolve(
        seed_candidate={"qa_prompt": seed_prompt},
        trainset=hotpotqa_samples,
        evaluate=evaluate_prompt,
        reflection_prompt="""You are optimizing a prompt for question-answering tasks.

The task involves multi-hop reasoning questions where the model must:
1. Understand the provided context
2. Extract relevant information
3. Synthesize a brief, direct answer

Analyze the examples where the current prompt failed. Look for patterns in:
- What information the model missed or misinterpreted
- Whether the prompt encourages thorough context analysis
- Whether the prompt guides toward concise, factual answers

Improve the prompt to:
- Encourage careful reading of context
- Emphasize extracting and combining relevant facts
- Guide toward brief, direct answers (not explanations)
- Handle comparison questions (e.g., "who is older", "which came first")

Provide only the improved prompt text without additional explanation.""",
        num_iterations=20,
        minibatch_size=3,
        teacher_lm="openai/gpt-4o-mini",  # Use a good model for reflection
        random_seed=42,
        output_dir="./hotpotqa_optimization_output",
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"\nBest Score: {result['best_score']:.2%}")
    print("\nBest Prompt:")
    print("-" * 80)
    print(result["best_candidate"]["qa_prompt"])
    print("-" * 80)

    # Test the evolved prompt
    print("\n" + "=" * 80)
    print("Testing Evolved Prompt on All Examples:")
    print("=" * 80)

    eval_results = evaluate_prompt(result["best_candidate"], hotpotqa_samples)
    correct = sum(1 for r in eval_results if r["score"] > 0)

    print(f"\nAccuracy: {correct}/{len(hotpotqa_samples)} = {correct / len(hotpotqa_samples):.2%}")

    for i, (sample, eval_result) in enumerate(zip(hotpotqa_samples, eval_results, strict=False)):
        status = "✓" if eval_result["score"] > 0 else "✗"
        print(f"\n{status} Q{i + 1}: {sample['question'][:60]}...")
        print(f"   Expected: {sample['answer']}")
        print(f"   Got: {eval_result['context_and_feedback']['outputs'][:80]}...")


if __name__ == "__main__":
    main()
