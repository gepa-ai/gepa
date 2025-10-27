"""
Example: Using TurboGEPA DSPy Adapter

This example demonstrates how to use the TurboGEPA DSPy adapter to optimize
DSPy program instructions. The adapter provides async evaluation, caching,
and Pareto-efficient optimization.

Requirements:
    - dspy-ai
    - litellm
    - API key for your chosen LLM (e.g., OpenAI, Anthropic, etc.)

Usage:
    export OPENAI_API_KEY="your-key-here"  # or other provider
    python examples/dspy_adapter_example.py
"""

import asyncio
import os
import dspy
from dspy.primitives import Example
from turbo_gepa.adapters.dspy_adapter import DSpyAdapter


# Step 1: Define your DSPy module
class SimpleQA(dspy.Module):
    """A simple question-answering module."""

    def __init__(self):
        super().__init__()
        # The predictor name here ("qa") will be used as the instruction key
        self.qa = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.qa(question=question)


# Step 2: Configure DSPy with your LLM
def configure_dspy():
    """Configure DSPy with an LLM backend."""
    # Option 1: OpenAI
    if os.getenv("OPENAI_API_KEY"):
        lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    # Option 2: Anthropic
    elif os.getenv("ANTHROPIC_API_KEY"):
        lm = dspy.LM("anthropic/claude-3-haiku-20240307", api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Option 3: OpenRouter (supports many models)
    elif os.getenv("OPENROUTER_API_KEY"):
        lm = dspy.LM(
            "openrouter/google/gemini-flash-1.5",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
        )

    dspy.configure(lm=lm)
    return lm


# Step 3: Create your dataset
def create_dataset():
    """Create a simple training dataset."""
    trainset = [
        Example(
            question="What is 2+2?",
            answer="4"
        ).with_inputs("question"),

        Example(
            question="What is the capital of France?",
            answer="Paris"
        ).with_inputs("question"),

        Example(
            question="What is the largest planet in our solar system?",
            answer="Jupiter"
        ).with_inputs("question"),

        Example(
            question="Who wrote Romeo and Juliet?",
            answer="Shakespeare"
        ).with_inputs("question"),
    ]
    return trainset


# Step 4: Define your metric
def simple_metric(example, prediction, trace=None):
    """
    Evaluation metric: checks if the expected answer appears in the prediction.

    Args:
        example: The input example with expected answer
        prediction: The model's prediction
        trace: Optional trace data (not used in this simple metric)

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    pred_answer = str(prediction.answer) if hasattr(prediction, 'answer') else str(prediction)
    expected = example.answer
    return 1.0 if expected.lower() in pred_answer.lower() else 0.0


# Step 5: Optional - Define feedback function for better reflection
def qa_feedback(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
    """
    Provide feedback for the QA predictor.

    This function analyzes failed predictions and provides actionable feedback
    for improving the instruction.
    """
    from turbo_gepa.adapters.dspy_adapter import ScoreWithFeedback

    # Check if prediction was correct
    expected = module_inputs.answer
    pred_answer = str(predictor_output.get("answer", "")) if isinstance(predictor_output, dict) else str(predictor_output)

    is_correct = expected.lower() in pred_answer.lower()
    score = 1.0 if is_correct else 0.0

    # Provide feedback
    if not is_correct:
        feedback = (
            f"The prediction '{pred_answer}' did not contain the expected answer '{expected}'. "
            f"Consider being more direct and concise in your response."
        )
    else:
        feedback = "Correct answer provided."

    return ScoreWithFeedback(score=score, feedback=feedback)


# Step 6: Optional - Define async reflection LLM function
async def reflection_lm_function(prompt: str) -> str:
    """
    Async LLM function for reflection-based instruction proposal.

    This uses a more powerful model to analyze failures and propose
    improved instructions.
    """
    # Use DSPy's configured LM for reflection
    lm = dspy.settings.lm

    # Call the LM with the reflection prompt
    response = await lm.async_call(prompt)

    return response


# Main optimization function
async def main():
    """Run TurboGEPA optimization on the DSPy program."""

    print("="*60)
    print("TurboGEPA DSPy Adapter Example")
    print("="*60)

    # Configure DSPy
    print("\n1. Configuring DSPy...")
    lm = configure_dspy()
    print(f"   Using model: {lm.model}")

    # Create dataset
    print("\n2. Creating dataset...")
    trainset = create_dataset()
    print(f"   Training examples: {len(trainset)}")

    # Create student module
    print("\n3. Creating DSPy module...")
    student = SimpleQA()
    print(f"   Predictors: {[name for name, _ in student.named_predictors()]}")

    # Create adapter
    print("\n4. Initializing TurboGEPA adapter...")
    adapter = DSpyAdapter(
        student_module=student,
        metric_fn=simple_metric,
        trainset=trainset,
        # Optional: Add feedback map for better reflection
        feedback_map={"qa": qa_feedback},
    )
    print("   ✓ Adapter initialized")

    # Define seed instructions
    seed_instructions = {
        "qa": "Answer the question accurately and concisely."
    }

    # Run optimization
    print("\n5. Starting optimization...")
    print(f"   Seed instruction: {seed_instructions['qa']}")
    print("   This will run for a few rounds (limited by max_rounds)...")

    try:
        result = await adapter.optimize_async(
            seed_instructions=seed_instructions,
            max_rounds=3,  # Run for 3 rounds (quick demo)
            # Optional: provide reflection LLM for better mutations
            # reflection_lm=reflection_lm_function,
        )

        # Display results
        print("\n" + "="*60)
        print("Optimization Results")
        print("="*60)

        print(f"\nBest Quality: {result['best_quality']:.2%}")
        print(f"\nBest Instructions:")
        for pred_name, instruction in result['best_instructions'].items():
            print(f"  {pred_name}: {instruction}")

        print(f"\nPareto Frontier Size: {len(result['pareto_entries'])}")
        print(f"QD Elites: {len(result['qd_elites'])}")

        # Test the best program
        print("\n" + "="*60)
        print("Testing Best Program")
        print("="*60)

        best_program = result['best_program']
        test_question = "What is the speed of light?"

        print(f"\nQuestion: {test_question}")
        prediction = best_program(question=test_question)
        print(f"Answer: {prediction.answer}")

        print("\n✓ Optimization completed successfully!")

    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


# Synchronous wrapper for convenience
def run_optimization():
    """Run the optimization synchronously."""
    asyncio.run(main())


if __name__ == "__main__":
    # Check for API key
    has_key = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("OPENROUTER_API_KEY"),
    ])

    if not has_key:
        print("❌ No API key found!")
        print("\nPlease set one of the following environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENROUTER_API_KEY")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-key-here'")
        exit(1)

    run_optimization()
