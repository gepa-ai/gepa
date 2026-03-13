import random
import dspy

from datasets import load_dataset


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


predictor = dspy.ChainOfThought(MathSolverSignature)


def run_llm(example, prompt: str):
    """Run the LLM on a single example with the given prompt."""
    predictor.predict.signature.instructions = prompt
    return predictor(input=example.input)


def math_metric(example, prediction):
    """Compute score and detailed feedback for math problems."""
    correct_answer, written_solution = int(example.answer), getattr(example, "solution", "")
    solution_suffix = (
        f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems"
        if written_solution
        else ""
    )

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting. The correct answer is '{correct_answer}'.{solution_suffix}{' and ensure your final answer is a valid integer.' if written_solution else ''}"
        return 0.0, feedback_text

    score = float(correct_answer == llm_answer)
    status = "correct" if score == 1.0 else "incorrect"
    feedback_text = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return score, feedback_text


def load_math_dataset():
    train_split = []
    test_split = []

    train_load_dataset = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    for item in train_load_dataset:
        question = item["problem"]
        solution = item["solution"]
        answer = item["answer"]

        train_split.append(dspy.Example(input=question, solution=solution, answer=answer).with_inputs("input"))

    random.Random(0).shuffle(train_split)

    test_load_dataset = load_dataset("MathArena/aime_2025", "default", split="train")
    for item in test_load_dataset:
        question = item["problem"]
        answer = item["answer"]

        test_split.append(dspy.Example(input=question, answer=answer).with_inputs("input"))

    train_size = len(train_split)
    trainset = train_split[: train_size // 2]
    valset = train_split[train_size // 2 :]
    testset = test_split

    return trainset, valset, testset


def evaluate_on_dataset(prompt, dataset):
    """Evaluate a predictor on a dataset using dspy.Evaluate."""
    predictor.predict.signature.instructions = prompt

    def dspy_metric(example, prediction):
        """Adapter: dspy.Evaluate expects a numeric score, not (score, feedback)."""
        return math_metric(example, prediction)[0]

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=dspy_metric,
        num_threads=16,
        display_progress=True,
    )

    eval_result = evaluator(predictor)
    return eval_result.score / 100.0
