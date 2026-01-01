from datasets import load_dataset
import random
import dspy


def load_math_dataset():
    # Load datasets
    train_data = load_dataset("AI-MO/aimo-validation-aime", split="train")
    test_data = load_dataset("MathArena/aime_2025", split="train")

    # Convert to DSPy Examples using list comprehensions
    train_split = [
        dspy.Example(input=ex["problem"], solution=ex["solution"], answer=ex["answer"]).with_inputs("input")
        for ex in train_data
    ]

    testset = [dspy.Example(input=ex["problem"], answer=ex["answer"]).with_inputs("input") for ex in test_data]

    # Shuffle and Slice
    random.Random(0).shuffle(train_split)
    mid = len(train_split) // 2
    trainset, valset = train_split[:mid], train_split[mid:]

    print(f"Loaded {len(trainset)} training examples")
    print(f"Loaded {len(valset)} validation examples")
    print(f"Loaded {len(testset)} test examples")

    return trainset, valset, testset
