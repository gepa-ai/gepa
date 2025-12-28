from datasets import load_dataset
import random
import dspy

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
    
    print(f"Loaded {len(trainset)} training examples")
    print(f"Loaded {len(valset)} validation examples")
    print(f"Loaded {len(testset)} test examples")

    return trainset, valset, testset
