from datasets import load_dataset
import dspy
import random
from typing import List
import pydantic

MATRIX = List[List[int]]


class TrainingExample(pydantic.BaseModel):
    input: MATRIX
    output: MATRIX


def load_data():
    ds = load_dataset("dataartist/arc-agi")
    trainset = [
        dspy.Example(
            training_examples=[TrainingExample(**train_ex) for train_ex in ex["train"]],
            test_inputs=[x["input"] for x in ex["test"]],
            test_outputs=[x["output"] for x in ex["test"]],
        ).with_inputs("training_examples", "test_inputs")
        for ex in ds["training"]
    ]
    testset = [
        dspy.Example(
            training_examples=[TrainingExample(**train_ex) for train_ex in ex["train"]],
            test_inputs=[x["input"] for x in ex["test"]],
            test_outputs=[x["output"] for x in ex["test"]],
        ).with_inputs("training_examples", "test_inputs")
        for ex in ds["evaluation"]
    ]

    random.Random(0).shuffle(trainset)

    test_set = testset
    val_set = trainset[-200:]
    train_set = [ex for ex in trainset[:-200]]
    print(f"Train set: {len(train_set)}")
    print(f"Val set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")

    return train_set, val_set, test_set
