import random

import dspy
from datasets import load_dataset


def format_dataset(data):
    return [
        dspy.Example(
            training_examples=ex["train"],
            test_inputs=[x["input"] for x in ex["test"]],
            test_outputs=[x["output"] for x in ex["test"]],
        ).with_inputs("training_examples", "test_inputs")
        for ex in data
    ]


def load_arc_agi_dataset():
    ds = load_dataset("dataartist/arc-agi")

    full_train = format_dataset(ds["training"])
    test_set = format_dataset(ds["evaluation"])

    random.Random(0).shuffle(full_train)

    # Using the split logic from the notebook (50/50 split)
    split_idx = len(full_train) // 2
    train_set, val_set = full_train[:split_idx], full_train[split_idx:]

    print(f"Train set: {len(train_set)}")
    print(f"Val set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")

    return train_set, val_set, test_set
