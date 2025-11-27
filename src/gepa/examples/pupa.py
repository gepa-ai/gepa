# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


def init_dataset():
    import random

    from datasets import load_dataset

    raw_ds = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")["train"]

    def _to_inst(item):
        return {
            "input": item["user_query"],
            "additional_context": {
                "predicted_category": str(item.get("predicted_category", "")),
                "pii_units": str(item.get("pii_units", "")),
                "target_response": str(item.get("target_response", "")),
                "redacted_query": str(item.get("redacted_query", "")),
            },
            "answer": str(item["redacted_query"]),
        }

    data = [_to_inst(item) for item in raw_ds]
    rng = random.Random(0)
    rng.shuffle(data)

    mid = len(data) // 2
    trainset = data[:mid]
    valset = data[mid:]
    testset = data[: min(20, len(data))]

    return trainset, valset, testset
