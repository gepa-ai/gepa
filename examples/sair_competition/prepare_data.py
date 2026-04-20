#!/usr/bin/env python3
"""Prepare train/val splits for the SAIR Equational Theories competition.

Downloads the public problem subsets from HuggingFace, builds a balanced
valset (~300 problems, 50% TRUE / 50% FALSE) sampled proportionally from
all difficulty levels, then expands both splits by 3x (one copy per
evaluation model).

The split is done BEFORE multiplying by models, so the same equation pair
never appears in both train and val.

Outputs two JSONL files:
  - sair_train.jsonl
  - sair_val.jsonl

Each line is a JSON object with fields:
  - id: original problem ID
  - equation1, equation2: the two equations
  - answer: True/False
  - model: OpenRouter model identifier
  - model_alias: short alias (gpt-oss-120b, llama-3-3-70b-instruct, gemma-4-31b-it)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

MODELS = [
    {"alias": "gpt-oss-120b", "model": "openai/gpt-oss-120b"},
    {"alias": "llama-3-3-70b-instruct", "model": "meta-llama/llama-3.3-70b-instruct"},
    {"alias": "gemma-4-31b-it", "model": "google/gemma-4-31b-it"},
]

# How many val problems to sample from each subset (before model expansion).
# Target: ~300 total, 50/50 balanced, proportional to subset size with
# slight overweight on hard problems (real eval likely includes harder ones).
VAL_BUDGET = {
    "normal": 180,  # 90 TRUE + 90 FALSE
    "hard1": 12,    # 6 TRUE + 6 FALSE (hard1 only has 24 TRUE total)
    "hard2": 36,    # 18 TRUE + 18 FALSE
    "hard3": 72,    # 36 TRUE + 36 FALSE
}

OUTPUT_DIR = Path(__file__).parent
SEED = 42


def _balanced_sample(problems: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Sample n problems (n/2 TRUE + n/2 FALSE). Return (sampled, remaining)."""
    true_problems = [p for p in problems if p["answer"] is True]
    false_problems = [p for p in problems if p["answer"] is False]

    n_true = n // 2
    n_false = n - n_true

    # Clamp to available
    n_true = min(n_true, len(true_problems))
    n_false = min(n_false, len(false_problems))

    rng.shuffle(true_problems)
    rng.shuffle(false_problems)

    sampled = true_problems[:n_true] + false_problems[:n_false]
    remaining_true = true_problems[n_true:]
    remaining_false = false_problems[n_false:]

    return sampled, remaining_true + remaining_false


def _expand_with_models(problems: list[dict]) -> list[dict]:
    """Triple each problem: one copy per evaluation model."""
    expanded = []
    for p in problems:
        for m in MODELS:
            expanded.append({
                "id": p["id"],
                "equation1": p["equation1"],
                "equation2": p["equation2"],
                "answer": p["answer"],
                "model": m["model"],
                "model_alias": m["alias"],
            })
    return expanded


def _assign_one_model_per_problem(problems: list[dict]) -> list[dict]:
    """Assign each problem to one model (round-robin), keeping ~equal model distribution."""
    expanded = []
    for i, p in enumerate(problems):
        m = MODELS[i % len(MODELS)]
        expanded.append({
            "id": p["id"],
            "equation1": p["equation1"],
            "equation2": p["equation2"],
            "answer": p["answer"],
            "model": m["model"],
            "model_alias": m["alias"],
        })
    return expanded


def main():
    rng = random.Random(SEED)

    all_val = []
    all_train = []

    for subset_name, val_count in VAL_BUDGET.items():
        print(f"Loading {subset_name}...")
        ds = load_dataset(
            "SAIRfoundation/equational-theories-selected-problems",
            subset_name,
            split="train",
        )
        problems = [dict(row) for row in ds]
        print(f"  {len(problems)} problems ({sum(1 for p in problems if p['answer'])} TRUE)")

        val_sample, remaining = _balanced_sample(problems, val_count, rng)
        all_val.extend(val_sample)
        all_train.extend(remaining)
        print(f"  Val: {len(val_sample)} ({sum(1 for p in val_sample if p['answer'])} TRUE)")
        print(f"  Train: {len(remaining)}")

    # Shuffle before model expansion
    rng.shuffle(all_val)
    rng.shuffle(all_train)

    # Val: one model per problem (300 examples total, ~100 per model)
    # Train: all three models per problem (full signal for reflection)
    val_expanded = _assign_one_model_per_problem(all_val)
    train_expanded = _expand_with_models(all_train)

    # Write output
    val_path = OUTPUT_DIR / "sair_val.jsonl"
    train_path = OUTPUT_DIR / "sair_train.jsonl"

    for path, data in [(val_path, val_expanded), (train_path, train_expanded)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val_true = sum(1 for p in all_val if p["answer"])
    val_false = len(all_val) - val_true
    print("\n--- Summary ---")
    print(f"Val: {len(val_expanded)} examples ({val_true} TRUE, {val_false} FALSE, 1 model per problem)")
    print(f"Train: {len(train_expanded)} examples ({len(all_train)} problems x 3 models)")
    print("\nWritten to:")
    print(f"  {val_path}")
    print(f"  {train_path}")


if __name__ == "__main__":
    main()
