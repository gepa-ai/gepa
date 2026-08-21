"""GEPA optimization on AIME math problems.

Train/val: AI-MO/aimo-validation-aime (older AIME problems with worked solutions)
Test:      MathArena/aime_2025 (×N to reduce variance)

Run:
    uv run python examples/langchain_adapter/aime.py
"""

from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from gepa import optimize
from gepa.adapters.langchain_adapter import (
    LangChainAdapter,
    last_message_text,
    make_reflection_lm,
)

SEED_SYSTEM_PROMPT = (
    "Solve the problem and provide the answer in the correct format. "
    "Think step by step, then provide your final answer as a single integer on the last line."
)


def init_dataset(test_repeat: int, data_seed: int):
    """taken from https://dspy.ai/tutorials/gepa_aime/"""
    train_split = load_dataset("AI-MO/aimo-validation-aime")["train"]
    train_split = [
        {
            "input": x["problem"],
            "answer": str(x["answer"]),
            "solution": x["solution"],
            "additional_context": {},
        }
        for x in train_split
    ]
    random.Random(data_seed).shuffle(train_split)
    tot_num = len(train_split)

    test_split = load_dataset("MathArena/aime_2025")["train"]
    test_split = [
        {
            "input": x["problem"],
            "answer": str(x["answer"]),
            "additional_context": {},
        }
        for x in test_split
    ] * test_repeat

    train_set = train_split[: int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num) :]
    return train_set, val_set, test_split


def extract_final_integer(text: str) -> int | None:
    matches = re.findall(r"\b(\d+)\b", text)
    return int(matches[-1]) if matches else None


def evaluate_response(data: dict, state: dict) -> tuple[float, str]:
    correct_answer = int(data["answer"])
    response = last_message_text(state)
    predicted = extract_final_integer(response)

    if predicted is None:
        feedback = (
            f"Could not parse an integer from your response. "
            f"The correct answer is '{correct_answer}'. "
            f"Ensure your final answer is a single integer."
        )
        if data.get("solution"):
            feedback += f"\n\nFull solution:\n{data['solution']}"
        return 0.0, feedback

    if predicted == correct_answer:
        return 1.0, f"Correct. The answer is {correct_answer}."

    feedback = f"Incorrect. You answered {predicted}, but the correct answer is {correct_answer}."
    if data.get("solution"):
        feedback += (
            f"\n\nFull solution:\n{data['solution']}\n\nThink about what takeaways you can learn from this solution."
        )
    return 0.0, feedback


def rollout(candidate: dict[str, str], example: dict, llm: BaseChatModel) -> dict:
    messages = [
        SystemMessage(content=candidate["system_prompt"]),
        HumanMessage(content=example["input"]),
    ]
    result = llm.invoke(messages)
    if not isinstance(result, AIMessage):
        result = AIMessage(content=getattr(result, "content", str(result)))
    return {"messages": messages + [result]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # dataset
    p.add_argument(
        "--test-repeat", type=int, default=5, help="Repeat the AIME 2025 test set N times to reduce variance"
    )
    p.add_argument("--data-seed", type=int, default=0)

    # models — defaults use standard LangChain `init_chat_model` strings.
    p.add_argument("--task-model", default="openai:gpt-5-mini")
    p.add_argument(
        "--task-model-kwargs",
        type=json.loads,
        default={"reasoning_effort": "minimal"},
        help="JSON dict of init_chat_model kwargs (base_url, api_key, default_headers, etc.)",
    )
    p.add_argument("--reflection-model", default="openai:gpt-5")
    p.add_argument(
        "--reflection-model-kwargs",
        type=json.loads,
        default={"reasoning_effort": "medium"},
        help="JSON dict of init_chat_model kwargs for the reflection model",
    )

    # optimizer
    p.add_argument("--max-metric-calls", type=int, default=200)
    p.add_argument("--reflection-minibatch-size", type=int, default=3)
    p.add_argument("--num-threads", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-baseline", action="store_true", help="Skip baseline test-set eval")
    p.add_argument("--skip-optimize", action="store_true", help="Run baseline only; skip optimization")

    return p.parse_args()


def main():
    args = parse_args()

    train_set, val_set, test_set = init_dataset(test_repeat=args.test_repeat, data_seed=args.data_seed)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    task_llm = init_chat_model(args.task_model, **args.task_model_kwargs)
    reflection_llm = init_chat_model(args.reflection_model, **args.reflection_model_kwargs)
    reflection_lm = make_reflection_lm(reflection_llm)

    adapter = LangChainAdapter(
        rollout_fn=lambda candidate, example: rollout(candidate, example, task_llm),
        eval_fn=evaluate_response,
        num_threads=args.num_threads,
    )

    n = len(test_set)
    baseline_correct = 0
    baseline_acc = 0.0
    if not args.skip_baseline:
        print("\nBaseline evaluation on test set...")
        baseline_batch = adapter.evaluate(
            batch=test_set,
            candidate={"system_prompt": SEED_SYSTEM_PROMPT},
            capture_traces=False,
        )
        baseline_correct = sum(1 for s in baseline_batch.scores if s == 1.0)
        baseline_acc = baseline_correct / n * 100
        print(f"\nBaseline:  {baseline_correct}/{n} ({baseline_acc:.1f}%)")

    if args.skip_optimize:
        return

    result = optimize(
        seed_candidate={"system_prompt": SEED_SYSTEM_PROMPT},
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        candidate_selection_strategy="pareto",
        use_merge=True,
        display_progress_bar=True,
        seed=args.seed,
    )

    print(f"\nBest val score: {result.val_aggregate_scores[result.best_idx]}")
    print("\nOptimized system prompt:")
    print("=" * 80)
    print(result.best_candidate["system_prompt"])
    print("=" * 80)

    print("\nOptimized evaluation on test set...")
    optimized_batch = adapter.evaluate(
        batch=test_set,
        candidate=result.best_candidate,
        capture_traces=False,
    )
    optimized_correct = sum(1 for s in optimized_batch.scores if s == 1.0)
    optimized_acc = optimized_correct / n * 100

    print(f"\nBaseline:  {baseline_correct}/{n} ({baseline_acc:.1f}%)")
    print(f"Optimized: {optimized_correct}/{n} ({optimized_acc:.1f}%)")
    print(f"Delta:     {optimized_acc - baseline_acc:+.1f}%")


if __name__ == "__main__":
    main()
