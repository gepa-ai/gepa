"""GEPA optimization of an agent that solves big-number arithmetic with a calculator tool.

Generates expressions like `(384729 * 192847) + (8273 * 99182) - 1029384` with
6-12 digit numbers — well past the size where chat models reliably do mental math.
The agent has access to a single `calculator(expr)` tool; the prompt being
optimized teaches it *when* and *how* to use the tool (decompose, verify, etc).

Run:
    uv run python examples/langchain_adapter/big_number_arithmetic.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Literal

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from gepa import optimize
from gepa.adapters.langchain_adapter import (
    LangChainAdapter,
    last_message_text,
    make_reflection_lm,
)

SEED_SYSTEM_PROMPT = (
    "You are given an arithmetic expression. Use the `calculator` tool to compute it. "
    "Provide your final answer as a single integer on the last line."
)


@tool
def calculator(a: int, b: int, op: Literal["+", "-", "*"]) -> str:
    """Compute a single binary arithmetic operation: returns `a op b` for op in {+, -, *}."""
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    return f"ERROR: unsupported op {op!r}"


def _random_int(rng: random.Random, min_digits: int, max_digits: int) -> int:
    d = rng.randint(min_digits, max_digits)
    return rng.randint(10 ** (d - 1), 10**d - 1)


def generate_problem(
    rng: random.Random,
    num_operands: int,
    min_digits: int,
    max_digits: int,
) -> dict:
    """Build a flat expression like `n0 op n1 op n2 op n3` with random +,-,*
    operators. Python's precedence handles ordering; the agent must use the
    tool to get the right answer either way."""
    ops = ["+", "-", "*"]
    nums = [_random_int(rng, min_digits, max_digits) for _ in range(num_operands)]
    op_seq = [rng.choice(ops) for _ in range(num_operands - 1)]
    parts = [str(nums[0])]
    for i, op in enumerate(op_seq):
        parts.append(op)
        parts.append(str(nums[i + 1]))
    expr = " ".join(parts)
    answer = eval(expr, {"__builtins__": {}}, {})
    return {
        "input": f"Compute: {expr}",
        "answer": str(answer),
        "expression": expr,
        "additional_context": {"expression": expr},
    }


def generate_dataset(
    num_examples: int,
    num_operands: int,
    min_digits: int,
    max_digits: int,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    return [generate_problem(rng, num_operands, min_digits, max_digits) for _ in range(num_examples)]


def extract_final_integer(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def evaluate_response(data: dict, state: dict) -> tuple[float, str]:
    correct = int(data["answer"])
    response = last_message_text(state)
    predicted = extract_final_integer(response)

    tool_calls = _summarize_tool_calls(state)
    trace_note = f"\n\nTool-call trace:\n{tool_calls}" if tool_calls else ""

    if not tool_calls:
        score = 0.0
        feedback = (
            f"You did not call the `calculator` tool. This task requires using the calculator "
            f"for every arithmetic step — mental math is unreliable for these numbers. "
            f"Expression: {data['expression']}. The correct answer is {correct}."
        )
        # print(f"Score: {score}\nFeedback: {feedback}\n")
        return score, feedback

    if predicted is None:
        score = 0.0
        feedback = (
            f"Could not parse an integer from your response. The correct answer is {correct}. "
            f"Ensure your final answer is a single integer on the last line.{trace_note}"
        )
        # print(f"Score: {score}\nFeedback: {feedback}\n")
        return score, feedback

    if predicted == correct:
        return 1.0, ""

    score = 0.0
    feedback = (
        f"Incorrect. You answered {predicted}, but the correct answer is {correct}. "
        f"Expression: {data['expression']}.{trace_note}"
    )
    # print(f"Score: {score}\nFeedback: {feedback}\n")

    return score, feedback


def _summarize_tool_calls(state: dict) -> str:
    """Render each calculator call as `a op b -> result` for the feedback string."""
    messages = state.get("messages") or []
    lines: list[str] = []
    pending: dict[str, str] = {}  # tool_call_id -> "a op b"
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "calculator":
                args = tc.get("args", {})
                pending[tc["id"]] = f"{args.get('a')} {args.get('op')} {args.get('b')}"
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id and tool_call_id in pending:
            lines.append(f"  {pending.pop(tool_call_id)} -> {msg.content}")
    return "\n".join(lines)


def rollout(candidate: dict[str, str], example: dict, llm: BaseChatModel) -> dict:
    agent = create_agent(
        model=llm,
        tools=[calculator],
        system_prompt=candidate["system_prompt"],
    )
    return agent.invoke({"messages": [HumanMessage(content=example["input"])]})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # dataset
    p.add_argument("--train-size", type=int, default=200)
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--test-size", type=int, default=50)
    p.add_argument("--num-operands", type=int, default=6)
    p.add_argument("--min-digits", type=int, default=2)
    p.add_argument("--max-digits", type=int, default=4)
    p.add_argument("--data-seed", type=int, default=42)

    # models — defaults use standard LangChain `init_chat_model` strings.
    p.add_argument("--task-model", default="openai:gpt-41-mini")
    p.add_argument(
        "--task-model-kwargs",
        type=json.loads,
        default={},
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
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-baseline", action="store_true", help="Skip baseline test-set eval")
    p.add_argument("--skip-optimize", action="store_true", help="Run baseline only; skip optimization")

    return p.parse_args()


def main():
    args = parse_args()

    total = args.train_size + args.val_size + args.test_size
    all_data = generate_dataset(
        num_examples=total,
        num_operands=args.num_operands,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        seed=args.data_seed,
    )
    train_set = all_data[: args.train_size]
    val_set = all_data[args.train_size : args.train_size + args.val_size]
    test_set = all_data[args.train_size + args.val_size :]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(f"Sample problem: {all_data[0]['input']}")
    print(f"Sample answer:  {all_data[0]['answer']}")

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
