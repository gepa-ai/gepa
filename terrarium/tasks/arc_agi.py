"""ARC-AGI task: optimize agent code for abstract reasoning puzzles.

This is a dataset task (generalization mode). The candidate is Python code
that implements an ARC-AGI solving agent.
"""

from __future__ import annotations

from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task

DESCRIPTION = """\
Optimize Python agent code for solving ARC-AGI abstract reasoning puzzles.
The candidate is Python code defining `solve(train_inputs, train_outputs, test_inputs, llm)`
that infers transformation patterns from training examples and predicts test outputs.
Score = test accuracy (fraction of correctly predicted test grids), with a cost penalty.
"""

INITIAL_CANDIDATE = '''\
import json, re

def solve(train_inputs, train_outputs, test_inputs, llm):
    training_examples = "\\n".join(f"Input: {i}\\nOutput: {o}" for i, o in zip(train_inputs, train_outputs))
    problem_inputs = "\\n".join(f"Input {i}: {x}" for i, x in enumerate(train_inputs + test_inputs))

    prompt = f"Solve an ARC AGI puzzle. Training examples:\\n{training_examples}\\n\\nPredict output for EACH input as JSON [[...]]:\\n{problem_inputs}"
    response = llm(prompt)

    grids = [json.loads(g) for g in re.findall(r"\\[\\[.*?\\]\\]", response.replace("\\n", ""))]
    n_train = len(train_inputs)
    return {
        "train": grids[:n_train],
        "test": [[g] for g in grids[n_train:]]
    }
'''


def _load_dataset() -> tuple[list[Example], list[Example], list[Example]]:
    """Load ARC-AGI dataset, returning (train, val, test) as Example lists."""
    import random

    from datasets import load_dataset

    ds = load_dataset("dataartist/arc-agi")

    examples: list[Example] = []
    for split_name in ds:
        for item in ds[split_name]:
            examples.append(Example(
                id=item.get("id", f"arc_{len(examples)}"),
                inputs={
                    "train_in": item["train_in"],
                    "train_out": item["train_out"],
                    "test_in": item["test_in"],
                },
                expected=item.get("test_out"),
            ))

    random.Random(0).shuffle(examples)
    n = len(examples)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return examples[:train_end], examples[train_end:val_end], examples[val_end:]


def evaluate(candidate: str, example: Example, model_id: str = "openrouter/google/gemini-3-flash-preview") -> tuple[float, dict[str, Any]]:
    """Evaluate an ARC-AGI agent on a single puzzle."""
    from examples.arc_agi.utils import run_agent

    result = run_agent(
        agent_code=candidate,
        train_in=example.inputs["train_in"],
        train_out=example.inputs["train_out"],
        test_in=example.inputs["test_in"],
        test_out=example.expected,
        model_id=model_id,
        max_llm_calls=10,
    )

    llms = result["llms"]
    score = result["test_score"] - 0.1 * (llms.total_cost > 1.0)

    return score, {
        "score": score,
        "problem_id": example.id,
        "training_score": result["training_score"],
        "test_score": result["test_score"],
        "cost": llms.total_cost,
        "error": result["error"],
    }


def _make_task() -> Task:
    """Build the ARC-AGI task (lazy dataset loading)."""
    train_set, val_set, test_set = _load_dataset()
    return Task(
        name="arc_agi",
        description=DESCRIPTION,
        initial_candidate=INITIAL_CANDIDATE,
        eval_fn=evaluate,
        train_set=train_set,
        test_set=test_set,
        metadata={
            "type": "generalization",
            "candidate_type": "code",
            "val_set": val_set,
            "objective": "Build an ARC-AGI agent that maximizes test accuracy.",
        },
    )


register_task_factory("arc_agi", _make_task)
