"""ARC-AGI utilities: dataset loading, prompts, and LLM tracking."""

import random
import time
from dataclasses import dataclass, field
from typing import Any

import dspy
import litellm
from datasets import load_dataset
from litellm import completion

litellm.suppress_debug_info = True


# =============================================================================
# PROMPTS
# =============================================================================

BACKGROUND = """You are optimizing an ARC-AGI solving agent.

ARC-AGI task format:
- Each task has training examples (input/output pairs) and test inputs
- The (multi) agent(s) must infer the transformation pattern from training examples
- Competition allows maximum of 2 parallel output attempts per test input (pass if either matches)
- You can also use up to 20 LLM calls to solve the problem.
- Freely explore diverse strategies like multi agent systems, ensembles, voting, etc.

LLM cost:
- You are allowed to build an agent system with up to 20 LLM calls and total of $0.8~1.0 LLM cost per problem.

The agent receives:
- train_in, train_out: Training examples (list of 2D grids)
- test_in: Test inputs (no ground truth given to agent)
- llm: Callable for LLM queries with token/call tracking

The agent must return:
{
    "train": [grid, ...],           # 1 prediction per train example
    "test": [[grid, grid], ...],    # up to 2 attempts per test example
}

We evaluate on both training (training_score) and test (test_score with 2 attempts)."""

OBJECTIVE = """Build an ARC-AGI agent program that maximizes a test score."""


# =============================================================================
# DATASET
# =============================================================================

def load_arc_dataset(seed: int = 0):
    """Load ARC-AGI dataset from HuggingFace.

    Returns (train_set, val_set, test_set) as dspy.Example lists.
    Format matches original: train_in, train_out, test_in, test_out
    """
    ds = load_dataset("dataartist/arc-agi")

    def make_example(ex):
        return dspy.Example(
            problem_id=ex["id"],
            train_in=[t["input"] for t in ex["train"]],
            train_out=[t["output"] for t in ex["train"]],
            test_in=[t["input"] for t in ex["test"]],
            test_out=[t["output"] for t in ex["test"]],
        ).with_inputs("problem_id", "train_in", "train_out", "test_in", "test_out")

    trainset = [make_example(ex) for ex in ds["training"]]
    testset = [make_example(ex) for ex in ds["evaluation"]]

    random.Random(seed).shuffle(trainset)

    val_set = trainset[-200:]
    train_set = trainset[:-200]
    test_set = testset

    print(f"Dataset: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    return train_set, val_set, test_set


# =============================================================================
# TRACKED LLM
# =============================================================================

@dataclass
class TrackedLLM:
    """Simple LLM wrapper that tracks calls and costs."""

    model_id: str = "openrouter/google/gemini-3-flash-preview"
    max_llm_calls: int = 20
    reasoning_effort: str = "high"

    # Tracking
    calls: list[dict] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(c.get("cost", 0.0) for c in self.calls)

    def __call__(self, prompt: str, temperature: float = 1.0) -> str:
        """Make an LLM call. Raises if budget exhausted."""
        if len(self.calls) >= self.max_llm_calls:
            raise RuntimeError(f"LLM budget exhausted ({self.max_llm_calls} calls)")

        start = time.time()

        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        # OpenRouter format: {"reasoning": {"effort": "high"}} for o1/o3/grok/gemini
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}

        resp = completion(**kwargs)

        duration = time.time() - start
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        # Get cost from litellm
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0

        call_data = {
            "prompt": prompt,
            "response": content,
            "cost": cost,
            "duration": duration,
        }
        if reasoning:
            call_data["reasoning"] = reasoning

        self.calls.append(call_data)
        return content

    def get_side_info(self) -> dict[str, Any]:
        """Get side info for GEPA reflection. Trajectory sampled to max 10 calls."""
        calls_to_include = self.calls
        sampled = False

        if len(self.calls) > 10:
            calls_to_include = random.sample(self.calls, 10)
            sampled = True

        trajectory = []
        for c in calls_to_include:
            entry = {
                "prompt": c["prompt"],
                "response": c["response"],
                "cost": c.get("cost", 0.0),
            }
            if c.get("reasoning"):
                entry["reasoning"] = c["reasoning"]
            trajectory.append(entry)

        result = {
            "llm_calls": len(self.calls),
            "llm_budget": self.max_llm_calls,
            "total_cost": self.total_cost,
            "trajectory": trajectory,
        }

        if sampled:
            result["trajectory_note"] = f"Randomly sampled 10 of {len(self.calls)} LLM calls"

        return result
