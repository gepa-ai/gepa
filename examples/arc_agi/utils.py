"""ARC-AGI utilities: dataset loading, prompts, eval, and LLM tracking."""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
- You can also use up to 10 LLM calls to solve the problem.
- Freely explore diverse strategies like multi agent systems, ensembles, voting, etc.

LLM cost:
- You are allowed to build an agent system with up to 10 LLM calls and total of $0.8~1.0 LLM cost per problem.

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

    def get_traces(self) -> dict[str, Any]:
        """Get traces for GEPA reflection. Trajectory sampled to max 10 calls."""
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


def compare_grid(pred, gold) -> tuple[bool, str]:
    """Compare predicted grid to gold. Returns (is_correct, feedback)."""
    if not isinstance(pred, list):
        return (
            False,
            f"The matrix must be a List[List[int]], found {type(pred).__name__}. The correct matrix is {gold}.",
        )

    n = len(pred)
    if n == 0:
        return False, f"The matrix must have at least one row. The correct matrix is {gold}."

    if not isinstance(pred[0], list):
        return False, f"The matrix must be a 2D list. Row 0 is {type(pred[0]).__name__}. The correct matrix is {gold}."

    m = len(pred[0])
    if m == 0:
        return False, f"The matrix must have at least one column. The correct matrix is {gold}."

    # Structural and type checks
    for i in range(n):
        if not isinstance(pred[i], list):
            return False, f"Row {i} must be a list, found {type(pred[i]).__name__}. The correct matrix is {gold}."
        if len(pred[i]) != m:
            return (
                False,
                f"The matrix is staggered. Row 0 has {m} columns, but row {i} has {len(pred[i])} columns. The correct matrix is {gold}.",
            )
        for j in range(m):
            if not isinstance(pred[i][j], (int, float)):
                return (
                    False,
                    f"Element at ({i}, {j}) must be an int, found {type(pred[i][j]).__name__}. The correct matrix is {gold}.",
                )

    # Shape check
    pred_shape = (n, m)
    gold_shape = (len(gold), len(gold[0]))

    if pred_shape != gold_shape:
        return False, f"Shape {pred_shape} != expected {gold_shape}. The correct matrix is {gold}."

    # Value check
    wrong = []
    for i in range(len(gold)):
        for j in range(len(gold[0])):
            if int(pred[i][j]) != gold[i][j]:
                wrong.append((i, j))

    if not wrong:
        return True, "Correct!"

    if len(wrong) < 10:
        return False, f"Incorrect values at indices: {wrong}. The correct matrix is {gold}."
    return False, f"Incorrect values at {len(wrong)} positions. The correct matrix is {gold}."


def evaluate_predictions(preds: list, golds: list) -> tuple[float, list[dict]]:
    """Evaluate single predictions against gold. Returns (score, results)."""
    if not preds:
        return 0.0, [{"idx": i, "correct": False, "feedback": "No prediction"} for i in range(len(golds))]

    results = []
    for i in range(len(golds)):
        if i < len(preds) and preds[i] is not None:
            correct, feedback = compare_grid(preds[i], golds[i])
        else:
            correct, feedback = False, "No prediction"
        results.append({"idx": i, "correct": correct, "feedback": feedback})

    score = sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    return score, results


def evaluate_test(test_preds: list[list], test_out: list) -> tuple[float, list[dict]]:
    """Evaluate test with up to 2 attempts per example. Pass if ANY attempt correct."""
    if not test_preds:
        return 0.0, [{"idx": i, "correct": False, "feedback": "No prediction"} for i in range(len(test_out))]

    # Normalize: ensure each entry is a list of attempts
    normalized = [a[:2] if isinstance(a, list) else [a] for a in test_preds]

    # Evaluate each attempt using evaluate_predictions
    attempt1 = [attempts[0] if attempts else None for attempts in normalized]
    attempt2 = [attempts[1] if len(attempts) > 1 else None for attempts in normalized]

    _, results1 = evaluate_predictions(attempt1, test_out)
    _, results2 = evaluate_predictions(attempt2, test_out)

    # Aggregate: pass if ANY attempt correct
    results = []
    for i in range(len(test_out)):
        r1, r2 = results1[i], results2[i]
        correct = r1["correct"] or r2["correct"]
        feedback = r1["feedback"] if r1["correct"] else (r2["feedback"] if r2["correct"] else r1["feedback"])
        results.append({"idx": i, "correct": correct, "feedback": feedback})

    # ARC-AGI: must get ALL test examples correct to solve the problem (binary score)
    all_correct = all(r["correct"] for r in results)
    score = 1.0 if all_correct else 0.0
    return score, results


def run_agent(
    agent_code: str,
    train_in: list,
    train_out: list,
    test_in: list,
    test_out: list | None,
    model_id: str,
    max_llm_calls: int,
    reasoning_effort: str | None = None,
) -> dict:
    """Run agent and return evaluation results.

    Agent's solve() should return:
    {
        "train": [grid, ...],              # 1 prediction per train example
        "test": [[grid, grid], ...],       # up to 2 attempts per test example
    }
    """
    llms = TrackedLLM(
        model_id=model_id,
        max_llm_calls=max_llm_calls,
        reasoning_effort=reasoning_effort,
    )

    try:
        namespace = {}
        exec(agent_code, namespace)
        result = namespace["solve"](train_in, train_out, test_in, llms)

        train_preds = result.get("train", [])
        test_preds = result.get("test", [])

    except Exception as e:
        return {
            "training_score": 0.0,
            "test_score": 0.0,
            "error": str(e),
            "train_examples": [],
            "test_examples": [],
            "llms": llms,
        }

    # Evaluate
    training_score, train_results = evaluate_predictions(train_preds, train_out)

    if test_out:
        test_score, test_results = evaluate_test(test_preds, test_out)
    else:
        test_score, test_results = 0.0, []

    # Build detailed examples for reflection
    train_examples = []
    for i, (inp, gold, res) in enumerate(zip(train_in, train_out, train_results)):
        pred = train_preds[i] if i < len(train_preds) else None
        train_examples.append(
            {
                "input": inp,
                "gold": gold,
                "prediction": pred,
                "correct": res["correct"],
                "feedback": res["feedback"],
            }
        )

    test_examples = []
    for i, res in enumerate(test_results):
        inp = test_in[i] if i < len(test_in) else None
        gold = test_out[i] if test_out and i < len(test_out) else None
        pred = test_preds[i] if i < len(test_preds) else None
        test_examples.append(
            {
                "input": inp,
                "gold": gold,
                "prediction": pred,
                "correct": res["correct"],
                "feedback": res["feedback"],
            }
        )

    return {
        "training_score": training_score,
        "test_score": test_score,
        "error": None,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "llms": llms,
    }


def evaluate_on_testset(
    agent_code: str,
    test_set: list,
    model_id: str,
    max_llm_calls: int = 10,
    max_workers: int = 32,
) -> float:
    """Evaluate agent on test set in parallel. Returns accuracy (fraction solved)."""

    def eval_one(ex):
        r = run_agent(
            agent_code=agent_code,
            train_in=ex.train_in,
            train_out=ex.train_out,
            test_in=ex.test_in,
            test_out=ex.test_out or None,
            model_id=model_id,
            max_llm_calls=max_llm_calls,
        )
        score = r["test_score"]
        cost = r["llms"].total_cost
        status = "ok" if score == 1.0 else "X"
        print(f"  [{ex.problem_id}] {status} test={score:.0%} cost=${cost:.4f}", flush=True)
        return score

    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_one, ex): ex for ex in test_set}
        for future in as_completed(futures):
            try:
                scores.append(future.result())
            except Exception as e:
                print(f"  [{futures[future].problem_id}] ERROR: {e}", flush=True)
                scores.append(0.0)

    solved = sum(1 for s in scores if s == 1.0)
    print(f"  Solved: {solved}/{len(scores)}")
    return solved / len(scores) if scores else 0.0

