"""Vendored runtime for booth mode — no dependency on the gepa examples directory.

Lifted from gepa/examples/arc_agi/utils.py (TrackedLLM, compare_grid,
evaluate_predictions, evaluate_test, run_agent). Imported by the notebook only
when DEMO=True. Reviewer mode never touches this file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import completion

litellm.suppress_debug_info = True


@dataclass
class TrackedLLM:
    model_id: str = "openrouter/google/gemini-3-flash-preview"
    max_llm_calls: int = 20
    reasoning_effort: str | None = "high"
    calls: list[dict] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(c.get("cost", 0.0) for c in self.calls)

    def __call__(self, prompt: str, temperature: float = 1.0) -> str:
        if len(self.calls) >= self.max_llm_calls:
            raise RuntimeError(f"LLM budget exhausted ({self.max_llm_calls} calls)")
        start = time.time()
        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
        resp = completion(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        self.calls.append({"prompt": prompt, "response": content, "cost": cost, "duration": time.time() - start})
        return content


def compare_grid(pred, gold) -> tuple[bool, str]:
    if not isinstance(pred, list):
        return False, f"The matrix must be a List[List[int]], found {type(pred).__name__}. The correct matrix is {gold}."
    n = len(pred)
    if n == 0 or not isinstance(pred[0], list):
        return False, f"Malformed matrix. The correct matrix is {gold}."
    m = len(pred[0])
    for i in range(n):
        if not isinstance(pred[i], list) or len(pred[i]) != m:
            return False, f"Staggered matrix. The correct matrix is {gold}."
        for j in range(m):
            if not isinstance(pred[i][j], (int, float)):
                return False, f"Bad element at ({i},{j}). The correct matrix is {gold}."
    if (n, m) != (len(gold), len(gold[0])):
        return False, f"Shape ({n}, {m}) != expected ({len(gold)}, {len(gold[0])}). The correct matrix is {gold}."
    wrong = [(i, j) for i in range(n) for j in range(m) if int(pred[i][j]) != gold[i][j]]
    if not wrong:
        return True, "Correct!"
    if len(wrong) < 10:
        return False, f"Incorrect values at indices: {wrong}. The correct matrix is {gold}."
    return False, f"Incorrect values at {len(wrong)} positions. The correct matrix is {gold}."


def _eval_preds(preds, golds):
    if not preds:
        return 0.0, [{"correct": False, "feedback": "No prediction"} for _ in golds]
    results = []
    for i, gold in enumerate(golds):
        if i < len(preds) and preds[i] is not None:
            ok, fb = compare_grid(preds[i], gold)
        else:
            ok, fb = False, "No prediction"
        results.append({"correct": ok, "feedback": fb})
    return sum(r["correct"] for r in results) / len(results), results


def _eval_test(test_preds, test_out):
    if not test_preds:
        return 0.0, [{"correct": False, "feedback": "No prediction"} for _ in test_out]
    norm = [a[:2] if isinstance(a, list) else [a] for a in test_preds]
    a1 = [x[0] if x else None for x in norm]
    a2 = [x[1] if len(x) > 1 else None for x in norm]
    _, r1 = _eval_preds(a1, test_out)
    _, r2 = _eval_preds(a2, test_out)
    results = [
        {"correct": r1[i]["correct"] or r2[i]["correct"], "feedback": r1[i]["feedback"]}
        for i in range(len(test_out))
    ]
    return (1.0 if all(r["correct"] for r in results) else 0.0), results


def run_agent(agent_code: str, train_in, train_out, test_in, test_out, model_id: str, max_llm_calls: int = 10) -> dict:
    llms = TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls)
    try:
        ns: dict[str, Any] = {}
        exec(agent_code, ns)
        result = ns["solve"](train_in, train_out, test_in, llms)
        train_preds, test_preds = result.get("train", []), result.get("test", [])
    except Exception as e:
        return {"training_score": 0.0, "test_score": 0.0, "error": str(e),
                "train_examples": [], "test_examples": [], "llms": llms}

    train_score, tr = _eval_preds(train_preds, train_out)
    test_score, te = _eval_test(test_preds, test_out) if test_out else (0.0, [])

    train_examples = [
        {"input": train_in[i], "gold": train_out[i],
         "prediction": train_preds[i] if i < len(train_preds) else None,
         "correct": tr[i]["correct"], "feedback": tr[i]["feedback"]}
        for i in range(len(train_in))
    ]
    test_examples = [
        {"input": test_in[i], "gold": test_out[i] if test_out else None,
         "prediction": test_preds[i] if i < len(test_preds) else None,
         "correct": te[i]["correct"] if i < len(te) else False,
         "feedback": te[i]["feedback"] if i < len(te) else "No prediction"}
        for i in range(len(test_in))
    ]
    return {"training_score": train_score, "test_score": test_score, "error": None,
            "train_examples": train_examples, "test_examples": test_examples, "llms": llms}
