"""Evaluation logic for ARC-AGI agent."""

from examples.arc_agi.tracked_llm import TrackedLLM


def compare_grid(pred, gold) -> tuple[bool, str]:
    """Compare predicted grid to gold. Returns (is_correct, feedback)."""
    if not isinstance(pred, list):
        return False, f"The matrix must be a List[List[int]], found {type(pred).__name__}. The correct matrix is {gold}."
    
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
            return False, f"The matrix is staggered. Row 0 has {m} columns, but row {i} has {len(pred[i])} columns. The correct matrix is {gold}."
        for j in range(m):
            if not isinstance(pred[i][j], (int, float)):
                return False, f"Element at ({i}, {j}) must be an int, found {type(pred[i][j]).__name__}. The correct matrix is {gold}."

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
    llm = TrackedLLM(
        model_id=model_id,
        max_llm_calls=max_llm_calls,
        reasoning_effort=reasoning_effort,
    )

    try:
        namespace = {}
        exec(agent_code, namespace)
        result = namespace["solve"](train_in, train_out, test_in, llm)

        train_preds = result.get("train", [])
        test_preds = result.get("test", [])

    except Exception as e:
        return {
            "training_score": 0.0,
            "test_score": 0.0,
            "error": str(e),
            "train_examples": [],
            "test_examples": [],
            "llm": llm,
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
        train_examples.append({
            "input": inp,
            "gold": gold,
            "prediction": pred,
            "correct": res["correct"],
            "feedback": res["feedback"],
        })

    test_examples = []
    for i, res in enumerate(test_results):
        inp = test_in[i] if i < len(test_in) else None
        gold = test_out[i] if test_out and i < len(test_out) else None
        pred = test_preds[i] if i < len(test_preds) else None
        test_examples.append({
            "input": inp,
            "gold": gold,
            "prediction": pred,
            "correct": res["correct"],
            "feedback": res["feedback"],
        })

    return {
        "training_score": training_score,
        "test_score": test_score,
        "error": None,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "llm": llm,
    }


def format_evaluation_feedback(result: dict) -> str:
    """Format per-example evaluation details."""
    lines = []

    for r in result.get("train_examples", []):
        status = "✓" if r["correct"] else "✗"
        lines.append(f"Train: {status} {r['feedback']}")

    for r in result.get("test_examples", []):
        status = "✓" if r["correct"] else "✗"
        lines.append(f"Test: {status} {r['feedback']}")

    return "\n".join(lines)
