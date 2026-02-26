#!/usr/bin/env python3
"""ARC-AGI Agent Optimization with GEPA."""

import os

from examples.arc_agi.utils import (
    BACKGROUND,
    OBJECTIVE,
    evaluate_on_testset,
    load_arc_dataset,
    run_agent,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)

LLM_MODEL = "openrouter/google/gemini-3-flash-preview"

SEED_AGENT_CODE = """
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
"""


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    """Evaluate an agent on a single ARC problem."""
    result = run_agent(
        agent_code=candidate,
        train_in=example.train_in,
        train_out=example.train_out,
        test_in=example.test_in,
        test_out=example.test_out or None,
        model_id=LLM_MODEL,
        max_llm_calls=10,
    )

    llms = result["llms"]
    score = result["test_score"] - 0.1 * (llms.total_cost > 1.0)  # score with a cost penalty

    side_info: SideInfo = {
        "score": score,
        "problem_id": example.problem_id,
        "agent_code": candidate,
        "training_score": result["training_score"],
        "test_score": result["test_score"],
        "cost": llms.total_cost,
        "error": result["error"],
        "train_examples": result["train_examples"],
        "test_examples": result["test_examples"],
        **llms.get_traces(),  # llm costs, number of calls, model outputs, etc.
    }

    print(
        f"[{example.problem_id}] train={result['training_score']:.0%} test={result['test_score']:.0%} cost=${llms.total_cost:.4f} llm_calls={len(llms.calls)}"
    )

    return score, side_info


# =============================================================================
# MAIN
# =============================================================================


def main():
    log_dir = "outputs/arc_agi"
    os.makedirs(log_dir, exist_ok=True)

    train_set, val_set, test_set = load_arc_dataset()

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=3000,
            parallel=True,
            max_workers=64,
            cache_evaluation=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=LLM_MODEL,  # We use Gemini 3 Flash, but a stronger model will to better results
        ),
    )

    result = optimize_anything(
        seed_candidate=SEED_AGENT_CODE,
        evaluator=evaluate,
        dataset=train_set,
        valset=val_set,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )

    best_agent_code = result.best_candidate
    print(f"\nBest score (on val): {result.val_aggregate_scores[result.best_idx]:.4f}")

    with open(f"{log_dir}/best_agent.py", "w") as f:
        f.write(best_agent_code)
    print(f"Saved: {log_dir}/best_agent.py")

    # Evaluate on test set
    print("\nEvaluating Baseline (Seed Agent)...")
    baseline_acc = evaluate_on_testset(SEED_AGENT_CODE, test_set, model_id=LLM_MODEL)

    print("\nEvaluating Best Agent...")
    optimized_acc = evaluate_on_testset(best_agent_code, test_set, model_id=LLM_MODEL)

    print(f"\nBaseline accuracy:  {baseline_acc:.1%}")
    print(f"Optimized accuracy: {optimized_acc:.1%}")
    print(f"Improvement:        {optimized_acc - baseline_acc:+.1%}")


if __name__ == "__main__":
    main()
