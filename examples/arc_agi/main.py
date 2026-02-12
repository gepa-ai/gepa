#!/usr/bin/env python3
"""
ARC-AGI Agent Optimization with GEPA.

Evolves agent code that uses an LLM to solve ARC puzzles.
All LLM calls go through TrackedLLM for cost/token tracking.
"""

import os

from examples.arc_agi.evaluate import run_agent
from examples.arc_agi.utils import BACKGROUND, OBJECTIVE, load_arc_dataset
from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

LLM_MODEL = "openrouter/google/gemini-3-flash-preview"
MAX_LLM_CALLS = 10  # Max LLM calls per problem
MAX_METRIC_CALLS = 4000
COST_PENALTY = 0.1  # Penalty when cost > $1.0


# =============================================================================
# SEED AGENT CODE
# =============================================================================

SEED_AGENT_CODE = '''
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


# =============================================================================
# FITNESS FUNCTION
# =============================================================================

def fitness_fn(candidate: dict, **kwargs) -> tuple[float, SideInfo]:
    """Evaluate an agent on a single ARC problem.

    Caching is handled by the adapter via cache_evaluation=True in EngineConfig.
    """
    ex = kwargs["example"]
    agent_code = candidate["agent_code"]

    # Run agent
    result = run_agent(
        agent_code=agent_code,
        train_in=ex.train_in,
        train_out=ex.train_out,
        test_in=ex.test_in,
        test_out=ex.test_out or None,
        model_id=LLM_MODEL,
        max_llm_calls=MAX_LLM_CALLS,
    )

    # Compute score
    llm = result["llm"]
    score = result["test_score"] - COST_PENALTY * (llm.total_cost > 1.0)

    # Build side_info
    side_info: SideInfo = {
        "score": score,
        "problem_id": ex.problem_id,
        "agent_code": agent_code,
        "training_score": result["training_score"],
        "test_score": result["test_score"],
        "cost": llm.total_cost,
        "error": result["error"],
        "train_examples": result["train_examples"],
        "test_examples": result["test_examples"],
        **llm.get_side_info(),
    }

    print(f"[{ex.problem_id}] train={result['training_score']:.0%} test={result['test_score']:.0%} cost=${llm.total_cost:.4f} llm_calls={len(llm.calls)}")

    return score, side_info


# =============================================================================
# MAIN
# =============================================================================

def main():
    log_dir = "outputs/arc_agi"
    os.makedirs(log_dir, exist_ok=True)

    print("ARC-AGI Agent Optimization with GEPA")
    print(f"LLM Model: {LLM_MODEL} | Max LLM calls/problem: {MAX_LLM_CALLS} | Max metric calls: {MAX_METRIC_CALLS}")
    print(f"Output: {log_dir}\n")

    train_set, val_set, test_set = load_arc_dataset()

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=MAX_METRIC_CALLS,
            parallel=True,
            max_workers=64,
            cache_evaluation=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=LLM_MODEL,
        ),
        refiner=None,
    )

    seed_candidate = {"agent_code": SEED_AGENT_CODE}

    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        dataset=train_set,
        valset=val_set,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )

    print(f"\nBest score (on train): {result.val_aggregate_scores[result.best_idx]:.4f}")

    with open(f"{log_dir}/best_agent.py", "w") as f:
        f.write(result.best_candidate["agent_code"])
    print(f"Saved: {log_dir}/best_agent.py")


if __name__ == "__main__":
    main()
