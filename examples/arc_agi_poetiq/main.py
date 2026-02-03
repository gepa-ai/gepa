#!/usr/bin/env python3
"""
ARC-AGI Agent Optimization with GEPA.

Evolves agent code that uses an LLM to solve ARC puzzles.
All LLM calls go through TrackedLLM for cost/token tracking.
"""

import argparse
import json
import os
import time
from pathlib import Path

import dspy

from examples.arc_agi_poetiq.cache import EvalCache
from examples.arc_agi_poetiq.evaluate import run_agent
from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)

# Local ARC data (from poetiq clone)
ARC_DATA_DIR = Path(__file__).parent.parent.parent / "external" / "poetiq" / "data" / "arc-prize-2024"


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
# SEED AGENT CODE
# =============================================================================

SEED_AGENT_CODE = '''
import json, re

def solve(train_inputs, train_outputs, test_inputs, llm):
    training_examples = "\\n".join(f"Input: {i}\\nOutput: {o}" for i, o in zip(train_inputs, train_outputs))
    problem_inputs = "\\n".join(f"Input {i}: {x}" for i, x in enumerate(train_inputs + test_inputs))

    prompt = f"ARC puzzle. Training examples:\\n{training_examples}\\n\\nPredict output for EACH input as JSON [[...]]:\\n{problem_inputs}"
    response = llm(prompt)

    grids = [json.loads(g) for g in re.findall(r"\\[\\[.*?\\]\\]", response.replace("\\n", ""))]
    n_train = len(train_inputs)
    return {
        "train": grids[:n_train],
        "test": [[g] for g in grids[n_train:]]
    }
'''


# =============================================================================
# GEPA INTEGRATION
# =============================================================================

def load_arc_dataset(seed: int = 0) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Load ARC problems with train/val/test split.

    Uses 'training' challenges for train+val, 'evaluation' challenges for test.
    Returns (train_set, val_set, test_set).
    """
    import random

    def load_split(split: str) -> list[dspy.Example]:
        with open(ARC_DATA_DIR / f"arc-agi_{split}_challenges.json") as f:
            challenges = json.load(f)
        
        solutions = {}
        sol_path = ARC_DATA_DIR / f"arc-agi_{split}_solutions.json"
        if sol_path.exists():
            with open(sol_path) as f:
                solutions = json.load(f)

        return [
            dspy.Example(
                problem_id=pid,
                train_in=[ex["input"] for ex in task["train"]],
                train_out=[ex["output"] for ex in task["train"]],
                test_in=[ex["input"] for ex in task["test"]],
                test_out=solutions.get(pid, []),
            ).with_inputs("problem_id", "train_in", "train_out", "test_in", "test_out")
            for pid, task in challenges.items()
        ]

    # Load training split and shuffle
    trainval = load_split("training")
    random.Random(seed).shuffle(trainval)

    # Split: last 200 for val, rest for train
    val_set = trainval[-200:]
    train_set = trainval[:-200]

    # Evaluation split becomes test set
    test_set = load_split("evaluation")

    return train_set, val_set, test_set


def create_fitness_fn(model_id: str, max_llm_calls: int, cost_penalty: float, cache: EvalCache | None, reasoning_effort: str | None = None):
    """Create fitness function for GEPA."""

    def fitness_fn(candidate: dict, **kwargs) -> tuple[float, dict, SideInfo]:
        ex = kwargs["example"]
        agent_code = candidate["agent_code"]

        # Check cache
        if cache:
            cached = cache.get(agent_code, ex.problem_id)
            if cached:
                side_info = cached["side_info"]
                print(f"[{ex.problem_id}] CACHED train={side_info['training_score']:.0%} test={side_info['test_score']:.0%} cost=${side_info['cost']:.4f} llm_calls={side_info['llm_calls']}")
                return cached["score"], side_info, side_info

        # Run agent
        result = run_agent(
            agent_code=agent_code,
            train_in=ex.train_in,
            train_out=ex.train_out,
            test_in=ex.test_in,
            test_out=ex.test_out or None,
            model_id=model_id,
            max_llm_calls=max_llm_calls,
            reasoning_effort=reasoning_effort,
        )

        # Compute score
        llm = result["llm"]
        score = result["test_score"] - cost_penalty * (llm.total_cost > 1.0)

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

        # Store in cache
        if cache:
            cache.put(agent_code, ex.problem_id, {
                "score": score,
                "side_info": side_info,
            })

        print(f"[{ex.problem_id}] train={result['training_score']:.0%} test={result['test_score']:.0%} cost=${llm.total_cost:.4f} llm_calls={len(llm.calls)}")

        return score, side_info, side_info

    return fitness_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openrouter/google/gemini-3-flash-preview")
    parser.add_argument("--max-llm-calls", type=int, default=10)
    parser.add_argument("--max-metric-calls", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-penalty", type=float, default=0.1,
                        help="Penalty per dollar spent (default: 0.1, meaning going over $0.8 reduces score by 0.1)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for output directory (default: auto-generated timestamp)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing run (e.g., gemini-3-flash_260201_181722)")
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["low", "medium", "high"],
                        help="Reasoning effort for supported models (o1/o3, gemini-2.5)")
    parser.add_argument("--no-cache", action="store_true", help="Disable evaluation caching")
    args = parser.parse_args()

    if args.resume:
        run_name = args.resume
        log_dir = f"outputs/artifacts/arc_agi_poetiq/{run_name}"
        if not os.path.exists(f"{log_dir}/gepa_state.bin"):
            raise ValueError(f"Cannot resume: {log_dir}/gepa_state.bin not found")
        print(f"Resuming from: {log_dir}")
    else:
        run_name = args.run_name or f"agent_{time.strftime('%y%m%d_%H%M%S')}"
        log_dir = f"outputs/artifacts/arc_agi_poetiq/{run_name}"
    cache_dir = f"{"/".join(log_dir.split("/")[:-1])}/cache"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    cache = None if args.no_cache else EvalCache(cache_dir)

    print(f"Model: {args.model}")
    print(f"Max LLM calls/problem: {args.max_llm_calls}")
    print(f"Cost penalty: {args.cost_penalty}/$ (score -= penalty * cost)")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Output: {log_dir}\n")

    train_set, val_set, test_set = load_arc_dataset(seed=args.seed)
    print(f"Dataset: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}\n")

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=args.max_metric_calls,
            seed=args.seed,
            parallel=True,
            max_workers=64,
            cache_evaluation=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            reflection_lm=args.model,
            skip_perfect_score=True,
            perfect_score=1.0,
        ),
    )

    seed_candidate = {"agent_code": SEED_AGENT_CODE}

    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=create_fitness_fn(args.model, args.max_llm_calls, args.cost_penalty, cache, args.reasoning_effort),
        dataset=train_set,
        valset=val_set,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )

    print(f"\nBest score (on train): {result.val_aggregate_scores[result.best_idx]:.4f}")

    if cache:
        stats = cache.stats()
        print(f"Cache: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']:.0%} hit rate)")

    with open(f"{log_dir}/best_agent.py", "w") as f:
        f.write(result.best_candidate["agent_code"])
    print(f"Saved: {log_dir}/best_agent.py")


if __name__ == "__main__":
    main()
