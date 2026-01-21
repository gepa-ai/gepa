#!/usr/bin/env python3
"""
Blackbox optimization with GEPA

Target candidate: code and refiner
"""

import os
import dspy
import time

from experiments.polynomial.gepa2.config import (
    parse_arguments,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)
from experiments.polynomial.problems import problems

# from experiments.polynomial.gepa.llm import create_reflection_lm, create_custom_proposer
from experiments.polynomial.gepa2.prompt import REFLECTION_PROMPT
from experiments.polynomial.gepa2.evaluator import FitnessEvaluator


BASELINE_CODE_TEMPLATE = """
# EVOLVE-BLOCK-START
import numpy as np

def solve(dim, total_evaluation_budgets, bounds, prev_best_x):
    # Initialize x within bounds (use midpoint of each dimension's bounds)
    bounds_arr = np.array(bounds)
    if prev_best_x is not None:
        x = np.array(prev_best_x)
    x = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1])

    y = objective_function(np.array(x))
    print("y: ", y)
    return x

# EVOLVE-BLOCK-END

# Always include the code below without changing or deleting it.
if __name__ == "__main__":
    # Store results in global variables so that a helper function that runs this code can capture the results
    # x will be the numpy array of shape (dim,) that I will pass to the function to evaluate.
    global x
    x = solve(dim, total_evaluation_budgets, bounds, prev_best_x)
"""


def create_bbox_dataset(problem_index):
    """Create dataset for blackbox optimization problems.

    Args:
        problem_index: Index of the problem in problems list (0-55)

    Returns:
        List of dspy.Example objects for blackbox optimization
    """
    if problem_index < 0 or problem_index >= len(problems):
        raise ValueError(
            f"Problem index {problem_index} out of range. Valid: 0-{len(problems) - 1}"
        )

    # problems list contains already-instantiated problem objects
    problem = problems[problem_index]

    description = f"""This is a blackbox optimization problem. There is a function that we need to maximize.
    Given a numpy array of shape (dim,), the function returns a scalar value, which you need to maximize.
    The bounds for x are {problem.bounds}.
    The dimension of the problem is {problem.dim}.
    """

    return [
        dspy.Example(
            {
                "problem_description": description,
                "baseline_code": BASELINE_CODE_TEMPLATE,
                "problem_index": problem_index,  # This is hidden from the LLMs for fairness
            }
        ).with_inputs("problem_description", "baseline_code", "problem_index")
    ]


def main():
    args = parse_arguments()

    # Get problem index (required)
    if args.problem_index is None:
        raise ValueError("Must provide --problem-index")
    problem_index = args.problem_index

    if args.run_name:
        run_folder_name = args.run_name
    else:
        run_folder_name = time.strftime("%y%m%d_%H:%M:%S")
    # Sanitize model name (remove "openai/" prefix, replace "/" with "_")
    model_name = args.llm_model.replace("openai/", "").replace("/", "_")
    log_dir = f"outputs/artifacts/polynomial/full_eval_gepa/problem_{problem_index}/{model_name}/{args.seed}/{run_folder_name}"
    os.makedirs(log_dir, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    dataset = create_bbox_dataset(problem_index=problem_index)

    seed_candidate = {
        "code": BASELINE_CODE_TEMPLATE,
    }

    # Calculate max_metric_calls from num_proposals if not explicitly set
    if args.max_metric_calls is None:
        num_candidates_to_generate = args.num_proposals
        max_metric_calls = num_candidates_to_generate * 100 // 18
    else:
        max_metric_calls = args.max_metric_calls

    evaluation_budget = (
        args.evaluation_budget if hasattr(args, "evaluation_budget") else 100
    )

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            seed=0,
            max_metric_calls=max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
            frontier_type="objective",
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_prompt_template=REFLECTION_PROMPT,
            reflection_lm="openai/gpt-5",
            skip_perfect_score=False,
        ),
    )

    # Run GEPA optimization
    print("\n" + "=" * 70)
    print("Running GEPA Bounding Box Optimization")
    print("=" * 70 + "\n")

    evaluator = FitnessEvaluator(
        timeout=args.timeout,
        evaluation_budget=evaluation_budget,
        log_dir=log_dir,
        base_seed=args.seed,
    )

    optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=evaluator.evaluate,
        dataset=dataset,
        config=gepa_config,
    )

    # Save trajectory files (candidate_history and evaluation_call_history)
    evaluator.save(verbose=True)

    # Save results
    print("\n" + "=" * 70)
    print("✓ Optimization Complete!")
    print("=" * 70)
    print(f"Log directory: {log_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
