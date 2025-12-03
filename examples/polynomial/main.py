#!/usr/bin/env python3
"""
Blackbox optimization with GEPA

Target candidate: code and refiner
"""

import os
import dspy

from examples.polynomial.config import (
    parse_arguments,
    get_log_directory,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)
from examples.polynomial.evalset import problems
from examples.polynomial.llm import REFLECTION_PROMPT, create_reflection_lm
from examples.polynomial.evaluator import create_fitness_function


BASELINE_CODE_TEMPLATE = """
import numpy as np

def solve(dim):
    # dim is available here because we pass it in global_vars
    x = [0.5] * dim
    y = evaluator.evaluate(np.array(x))
    print("y: ", y)
    return x

if __name__ == "__main__":
    # Store results in global variables so that a helper function that runs this code can capture the results
    # x will be the numpy array of shape (dim,) that I will pass to the function to evaluate.
    # Do not delete these comments.
    global x
    x = solve(dim)
"""


def create_bbox_dataset():
    """Create dataset for circle packing problems.

    Returns:
        List of dspy.Example objects for unit square packing
    """
    examples = []

    for problem_name, problem in problems.items():
        baseline_code = BASELINE_CODE_TEMPLATE

        description = f"""This is a blackbox optimization problem. There is a function that we need to minimize.
        Given a numpy array of shape (dim,), the function returns a scalar value, which you need to minimize.
        The bounds for x are {problem.bounds}.
        The tags for this problem is {problem.classifiers} (Even if there are no tags, it's fine.).
        The dimension of the problem is {problem.dim}.
        """

        example = dspy.Example(
            {
                "problem_description": description,
                "baseline_code": baseline_code,
                "problem_name": problem_name,  # This should be hidden from the LLMs for fairness
            }
        ).with_inputs("problem_description", "baseline_code", "problem_name")

        examples.append(example)

    return examples


def main():
    args = parse_arguments()
    log_dir = get_log_directory(args)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    dataset = create_bbox_dataset()

    seed_candidate = {
        "code": BASELINE_CODE_TEMPLATE,
    }

    reflection_lm = create_reflection_lm(args.llm_model)
    print("Reflection LM: ", reflection_lm)

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            seed=0,
            max_metric_calls=args.max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            reflection_prompt_template=REFLECTION_PROMPT,
            reflection_lm=reflection_lm,
            skip_perfect_score=False,
        ),
        tracking=TrackingConfig(
            use_wandb=True,
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            wandb_init_kwargs={
                "name": f"polynomial_{len(dataset)}problems",
                "project": "gepa_polynomial",
            },
        ),
    )

    # Run GEPA optimization
    print("\n" + "=" * 70)
    print("Running GEPA Bounding Box Optimization")
    print("=" * 70 + "\n")

    fitness_fn = create_fitness_function(
        timeout=900,
    )

    optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        dataset=dataset,
        config=gepa_config,
    )

    # Save results
    print("\n" + "=" * 70)
    print("✓ Optimization Complete!")
    print("=" * 70)
    print(f"Log directory: {log_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
