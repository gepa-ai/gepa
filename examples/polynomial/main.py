#!/usr/bin/env python3
"""
Blackbox optimization with GEPA

Key features:
- Uses objective/background pattern instead of custom reflection prompt
- Uses cache_evaluation=True for automatic caching
- Warm-start support via prev_best_x
- Single score metric
"""

from examples.polynomial.config import parse_arguments, get_log_directory
from examples.polynomial.prompt import OBJECTIVE, BACKGROUND
from examples.polynomial.evaluator import FitnessEvaluator

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)


BASELINE_CODE_TEMPLATE = """
import numpy as np

def solve(objective_function, config, prev_best_x=None):
    bounds = np.array(config['bounds'])
    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    y = objective_function(x)
    return x
"""


def main():
    args = parse_arguments()
    log_dir = get_log_directory(args)

    seed_candidate = {
        "code": BASELINE_CODE_TEMPLATE,
    }

    # Configure GEPA with cache_evaluation=True and objective/background pattern
    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            seed=args.seed,
            max_metric_calls=args.max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm=args.llm_model,
        ),
    )

    # Create evaluator
    evaluator = FitnessEvaluator(
        problem_index=args.problem_index,
        timeout=args.timeout,
        evaluation_budget=args.evaluation_budget,
        log_dir=log_dir,
        seed=args.seed,
    )

    # Run GEPA optimization with objective and background
    print("\n" + "=" * 70)
    print(f"Running GEPA Blackbox Optimization for Polynomial Problem {args.problem_index}")
    print("=" * 70 + "\n")

    optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=evaluator.evaluate,
        config=gepa_config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
