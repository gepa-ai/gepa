import argparse
import os

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)

from llm import REFLECTION_PROMPT_TEMPLATE
from evaluator import create_fitness_function


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Circle packing optimization with GEPA"
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
        help="Maximum number of metric calls (evaluations)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def get_problem_sizes():
    """Get fixed list of problem sizes for batch mode."""
    return [7, 10, 21, 22, 26, 28, 29, 31, 32, 33, 52, 68, 99]


def get_log_directory():
    log_dir = "results/circle_packing"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


SEED_CANDIDATE = {
    "code": """
import numpy as np
def circle_packing(num_circles):
    best_circles = np.array([[0.5, 0.5, 0.5]]  * num_circles)
    best_score = evaluate_circles(best_circles, num_circles)
    return best_circles, best_score

##### IMMUTABLE CODE BLOCK START #####
# Always include the code below without changing or deleting it.
if __name__ == "__main__":
    global best_circles, best_score
    best_circles, best_score = circle_packing(num_circles)
##### IMMUTABLE CODE BLOCK END #####
""",
}


if __name__ == "__main__":
    args = parse_arguments()
    log_dir = get_log_directory()

    # Set random seed
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    max_metric_calls = args.max_metric_calls

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
            skip_perfect_score=False,
        ),
        tracking=TrackingConfig(
            use_wandb=True, wandb_api_key=os.environ.get("WANDB_API_KEY")
        ),
    )

    # Create dataset with all problem sizes (batch mode)
    problem_sizes = get_problem_sizes()
    dataset = [{"num_circles": n} for n in problem_sizes]

    fitness_fn = create_fitness_function()

    result = optimize_anything(
        seed_candidate=SEED_CANDIDATE,
        fitness_fn=fitness_fn,
        dataset=dataset,
        config=gepa_config,
    )

    print("\n" + "=" * 80)
    print("Evaluating Best Candidate on Test Set")
    print("=" * 80)

    # Evaluate best candidate on test set
    best_candidate = result.best_candidate
    print("\n🔍 Testing best candidate...")

    # Run evaluation on test set (multiple runs for robustness)
    test_results = fitness_fn(best_candidate, dataset)

    # Extract scores
    test_scores = [score for score, _, _ in test_results]
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0

    print("\n✅ Test Set Evaluation Complete:")
    print(f"   Test Scores: {test_scores}")
    print(f"   Average Test Score: {avg_test_score:.6f}")
    print(f"\n📁 Results saved to: {log_dir}")
