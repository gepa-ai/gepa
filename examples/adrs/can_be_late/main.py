#!/usr/bin/env python3
"""
Can't Be Late optimization with GEPA optimize_anything API.

This example optimizes a cloud scheduling strategy that decides when to use
SPOT instances (cheap but preemptible) vs ON_DEMAND instances (expensive but reliable)
to complete tasks before deadlines while minimizing cost.
"""

import json
import logging
import os
import argparse
import sys
from netrc import NetrcParseError, netrc
from datetime import datetime
from pathlib import Path
from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)

try:
    # When running as part of the repo package (repo root on PYTHONPATH)
    from examples.adrs.can_be_late.evaluator import create_fitness_function
    from examples.adrs.can_be_late.trace_dataset import load_trace_dataset
except ModuleNotFoundError:
    # When running as a script: `python examples/adrs/can_be_late/main.py`
    # Add repo root to sys.path and fall back to local imports.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from evaluator import create_fitness_function  # type: ignore
    from trace_dataset import load_trace_dataset  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _has_wandb_netrc_credentials() -> bool:
    """
    Check whether the current user has W&B credentials stored via `wandb login`,
    which typically writes an entry into ~/.netrc.
    """
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        return False
    try:
        nrc = netrc(netrc_path)
    except (NetrcParseError, OSError):
        return False

    # `wandb login` commonly stores credentials for `api.wandb.ai`.
    for host in ("api.wandb.ai", "wandb.ai"):
        auth = nrc.authenticators(host)
        if auth is not None:
            _login, _account, password = auth
            if password:
                return True
    return False

# Initial baseline strategy
INITIAL_PROGRAM = """import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'
    
    def __init__(self, args):
        super().__init__(args)
    
    def reset(self, env, task):
        super().reset(env, task)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        
        # Task completion check
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE
        
        # Calculate remaining time until deadline
        remaining_time = self.deadline - env.elapsed_seconds
        
        # Simple deadline check: if we're running out of time, use ON_DEMAND
        # Add restart overhead to account for potential restart
        if remaining_task_time + self.restart_overhead >= remaining_time:
            # We need ON_DEMAND to guarantee completion
            return ClusterType.ON_DEMAND
        
        # Simple greedy logic: use SPOT if available, wait otherwise
        if has_spot:
            return ClusterType.SPOT
        else:
            # Just wait for SPOT to become available
            return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
"""

# Optimization objective for the Can't Be Late problem
OPTIMIZATION_OBJECTIVE = """Optimize a cloud scheduling strategy for the "Can't Be Late" problem.

The strategy decides when to use SPOT instances (cheap but can be preempted) vs ON_DEMAND 
instances (expensive but reliable) to complete a task before its deadline. The goal is to 
minimize cost while ensuring the task completes on time."""

# Domain background and constraints for the optimization
OPTIMIZATION_BACKGROUND = """Key information about the problem domain:

- ClusterType.SPOT: Use spot instances (cheap, ~$0.3/hour, but can be preempted at any time)
- ClusterType.ON_DEMAND: Use on-demand instances (expensive, ~$1/hour, but guaranteed availability)
- ClusterType.NONE: Wait without using any instances (no cost, but no progress)
- restart_overhead: Time penalty incurred when switching from one instance type to another
- The strategy MUST ensure task completion before the deadline (hard constraint)
- Lower cost is better (scores are negative, representing cost in dollars)

Evaluation feedback format:
- Timeline format: start-end:TYPE@REGION[progress%] (e.g., "0.0-5.0:S@R0[50%]" means SPOT from hour 0-5 reaching 50% progress)
- Spot availability: S=available, X=unavailable (e.g., "0.0-10.0:S | 10.0-15.0:X" means spot available first 10h, then unavailable)

Optimization targets:
1. Reduce overall cost while maintaining deadline guarantees
2. Make better decisions about when to use SPOT vs ON_DEMAND
3. Handle spot unavailability more intelligently
4. Consider the trade-offs between waiting for spot and using on-demand"""

# Dataset root for trace files
# NOTE: Update this path to point to your trace dataset
DATASET_ROOT = Path(__file__).resolve().parent / "simulator" / "real"


def _resolve_run_dir(run_dir_cli: Path | None = None) -> Path:
    """Resolve the run directory for saving artifacts."""
    if run_dir_cli is not None:
        run_dir = run_dir_cli
    else:
        run_dir_env = os.environ.get("GEPA_RUN_DIR")
        if run_dir_env is not None and run_dir_env != "":
            run_dir = Path(run_dir_env)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = Path("runs") / "cant_be_late" / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_results(
    run_dir: Path,
    gepa_result,
    best_candidate: dict[str, str],
    base_score: float | None = None,
    optimized_score: float | None = None,
):
    """Save optimization results to disk."""
    # Save the best program
    best_program_path = run_dir / "best_program.py"
    best_program_path.write_text(best_candidate["program"], encoding="utf-8")
    logger.info(f"Saved best program to {best_program_path}")

    # Save metrics summary
    metrics = {
        "base_score": base_score,
        "optimized_score": optimized_score,
        "best_candidate_index": gepa_result.best_idx,
        "num_candidates": len(gepa_result.candidates),
    }
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save all candidates
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for idx, candidate in enumerate(gepa_result.candidates):
        candidate_path = candidates_dir / f"candidate_{idx:03d}.py"
        candidate_path.write_text(candidate["program"], encoding="utf-8")

def get_reflection_lm(model: str) -> LanguageModel:
    """Create a reflection LM callable for the given model.
    
    Args:
        model: Model name (e.g., "gpt-4o-mini", "gemini-1.5-flash")
        
    Returns:
        A callable that takes a prompt (string or chat messages) and returns the model response.
    """
    import litellm  # type: ignore

    gemini_client = None
    if "gemini" in model:
        from google import genai
        from google.genai.types import HttpOptions
        gemini_client = genai.Client(http_options=HttpOptions(api_version="v1"))

    def _call_lm(prompt: str | list[dict[str, str]]) -> str:

        # Convert chat messages to a single string for Gemini
        if isinstance(prompt, list):
            prompt_str = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                for msg in prompt
            )
        else:
            prompt_str = prompt

        if "gemini" in model and gemini_client is not None:
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt_str,
            )
            return response.text or ""
        else:
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]
            completion = litellm.completion(model=model, messages=messages)
            return completion.choices[0].message.content or ""  # type: ignore

    return _call_lm

def main():
    """Run the Can't Be Late optimization."""
    parser = argparse.ArgumentParser(
        description="Run the Can't Be Late optimization example (GEPA optimize_anything)."
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Max traces per split to load. Defaults to no limit.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
        help="Max metric calls budget. Defaults to 100.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=3,
        help="Reflection minibatch size. Defaults to 3.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Reflection LLM model name.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Path to trace dataset root. Defaults to the example's data dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory to save artifacts/results. If unset, uses env GEPA_RUN_DIR; otherwise runs/cant_be_late/<timestamp>.",
    )

    args = parser.parse_args()

    max_traces = args.max_traces
    max_metric_calls = args.max_metric_calls
    reflection_minibatch_size = args.minibatch_size
    llm_model = get_reflection_lm(args.model)
    dataset_root = args.dataset_root

    # Resolve run directory
    run_dir = _resolve_run_dir(args.run_dir)
    logger.info(f"Run directory: {run_dir}")

    # Load dataset using the trace_dataset module
    try:
        splits = load_trace_dataset(
            dataset_root=str(dataset_root),
            max_traces_per_split=max_traces,
        )
        train_set = splits["train"]
        val_set = splits["val"]
        test_set = splits["test"]
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error(f"Please ensure traces are available at: {dataset_root}")
        logger.error("You can set CANT_BE_LATE_DATASET_ROOT to specify a different location.")
        return

    logger.info(
        f"Dataset sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}"
    )

    # Create seed candidate
    seed_candidate = {"program": INITIAL_PROGRAM}

    # Create fitness function
    fitness_fn = create_fitness_function(timeout=300)

    # Configure GEPA
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    use_wandb = (wandb_api_key is not None) or _has_wandb_netrc_credentials()

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            max_metric_calls=max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
            display_progress_bar=True,
            parallel=True,
            max_workers=128,  # Large number for high parallelism
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=reflection_minibatch_size,
            reflection_lm=llm_model,
        ),
        tracking=TrackingConfig(
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
            wandb_init_kwargs={
                "name": f"cant_be_late_{len(train_set)}samples",
                "project": "gepa_cant_be_late",
            },
        ),
        refiner=None,
    )

    # Run GEPA optimization
    logger.info("=" * 70)
    logger.info("Starting GEPA Optimization for Can't Be Late")
    logger.info("=" * 70)

    gepa_result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,  # type: ignore[arg-type]
        dataset=train_set,
        valset=val_set,
        objective=OPTIMIZATION_OBJECTIVE,
        background=OPTIMIZATION_BACKGROUND,
        config=gepa_config,
    )

    # Get best candidate
    best_candidate = gepa_result.best_candidate
    logger.info(f"Best candidate index: {gepa_result.best_idx}")

    # Evaluate on test set (optional)
    skip_test = os.environ.get("GEPA_SKIP_TEST", "0") == "1"

    if not skip_test and test_set:
        logger.info("Evaluating best candidate on test set...")
        test_scores = [fitness_fn(best_candidate, example)[0] for example in test_set]
        optimized_score = sum(test_scores) / len(test_scores) if test_scores else None
        logger.info(f"Test score (optimized): {optimized_score}")

        # Also evaluate baseline
        base_scores = [fitness_fn(seed_candidate, example)[0] for example in test_set]
        base_score = sum(base_scores) / len(base_scores) if base_scores else None
        logger.info(f"Test score (baseline): {base_score}")
    else:
        base_score = None
        optimized_score = None
        logger.info("Skipping test evaluation (GEPA_SKIP_TEST=1 or no test set)")

    # Save results
    _save_results(run_dir, gepa_result, best_candidate, base_score, optimized_score)

    logger.info("=" * 70)
    logger.info("✓ Optimization Complete!")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
