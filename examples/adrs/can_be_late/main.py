#!/usr/bin/env python3
"""
Can't Be Late: optimize a cloud scheduling strategy with GEPA.

The strategy decides at each step whether to use a cheap-but-preemptible SPOT
instance, a reliable ON_DEMAND instance, or wait — aiming to complete a task
before its deadline at minimum cost.
"""

import json
import logging
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Ensure the repository root is importable regardless of how this script is run
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, TrackingConfig, SideInfo, optimize_anything

from utils.dataset import load_trace_dataset
from utils.lm import make_reflection_lm
from utils.wandb_auth import has_wandb_credentials
from utils.simulation import (
    FAILED_SCORE,
    get_program_path,
    syntax_is_valid,
    syntax_failure_info,
    run_simulation,
    simulation_failure_info,
    simulation_success_info,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# What GEPA will evolve
# ---------------------------------------------------------------------------

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

        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        remaining_time = self.deadline - env.elapsed_seconds

        if remaining_task_time + self.restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
"""

OPTIMIZATION_OBJECTIVE = """Optimize a cloud scheduling strategy for the "Can't Be Late" problem.

The strategy decides when to use SPOT instances (cheap but can be preempted) vs ON_DEMAND
instances (expensive but reliable) to complete a task before its deadline. The goal is to
minimize cost while ensuring the task completes on time."""

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

DATASET_ROOT = Path(__file__).resolve().parent / "utils" / "simulator" / "real"


# ---------------------------------------------------------------------------
# Evaluator — called by GEPA for every (candidate, example) pair
# ---------------------------------------------------------------------------

def evaluate(candidate: dict, example: dict, **kwargs) -> tuple[float, SideInfo]:
    program_path = get_program_path(candidate["program"])

    if not syntax_is_valid(program_path):
        return FAILED_SCORE, syntax_failure_info(example)

    success, cost, error, details = run_simulation(program_path, example["trace_file"], example["config"])

    if not success:
        return FAILED_SCORE, simulation_failure_info(error, example)

    score = -cost
    return score, simulation_success_info(score, example, details)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    logger.info(f"Run directory: {run_dir}")

    dataset = load_trace_dataset(
        dataset_root=str(args.dataset_root),
        max_traces_per_split=args.max_traces,
    )
    logger.info(f"Dataset — train: {len(dataset['train'])}, val: {len(dataset['val'])}, test: {len(dataset['test'])}")

    reflection_lm = make_reflection_lm(args.model)

    wandb_api_key = os.environ.get("WANDB_API_KEY")

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            max_metric_calls=args.max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
            display_progress_bar=True,
            parallel=True,
            max_workers=128,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=args.minibatch_size,
            reflection_lm=reflection_lm,
        ),
        tracking=TrackingConfig(
            use_wandb=has_wandb_credentials(),
            wandb_api_key=wandb_api_key,
            wandb_init_kwargs={
                "name": f"cant_be_late_{len(dataset['train'])}samples",
                "project": "gepa_cant_be_late",
            },
        ),
        refiner=None,
    )

    logger.info("Starting GEPA optimization for Can't Be Late")
    result = optimize_anything(
        seed_candidate={"program": INITIAL_PROGRAM},
        evaluator=evaluate,
        dataset=dataset["train"],
        valset=dataset["val"],
        objective=OPTIMIZATION_OBJECTIVE,
        background=OPTIMIZATION_BACKGROUND,
        config=gepa_config,
    )

    _save_results(run_dir, result)
    _log_test_scores(result, {"program": INITIAL_PROGRAM}, dataset["test"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Can't Be Late optimization with GEPA.")
    parser.add_argument("--model", type=str, required=True, help="Reflection LLM model name.")
    parser.add_argument("--max-traces", type=int, default=None, help="Max samples per split (for quick tests).")
    parser.add_argument("--max-metric-calls", type=int, default=100, help="Max fitness evaluations.")
    parser.add_argument("--minibatch-size", type=int, default=3, help="Reflection minibatch size.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT, help="Path to trace dataset root.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory for output artifacts.")
    return parser.parse_args()


def _resolve_run_dir(run_dir_arg: Path | None) -> Path:
    if run_dir_arg is not None:
        run_dir = run_dir_arg
    elif os.environ.get("GEPA_RUN_DIR"):
        run_dir = Path(os.environ["GEPA_RUN_DIR"])
    else:
        run_dir = Path("runs") / "cant_be_late" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_results(run_dir: Path, result: object) -> None:
    best = result.best_candidate  # type: ignore[attr-defined]
    (run_dir / "best_program.py").write_text(best["program"], encoding="utf-8")

    metrics = {
        "best_candidate_index": result.best_idx,  # type: ignore[attr-defined]
        "num_candidates": len(result.candidates),  # type: ignore[attr-defined]
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for idx, candidate in enumerate(result.candidates):  # type: ignore[attr-defined]
        (candidates_dir / f"candidate_{idx:03d}.py").write_text(candidate["program"], encoding="utf-8")

    logger.info(f"Results saved to: {run_dir}")


def _log_test_scores(result: object, seed_candidate: dict, test_set: list) -> None:
    if os.environ.get("GEPA_SKIP_TEST") == "1" or not test_set:
        return

    best = result.best_candidate  # type: ignore[attr-defined]

    optimized_scores = [evaluate(best, ex)[0] for ex in test_set]
    baseline_scores = [evaluate(seed_candidate, ex)[0] for ex in test_set]

    logger.info(f"Test score (optimized): {sum(optimized_scores) / len(optimized_scores):.4f}")
    logger.info(f"Test score (baseline):  {sum(baseline_scores) / len(baseline_scores):.4f}")


if __name__ == "__main__":
    main()
