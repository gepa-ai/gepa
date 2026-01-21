#!/usr/bin/env python3
"""
Evaluate base and best candidates on test traces for a completed GEPA run.

Usage:
    python eval_test.py --run-dir runs/cant_be_late/20260115-125824
    python eval_test.py --run-dir runs/cant_be_late/20260115-125824 --max-traces 10
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    # When running as part of the repo package (repo root on PYTHONPATH)
    from examples.adrs.can_be_late.evaluator import create_fitness_function
    from examples.adrs.can_be_late.trace_dataset import load_trace_dataset
except ModuleNotFoundError:
    # When running as a script: add src to path and use local imports
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evaluator import create_fitness_function  # type: ignore
    from trace_dataset import load_trace_dataset  # type: ignore

# Default dataset root
DATASET_ROOT = SCRIPT_DIR / "simulator" / "real"


def load_gepa_state(run_dir: Path) -> dict:
    """Load the GEPA state from a run directory."""
    state_path = run_dir / "gepa_state.bin"
    if not state_path.exists():
        raise FileNotFoundError(f"No gepa_state.bin found in {run_dir}")
    
    with open(state_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_best_candidate_idx(state: dict) -> int:
    """Get the index of the best candidate based on aggregate validation scores."""
    prog_scores = state.get("prog_candidate_val_subscores", [])
    
    # Calculate aggregate scores for each program
    aggregate_scores = []
    for scores_dict in prog_scores:
        if scores_dict:
            avg = sum(scores_dict.values()) / len(scores_dict)
        else:
            avg = float("-inf")
        aggregate_scores.append(avg)
    
    if not aggregate_scores:
        raise ValueError("No candidates found in state")
    
    return aggregate_scores.index(max(aggregate_scores))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base and best candidates on test traces."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the GEPA run directory containing gepa_state.bin",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Path to trace dataset root. Defaults to example's simulator/real dir.",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of test traces to evaluate (for quick testing).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each simulation. Default: 300",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results. Defaults to <run-dir>/test_evaluation.json",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-sample results.",
    )
    
    args = parser.parse_args()
    
    run_dir = args.run_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    
    if not run_dir.exists():
        logger.error(f"Run directory does not exist: {run_dir}")
        sys.exit(1)
    
    # Load GEPA state
    logger.info(f"Loading GEPA state from: {run_dir}")
    try:
        state = load_gepa_state(run_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Extract candidates
    candidates = state.get("program_candidates", [])
    num_candidates = len(candidates)
    logger.info(f"Found {num_candidates} candidates in state")
    
    if num_candidates == 0:
        logger.error("No candidates found in state")
        sys.exit(1)
    
    # Get base (seed) and best candidates
    seed_candidate = candidates[0]  # First candidate is always the seed
    best_idx = get_best_candidate_idx(state)
    best_candidate = candidates[best_idx]
    
    logger.info(f"Seed candidate: index 0")
    logger.info(f"Best candidate: index {best_idx}")
    
    # Load test dataset
    logger.info(f"Loading test dataset from: {dataset_root}")
    try:
        splits = load_trace_dataset(
            dataset_root=str(dataset_root),
            max_traces_per_split=args.max_traces,
        )
        test_set = splits["test"]
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)
    
    if not test_set:
        logger.error("Test set is empty")
        sys.exit(1)
    
    logger.info(f"Test set size: {len(test_set)} samples")
    
    # Create fitness function
    fitness_fn = create_fitness_function(timeout=args.timeout)
    
    FAILED_SCORE = -100000.0  # From evaluator.py
    
    # Evaluate best candidate on test set
    logger.info("=" * 70)
    logger.info("Evaluating BEST candidate on test set...")
    logger.info("=" * 70)
    
    best_results = fitness_fn(best_candidate, test_set)
    best_scores = [r[0] for r in best_results]
    best_valid_scores = [s for s in best_scores if s > FAILED_SCORE]
    best_failed = len(best_scores) - len(best_valid_scores)
    best_score = sum(best_valid_scores) / len(best_valid_scores) if best_valid_scores else None
    
    if args.verbose:
        for i, (score, output, side_info) in enumerate(best_results):
            status = "FAIL" if score <= FAILED_SCORE else "OK"
            logger.info(f"  [{status}] Sample {i}: score={score:.4f}")
    
    logger.info(f"Best candidate: {len(best_valid_scores)} succeeded, {best_failed} failed")
    logger.info(f"Best candidate test score: {best_score:.4f}" if best_score else "Best candidate test score: N/A")
    
    # Evaluate seed (base) candidate on test set  
    logger.info("=" * 70)
    logger.info("Evaluating SEED (base) candidate on test set...")
    logger.info("=" * 70)
    
    seed_results = fitness_fn(seed_candidate, test_set)
    seed_scores = [r[0] for r in seed_results]
    seed_valid_scores = [s for s in seed_scores if s > FAILED_SCORE]
    seed_failed = len(seed_scores) - len(seed_valid_scores)
    seed_score = sum(seed_valid_scores) / len(seed_valid_scores) if seed_valid_scores else None
    
    if args.verbose:
        for i, (score, output, side_info) in enumerate(seed_results):
            status = "FAIL" if score <= FAILED_SCORE else "OK"
            logger.info(f"  [{status}] Sample {i}: score={score:.4f}")
    
    logger.info(f"Seed candidate: {len(seed_valid_scores)} succeeded, {seed_failed} failed")
    logger.info(f"Seed candidate test score: {seed_score:.4f}" if seed_score else "Seed candidate test score: N/A")
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Run directory:        {run_dir}")
    logger.info(f"  Test samples:         {len(test_set)}")
    logger.info(f"  Seed (base) score:    {seed_score:.4f}" if seed_score else "  Seed (base) score:    N/A")
    logger.info(f"  Best candidate idx:   {best_idx}")
    logger.info(f"  Best candidate score: {best_score:.4f}" if best_score else "  Best candidate score: N/A")
    
    if seed_score is not None and best_score is not None:
        improvement = best_score - seed_score
        logger.info(f"  Improvement:          {improvement:+.4f}")
        if seed_score != 0:
            pct_improvement = (improvement / abs(seed_score)) * 100
            logger.info(f"  Relative improvement: {pct_improvement:+.2f}%")
    
    logger.info("=" * 70)
    
    # Save results
    output_path = args.output or (run_dir / "test_evaluation.json")
    results = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "num_candidates": num_candidates,
        "best_candidate_idx": best_idx,
        "test_set_size": len(test_set),
        "seed_test_score": seed_score,
        "seed_succeeded": len(seed_valid_scores),
        "seed_failed": seed_failed,
        "best_test_score": best_score,
        "best_succeeded": len(best_valid_scores),
        "best_failed": best_failed,
        "improvement": best_score - seed_score if (seed_score and best_score) else None,
        "seed_scores": seed_scores,
        "best_scores": best_scores,
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
