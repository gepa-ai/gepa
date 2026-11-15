#!/usr/bin/env python3
"""
Cloudcast: optimize a broadcast routing algorithm with GEPA.

The algorithm finds paths from a single source to multiple cloud destinations
(AWS, GCP, Azure), aiming to minimize egress and instance costs.
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

from utils.dataset import load_config_dataset
from utils.lm import make_reflection_lm
from utils.wandb_auth import has_wandb_credentials
from utils.simulation import (
    FAILED_SCORE,
    get_program_path,
    syntax_is_valid,
    syntax_failure_info,
    run_evaluation,
    evaluation_failure_info,
    evaluation_success_info,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# What GEPA will evolve
# ---------------------------------------------------------------------------

INITIAL_PROGRAM = """import networkx as nx
import pandas as pd
import os
from typing import Dict, List


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, 'SingleDstPath'] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def search_algorithm(src, dsts, G, num_partitions):
    \"\"\"
    Find broadcast paths from source to all destinations.

    Uses Dijkstra's shortest path algorithm based on cost as the edge weight.

    Args:
        src: Source node identifier (e.g., "aws:ap-northeast-1")
        dsts: List of destination node identifiers
        G: NetworkX DiGraph with cost and throughput edge attributes
        num_partitions: Number of data partitions

    Returns:
        BroadCastTopology object with paths for all destinations and partitions
    \"\"\"
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology
"""

OPTIMIZATION_OBJECTIVE = """Optimize a broadcast routing algorithm for multi-cloud data transfer.

The algorithm decides how to route data from a single source to multiple destinations
across cloud providers (AWS, GCP, Azure). The goal is to minimize total cost
(egress fees + instance costs) while maintaining good transfer times."""

OPTIMIZATION_BACKGROUND = """Key information about the problem domain:

- The network is represented as a directed graph where:
  - Nodes are cloud regions (e.g., "aws:us-east-1", "gcp:europe-west1-a", "azure:eastus")
  - Edges have 'cost' ($/GB for egress) and 'throughput' (Gbps bandwidth) attributes

- Data is partitioned into num_partitions chunks that can be routed independently
- Each partition can take a different path to reach each destination
- Total cost = egress costs (data_vol × edge_cost) + instance costs (runtime × cost_per_hour)

- The algorithm must return a BroadCastTopology object containing:
  - paths[dst][partition] = list of edges [[src, dst, edge_data], ...]
  - Each destination must have at least one valid path for each partition

Evaluation feedback format:
- Cost: Total transfer cost in dollars
- Transfer time: Maximum time for all destinations to receive data (seconds)

Optimization targets:
1. Reduce total cost (egress + instance costs)
2. Find paths that balance cost and throughput
3. Consider multipath routing for better bandwidth utilization
4. Exploit cloud provider pricing differences (e.g., intra-provider is cheaper)"""

DATASET_ROOT = Path(__file__).resolve().parent / "utils" / "cloudcast" / "config"


# ---------------------------------------------------------------------------
# Evaluator — called by GEPA for every (candidate, example) pair
# ---------------------------------------------------------------------------

def evaluate(candidate: dict, example: dict, **kwargs) -> tuple[float, SideInfo]:
    program_path = get_program_path(candidate["program"])

    if not syntax_is_valid(program_path):
        return FAILED_SCORE, syntax_failure_info(example)

    success, cost, transfer_time, error, details = run_evaluation(
        program_path, example["config_file"], example["num_vms"]
    )

    if not success:
        return FAILED_SCORE, evaluation_failure_info(error, example)

    score = 1.0 / (1.0 + cost)
    return score, evaluation_success_info(score, cost, transfer_time, example, details)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    logger.info(f"Run directory: {run_dir}")

    dataset = load_config_dataset(config_dir=args.config_dir)
    if not dataset:
        logger.error(f"No configuration files found in: {args.config_dir}")
        return
    logger.info(f"Loaded {len(dataset)} configuration samples (used for train, val, and test)")

    reflection_lm = make_reflection_lm(args.model)

    wandb_api_key = os.environ.get("WANDB_API_KEY")

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            seed=0,
            max_metric_calls=args.max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
            display_progress_bar=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=args.minibatch_size,
            reflection_lm=reflection_lm,
            skip_perfect_score=False,
        ),
        tracking=TrackingConfig(
            use_wandb=has_wandb_credentials(),
            wandb_api_key=wandb_api_key,
            wandb_init_kwargs={
                "name": f"cloudcast_{len(dataset)}configs",
                "project": "gepa_cloudcast",
            },
        ),
    )

    logger.info("Starting GEPA optimization for Cloudcast")
    result = optimize_anything(
        seed_candidate={"program": INITIAL_PROGRAM},
        evaluator=evaluate,
        dataset=dataset,
        valset=dataset,
        objective=OPTIMIZATION_OBJECTIVE,
        background=OPTIMIZATION_BACKGROUND,
        config=gepa_config,
    )

    _save_results(run_dir, result)
    _log_test_scores(result, {"program": INITIAL_PROGRAM}, dataset)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloudcast broadcast optimization with GEPA.")
    parser.add_argument("--model", type=str, required=True, help="Reflection LLM model name.")
    parser.add_argument("--max-metric-calls", type=int, default=100, help="Max fitness evaluations.")
    parser.add_argument("--minibatch-size", type=int, default=3, help="Reflection minibatch size.")
    parser.add_argument("--config-dir", type=Path, default=DATASET_ROOT, help="Path to config files directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory for output artifacts.")
    return parser.parse_args()


def _resolve_run_dir(run_dir_arg: Path | None) -> Path:
    if run_dir_arg is not None:
        run_dir = run_dir_arg
    elif os.environ.get("GEPA_RUN_DIR"):
        run_dir = Path(os.environ["GEPA_RUN_DIR"])
    else:
        run_dir = Path("runs") / "cloudcast" / datetime.now().strftime("%Y%m%d-%H%M%S")
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
