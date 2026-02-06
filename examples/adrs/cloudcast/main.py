#!/usr/bin/env python3
"""
Cloudcast broadcast optimization with GEPA optimize_anything API.

This example optimizes a broadcast routing algorithm that finds efficient paths
for transferring data from a single source to multiple destinations across
multi-cloud environments (AWS, GCP, Azure).
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
    from examples.adrs.cloudcast.evaluator import (
        create_fitness_function,
        load_config_dataset,
    )
except ModuleNotFoundError:
    # When running as a script: `python examples/adrs/cloudcast/main.py`
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from evaluator import create_fitness_function, load_config_dataset  # type: ignore

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

    for host in ("api.wandb.ai", "wandb.ai"):
        auth = nrc.authenticators(host)
        if auth is not None:
            _login, _account, password = auth
            if password:
                return True
    return False


# Initial baseline search algorithm
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

# Optimization objective for the Cloudcast problem
OPTIMIZATION_OBJECTIVE = """Optimize a broadcast routing algorithm for multi-cloud data transfer.

The algorithm decides how to route data from a single source to multiple destinations
across cloud providers (AWS, GCP, Azure). The goal is to minimize total cost 
(egress fees + instance costs) while maintaining good transfer times."""

# Domain background and constraints for the optimization
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

# Dataset root for config files
DATASET_ROOT = Path(__file__).resolve().parent / "cloudcast" / "config"


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
            run_dir = Path("runs") / "cloudcast" / timestamp

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
    """Run the Cloudcast broadcast optimization."""
    parser = argparse.ArgumentParser(
        description="Run the Cloudcast broadcast optimization example (GEPA optimize_anything)."
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
        "--config-dir",
        type=Path,
        default=DATASET_ROOT,
        help="Path to config directory. Defaults to the example's config dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory to save artifacts/results. If unset, uses env GEPA_RUN_DIR; otherwise runs/cloudcast/<timestamp>.",
    )

    args = parser.parse_args()

    max_metric_calls = args.max_metric_calls
    reflection_minibatch_size = args.minibatch_size
    llm_model = get_reflection_lm(args.model)
    config_dir = args.config_dir

    # Resolve run directory
    run_dir = _resolve_run_dir(args.run_dir)
    logger.info(f"Run directory: {run_dir}")

    # Load dataset
    try:
        samples = load_config_dataset(config_dir=str(config_dir))
        if not samples:
            logger.error(f"No configuration files found in: {config_dir}")
            return
            
        # Split samples into train/val/test
        # For cloudcast, we use all configs for training and validation
        n = len(samples)
        train_set = samples  # Use all for training
        val_set = samples    # Use all for validation too (small dataset)
        test_set = samples   # Use all for test
        
    except FileNotFoundError as e:
        logger.error(f"Config files not found: {e}")
        logger.error(f"Please ensure config files are available at: {config_dir}")
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
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=reflection_minibatch_size,
            reflection_lm=llm_model,
        ),
        tracking=TrackingConfig(
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
            wandb_init_kwargs={
                "name": f"cloudcast_{len(train_set)}configs",
                "project": "gepa_cloudcast",
            },
        ),
        refiner=None,
    )

    # Run GEPA optimization
    logger.info("=" * 70)
    logger.info("Starting GEPA Optimization for Cloudcast Broadcast")
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
