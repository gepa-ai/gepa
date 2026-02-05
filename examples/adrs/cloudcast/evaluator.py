"""
Evaluator helpers for the Cloudcast broadcast optimization problem.

This module provides:
- Syntax validation (stage1)
- Broadcast simulation execution
- Fitness function factory for GEPA optimization
"""

import importlib.util
import json
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from gepa.optimize_anything import SideInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Score for failed programs
FAILED_SCORE = -100000.0

# Path to the cloudcast module (relative to this file)
CURRENT_DIR = Path(__file__).resolve().parent
CLOUDCAST_DIR = CURRENT_DIR / "cloudcast"

# Default configuration files
CONFIG_FILES = [
    "intra_aws.json",
    "intra_azure.json",
    "intra_gcp.json",
    "inter_agz.json",
    "inter_gaz2.json",
]


def evaluate_stage1(program_path: str) -> dict:
    """
    Stage 1: Quick syntax and import check.

    Args:
        program_path: Path to the search algorithm Python file

    Returns:
        Dict with 'runs_successfully' (0.0 or 1.0) and optional 'error', 'score'
    """
    try:
        with open(program_path, "r") as f:
            code = f.read()

        # Try to compile the code
        compile(code, program_path, "exec")

        # Basic validation - check for required function
        if "search_algorithm" not in code:
            return {
                "runs_successfully": 0.0,
                "error": "No search_algorithm function found",
                "score": FAILED_SCORE,
            }

        if "def search_algorithm" not in code:
            return {
                "runs_successfully": 0.0,
                "error": "search_algorithm must be a function definition",
                "score": FAILED_SCORE,
            }

        return {"runs_successfully": 1.0}

    except SyntaxError as e:
        return {
            "runs_successfully": 0.0,
            "error": f"Syntax error: {e}",
            "score": FAILED_SCORE,
        }
    except Exception as e:
        return {
            "runs_successfully": 0.0,
            "error": str(e),
            "score": FAILED_SCORE,
        }


def run_single_config(
    program_path: str,
    config_file: str,
    num_vms: int = 2,
) -> tuple[bool, float, float, str, dict[str, Any]]:
    """
    Run evaluation for a single configuration.

    Args:
        program_path: Path to the evolved program file
        config_file: Path to the configuration JSON file
        num_vms: Number of VMs per region

    Returns:
        Tuple of (success, cost, transfer_time, error_msg, detailed_info)
    """
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        if spec is None or spec.loader is None:
            error_msg = f"Failed to create module spec for {program_path}"
            return (False, 0.0, 0.0, error_msg, {"error": error_msg})
        
        program = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(program)
        except Exception as load_error:
            error_msg = f"Failed to load program: {str(load_error)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return (False, 0.0, 0.0, error_msg, {"error": str(load_error), "traceback": traceback.format_exc()})

        # Check if the required function exists
        if not hasattr(program, "search_algorithm"):
            return (
                False, 0.0, 0.0,
                "Missing search_algorithm function",
                {"error": "Missing search_algorithm function"}
            )

        # Import cloudcast modules
        from examples.adrs.cloudcast.cloudcast.broadcast import BroadCastTopology
        from examples.adrs.cloudcast.cloudcast.simulator import BCSimulator
        from examples.adrs.cloudcast.cloudcast.utils import make_nx_graph

        # Load configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        config_name = os.path.basename(config_file).split(".")[0]

        # Create graph
        G = make_nx_graph(num_vms=int(num_vms))

        # Source and destination nodes
        source_node = config["source_node"]
        terminal_nodes = config["dest_nodes"]

        # Run the evolved algorithm
        num_partitions = config["num_partitions"]
        try:
            bc_t = program.search_algorithm(source_node, terminal_nodes, G, num_partitions)
        except Exception as algo_error:
            error_msg = f"search_algorithm execution failed: {str(algo_error)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return (False, 0.0, 0.0, error_msg, {"error": str(algo_error), "traceback": traceback.format_exc()})
        
        if bc_t is None:
            error_msg = "search_algorithm returned None"
            return (False, 0.0, 0.0, error_msg, {"error": error_msg})
            
        bc_t.set_num_partitions(config["num_partitions"])

        # VALIDATION: Check that ALL destinations have at least one valid path
        missing_destinations = []
        for dest in terminal_nodes:
            if dest not in bc_t.paths:
                missing_destinations.append(dest)
                continue

            # Check if this destination has at least one non-empty path across all partitions
            has_valid_path = False
            dest_partitions = bc_t.paths.get(dest, {})
            for partition_id, path_list in dest_partitions.items():
                if path_list and len(path_list) > 0:
                    has_valid_path = True
                    break

            if not has_valid_path:
                missing_destinations.append(dest)

        if missing_destinations:
            error_msg = f"No paths for destinations: {', '.join(missing_destinations)}"
            return (False, 0.0, 0.0, error_msg, {"missing_destinations": missing_destinations})

        # Create output directory for simulation
        output_dir = tempfile.mkdtemp(prefix="cloudcast_eval_")

        # Evaluate the generated paths
        simulator = BCSimulator(int(num_vms), output_dir)
        transfer_time, cost = simulator.evaluate_path(bc_t, config)

        # Clean up
        shutil.rmtree(output_dir, ignore_errors=True)

        detailed_info = {
            "config_name": config_name,
            "cost": cost,
            "transfer_time": transfer_time,
            "source": source_node,
            "destinations": terminal_nodes,
            "num_partitions": num_partitions,
        }

        return (True, cost, transfer_time, "", detailed_info)

    except Exception as e:
        error_msg = f"Error evaluating {os.path.basename(config_file)}: {str(e)}"
        tb = traceback.format_exc()
        logger.error(error_msg)
        logger.error(tb)
        return (False, 0.0, 0.0, error_msg, {"error": str(e), "traceback": tb})


def create_fitness_function(timeout: int = 300):
    """
    Create fitness function for GEPA optimization.

    The fitness function evaluates a candidate search algorithm on a batch of
    configuration samples, running simulations and returning scores with
    diagnostic information.

    Args:
        timeout: Timeout in seconds for each simulation (not currently enforced)

    Returns:
        Fitness function compatible with GEPA's optimize_anything API
    """
    # Cache for program file and syntax check to avoid redundant work
    _cache: dict[str, Any] = {
        "program_code": None,
        "program_path": None,
        "tmpdir": None,
        "stage1_result": None,
    }

    def _get_or_create_program_file(program_code: str) -> tuple[str, dict]:
        """Get cached program file or create new one if program changed."""
        if _cache["program_code"] != program_code:
            # Clean up old temp directory if exists
            if _cache["tmpdir"] is not None:
                shutil.rmtree(_cache["tmpdir"], ignore_errors=True)
            
            # Create new temp directory and write program
            tmpdir = tempfile.mkdtemp(prefix="cloudcast_eval_")
            program_path = os.path.join(tmpdir, "program.py")
            with open(program_path, "w", encoding="utf-8") as f:
                f.write(program_code)
            
            # Run syntax check once
            stage1_result = evaluate_stage1(program_path)
            
            # Update cache
            _cache["program_code"] = program_code
            _cache["program_path"] = program_path
            _cache["tmpdir"] = tmpdir
            _cache["stage1_result"] = stage1_result
        
        return _cache["program_path"], _cache["stage1_result"]

    def fitness_fn(
        candidate: dict[str, str], example: dict[str, Any], **kwargs
    ) -> tuple[float, SideInfo]:
        """
        Evaluate a candidate search algorithm on a single configuration.

        Args:
            candidate: Dict with "program" key containing search algorithm code
            example: Sample dict with 'config_file' and optional 'num_vms'

        Returns:
            Tuple of (score, side_info)
        """
        program_code = candidate["program"]

        # Get cached program file or create new one
        program_path, stage1_result = _get_or_create_program_file(program_code)

        # Stage 1: Check cached syntax result
        if stage1_result.get("runs_successfully", 0.0) < 1.0:
            error_msg = stage1_result.get("error", "Syntax validation failed")
            side_info: SideInfo = {
                "scores": {"cost": FAILED_SCORE},
                "Input": {
                    "config_file": example.get("config_file", "unknown"),
                },
                "Error": error_msg,
                "stage": "stage1",
            }
            # output = {"error": error_msg, "stage": "stage1"}
            return (FAILED_SCORE, side_info)

        # Stage 2: Run simulation
        config_file = example.get("config_file")
        num_vms = example.get("num_vms", 2)

        if not config_file:
            side_info = {
                "scores": {"cost": FAILED_SCORE},
                "Input": {"config_file": config_file},
                "Error": "Invalid sample: missing config_file",
            }
            # output = {"error": "Invalid sample"}
            return (FAILED_SCORE, side_info)

        # Run simulation
        success, cost, transfer_time, error_msg, detailed_info = run_single_config(
            program_path, config_file, num_vms
        )

        if success:
            # Score is based on cost (lower cost = higher score)
            # Use normalized score: 1 / (1 + cost)
            score = 1.0 / (1.0 + cost)

            config_name = detailed_info.get("config_name", os.path.basename(config_file))
            
            side_info = {
                "scores": {"cost_score": score, "raw_cost": cost},
                "Input": {
                    "config": config_name,
                    "source": detailed_info.get("source", "N/A"),
                    "num_destinations": len(detailed_info.get("destinations", [])),
                    "num_partitions": detailed_info.get("num_partitions", "N/A"),
                },
                "Output": {
                    "cost": f"${cost:.4f}",
                    "transfer_time": f"{transfer_time:.2f}s",
                },
            }
            # output = {
            #     "config_file": config_file,
            #     "cost": cost,
            #     "transfer_time": transfer_time,
            #     "score": score,
            #     "detailed_info": detailed_info,
            # }
            return (score, side_info)
        else:
            score = FAILED_SCORE
            side_info = {
                "scores": {"cost": score},
                "Input": {
                    "config_file": os.path.basename(config_file) if config_file else "unknown",
                },
                "Error": error_msg,
            }
            # output = {
            #     "config_file": config_file,
            #     "error": error_msg,
            # }
            return (score, side_info)

    return fitness_fn


def load_config_dataset(
    config_dir: str | None = None,
    config_files: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Load configuration files as dataset samples.

    Args:
        config_dir: Directory containing config JSON files
        config_files: List of specific config file names to use

    Returns:
        List of sample dicts with 'config_file' key
    """
    if config_dir is None:
        config_dir = str(CLOUDCAST_DIR / "config")

    if config_files is None:
        config_files = CONFIG_FILES

    samples = []
    for config_name in config_files:
        config_path = os.path.join(config_dir, config_name)
        if os.path.exists(config_path):
            samples.append({
                "config_file": config_path,
                "num_vms": 2,
            })
        else:
            logger.warning(f"Config file not found: {config_path}")

    return samples


if __name__ == "__main__":
    # Quick test of the evaluator
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "program_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the program to evaluate",
    )
    args = parser.parse_args()

    if args.program_path:
        print(f"Testing evaluator with program: {args.program_path}")
        print("\nStage 1 (syntax check):")
        result1 = evaluate_stage1(args.program_path)
        print(json.dumps(result1, indent=2))
    else:
        print("Loading config dataset...")
        samples = load_config_dataset()
        print(f"Found {len(samples)} configuration samples")
        for sample in samples:
            print(f"  - {os.path.basename(sample['config_file'])}")
