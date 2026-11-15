"""Run Cloudcast broadcast simulations and build GEPA side-info dicts.

Exposes four functions used by ``evaluate`` in main.py:
  - get_program_path(code)        – write candidate to temp file (cached)
  - syntax_is_valid(program_path) – True when the program compiles and has search_algorithm
  - run_evaluation(...)           – load the program and run the broadcast simulator
  - build_side_info(...)          – format results for GEPA's reflective feedback
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

logger = logging.getLogger(__name__)

FAILED_SCORE = -100_000.0

# ---------------------------------------------------------------------------
# Program file management (cached per process)
# ---------------------------------------------------------------------------

_program_cache: dict[str, Any] = {
    "code": None,
    "path": None,
    "tmpdir": None,
    "syntax_ok": None,
    "syntax_error": None,
}

# Graph cache: building the NetworkX graph from CSV is slow; cache by num_vms
_graph_cache: dict[int, Any] = {}


def get_program_path(program_code: str) -> str:
    """Write the candidate program to a temp file if it has changed; return its path."""
    if _program_cache["code"] != program_code:
        _evict_cached_program()
        _write_program_to_cache(program_code)
    return _program_cache["path"]


def syntax_is_valid(program_path: str) -> bool:
    """Return True if the cached program passed the syntax and structure check."""
    return _program_cache["syntax_ok"] is True


def syntax_failure_info(example: dict[str, Any]) -> SideInfo:
    """Build a SideInfo dict describing a syntax or structure validation failure."""
    return {
        "scores": {"cost": FAILED_SCORE},
        "Input": {"config_file": os.path.basename(example.get("config_file", "?"))},
        "Error": _program_cache["syntax_error"] or "Syntax validation failed",
    }


def _evict_cached_program() -> None:
    if _program_cache["tmpdir"]:
        shutil.rmtree(_program_cache["tmpdir"], ignore_errors=True)
    _program_cache.update(code=None, path=None, tmpdir=None, syntax_ok=None, syntax_error=None)


def _write_program_to_cache(code: str) -> None:
    tmpdir = tempfile.mkdtemp(prefix="cloudcast_eval_")
    path = os.path.join(tmpdir, "program.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    ok, error = _check_syntax(path)
    _program_cache.update(code=code, path=path, tmpdir=tmpdir, syntax_ok=ok, syntax_error=error)


def _check_syntax(program_path: str) -> tuple[bool, str | None]:
    """Return (ok, error_message) after validating syntax and required structure."""
    try:
        with open(program_path) as f:
            code = f.read()
        compile(code, program_path, "exec")
        if "def search_algorithm" not in code:
            return False, "search_algorithm function definition not found"
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Broadcast simulation
# ---------------------------------------------------------------------------


def run_evaluation(
    program_path: str,
    config_file: str,
    num_vms: int = 2,
) -> tuple[bool, float, float, str, dict[str, Any]]:
    """Load the candidate program and evaluate it on one broadcast configuration.

    Args:
        program_path: Path to the evolved search_algorithm file.
        config_file: Path to the broadcast scenario JSON file.
        num_vms: Number of VMs per cloud region.

    Returns:
        ``(success, cost, transfer_time, error_message, details)`` where
        ``details`` contains per-destination breakdown for LLM reflection.
    """
    program = _load_program_module(program_path)
    if program is None:
        return False, 0.0, 0.0, f"Failed to load program from {program_path}", {}

    if not hasattr(program, "search_algorithm"):
        return False, 0.0, 0.0, "Missing search_algorithm function", {}

    try:
        config = _load_config(config_file)
        G = _get_graph(num_vms)
        topology = _run_search_algorithm(program, config, G)

        if topology is None:
            return False, 0.0, 0.0, "search_algorithm returned None", {}

        missing = _find_missing_destinations(topology, config["dest_nodes"], config["num_partitions"])
        if missing:
            return False, 0.0, 0.0, f"No paths for destinations: {', '.join(missing)}", {}

        transfer_time, cost, details = _simulate_and_extract(topology, config, config_file)
        return True, cost, transfer_time, "", details

    except Exception as e:
        return False, 0.0, 0.0, f"Error evaluating {os.path.basename(config_file)}: {e}", {}


def _load_program_module(program_path: str) -> Any:
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    except Exception as e:
        logger.error(f"Failed to load program: {e}\n{traceback.format_exc()}")
        return None


def _load_config(config_file: str) -> dict[str, Any]:
    with open(config_file) as f:
        return json.load(f)


def _get_graph(num_vms: int) -> Any:
    """Return a cached NetworkX graph for the given number of VMs."""
    if num_vms not in _graph_cache:
        from examples.adrs.cloudcast.utils.cloudcast.utils import make_nx_graph

        _graph_cache[num_vms] = make_nx_graph(num_vms=num_vms)
    return _graph_cache[num_vms].copy()


def _run_search_algorithm(program: Any, config: dict[str, Any], G: Any) -> Any:
    topology = program.search_algorithm(
        config["source_node"],
        config["dest_nodes"],
        G,
        config["num_partitions"],
    )
    if topology is not None:
        topology.set_num_partitions(config["num_partitions"])
    return topology


def _find_missing_destinations(topology: Any, dest_nodes: list[str], num_partitions: int) -> list[str]:
    missing = []
    for dest in dest_nodes:
        if dest not in topology.paths:
            missing.append(dest)
            continue
        has_path = any(
            topology.paths[dest].get(str(i)) for i in range(num_partitions)
        )
        if not has_path:
            missing.append(dest)
    return missing


def _simulate_and_extract(
    topology: Any,
    config: dict[str, Any],
    config_file: str,
) -> tuple[float, float, dict[str, Any]]:
    """Run the simulator once and extract both the scores and the detailed breakdown."""
    from examples.adrs.cloudcast.utils.cloudcast.simulator import BCSimulator

    output_dir = tempfile.mkdtemp(prefix="cloudcast_sim_")
    try:
        simulator = BCSimulator(num_vms=2, output_dir=output_dir)
        transfer_time, cost = simulator.evaluate_path(topology, config)
        details = _extract_details_from_simulator(simulator, config, cost, config_file)
        return transfer_time, cost, details
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _extract_details_from_simulator(
    simulator: Any,
    config: dict[str, Any],
    total_cost: float,
    config_file: str,
) -> dict[str, Any]:
    """Extract per-destination route, timing, and cost breakdown from the simulator graph."""
    g = simulator.g
    dest_nodes = config["dest_nodes"]
    num_partitions = config["num_partitions"]

    egress_cost = sum(
        len(data["partitions"]) * simulator.partition_data_vol * data["cost"]
        for _, _, data in g.edges(data=True)
    )
    instance_cost = total_cost - egress_cost

    per_dest_time: dict[str, float] = {}
    per_dest_paths: dict[str, list[str]] = {}
    per_dest_cost: dict[str, float] = {}
    for dst in dest_nodes:
        per_dest_time[dst] = _compute_dest_transfer_time(simulator, g, dst, num_partitions)
        per_dest_paths[dst], per_dest_cost[dst] = _compute_dest_route(simulator, g, dst, num_partitions)

    bottleneck = max(per_dest_time, key=per_dest_time.get)  # type: ignore[arg-type]

    top_edges = sorted(
        [
            {
                "edge": f"{u} -> {v}",
                "cost_per_gb": data["cost"],
                "throughput_gbps": round(data["throughput"], 3),
                "egress_cost": round(
                    len(data["partitions"]) * simulator.partition_data_vol * data["cost"], 4
                ),
            }
            for u, v, data in g.edges(data=True)
        ],
        key=lambda e: e["egress_cost"],
        reverse=True,
    )

    return {
        "config_name": os.path.basename(config_file).split(".")[0],
        "source": config["source_node"],
        "destinations": dest_nodes,
        "num_partitions": num_partitions,
        "egress_cost": round(egress_cost, 4),
        "instance_cost": round(instance_cost, 4),
        "per_dest_time": per_dest_time,
        "per_dest_paths": per_dest_paths,
        "per_dest_cost": per_dest_cost,
        "bottleneck_destination": bottleneck,
        "top_edges": top_edges[:5],
        "data_vol_gb": config.get("data_vol", 4.0),
    }


def _compute_dest_transfer_time(simulator: Any, g: Any, dst: str, num_partitions: int) -> float:
    t = float("-inf")
    for i in range(num_partitions):
        for edge in simulator.paths[dst][str(i)]:
            ed = g[edge[0]][edge[1]]
            t = max(t, len(ed["partitions"]) * simulator.partition_data_vol * 8 / ed["flow"])
    return round(t, 2)


def _compute_dest_route(
    simulator: Any, g: Any, dst: str, num_partitions: int
) -> tuple[list[str], float]:
    seen: set[tuple[str, str]] = set()
    hops: list[str] = []
    egress = 0.0
    for i in range(num_partitions):
        for edge in simulator.paths[dst][str(i)]:
            src_node, dst_node = edge[0], edge[1]
            if (src_node, dst_node) not in seen:
                seen.add((src_node, dst_node))
                if not hops:
                    hops.append(src_node)
                hops.append(dst_node)
            ed = g[src_node][dst_node]
            egress += simulator.partition_data_vol * ed["cost"]
    return hops, round(egress, 4)


# ---------------------------------------------------------------------------
# Side-info builders for GEPA reflection feedback
# ---------------------------------------------------------------------------


def evaluation_failure_info(error_msg: str, example: dict[str, Any]) -> SideInfo:
    """Build a SideInfo dict describing an evaluation failure."""
    return {
        "scores": {"cost": FAILED_SCORE},
        "Input": {"config_file": os.path.basename(example.get("config_file", "?"))},
        "Error": error_msg,
    }


def evaluation_success_info(
    score: float,
    cost: float,
    transfer_time: float,
    example: dict[str, Any],
    details: dict[str, Any],
) -> SideInfo:
    """Build a SideInfo dict with routing and cost breakdown for LLM feedback."""
    dest_summaries = [
        f"  {dst}: route=[{' → '.join(details['per_dest_paths'].get(dst, []))}], "
        f"egress=${details['per_dest_cost'].get(dst, 0):.2f}, "
        f"time={details['per_dest_time'].get(dst, 0):.1f}s"
        for dst in details.get("destinations", [])
    ]
    top_edge_lines = [
        f"  {e['edge']}: ${e['cost_per_gb']}/GB, {e['throughput_gbps']:.2f}Gbps, egress=${e['egress_cost']:.2f}"
        for e in details.get("top_edges", [])
    ]

    return {
        "scores": {"cost_score": score, "raw_cost": cost},
        "Input": {
            "config": details.get("config_name", os.path.basename(example.get("config_file", "?"))),
            "source": details.get("source", "N/A"),
            "destinations": details.get("destinations", []),
            "num_partitions": details.get("num_partitions", "N/A"),
            "data_volume_gb": details.get("data_vol_gb", "N/A"),
        },
        "Output": {
            "total_cost": f"${cost:.4f}",
            "cost_breakdown": f"egress=${details.get('egress_cost', 0):.2f} + instance=${details.get('instance_cost', 0):.2f}",
            "transfer_time": f"{transfer_time:.2f}s",
            "bottleneck_destination": details.get("bottleneck_destination", "N/A"),
        },
        "Per-Destination Breakdown": "\n".join(dest_summaries),
        "Most Expensive Edges (top 5)": "\n".join(top_edge_lines),
    }
