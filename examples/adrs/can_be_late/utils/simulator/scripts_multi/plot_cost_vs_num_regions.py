import os
import json
import subprocess
import re
import argparse
from pathlib import Path
import logging
from typing import List, Dict
from collections import defaultdict
import time
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
DATA_PATH = "data/converted_multi_region"
OUTPUT_DIR = Path("outputs/cost_vs_num_regions_analysis")
TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
UNION_TRACES_DIR = OUTPUT_DIR / "union_traces"
UNION_TRACES_DIR.mkdir(exist_ok=True)

# These are the regions available in the specified DATA_PATH
# The order determines the sequence of addition in the plot.
ALL_AVAILABLE_REGIONS = [
    "us-east-2a_v100_1",
    "us-west-2c_v100_1",
    "us-east-1d_v100_1",
    "us-east-2b_v100_1",
    "us-west-2a_v100_1",
    "us-east-1f_v100_1",
    "us-east-1a_v100_1",
    "us-west-2b_v100_1",
    "us-east-1c_v100_1",
]

# shuffle the regions
# random.shuffle(ALL_AVAILABLE_REGIONS)


def get_or_create_union_trace(regions: List[str], trace_index: int) -> str:
    """
    Create or retrieve a union trace for a specific trace file index across multiple regions.
    The union trace is available if at least one of the underlying region traces is available.
    """
    trace_idx_file = f"{trace_index}.json"
    regions_str = "+".join(sorted(regions))
    union_trace_filename = f"union_{regions_str}_{trace_index}.json"
    union_trace_path = UNION_TRACES_DIR / union_trace_filename

    if union_trace_path.exists():
        return str(union_trace_path)

    all_data, metadata_template, min_len = [], None, float("inf")

    for region_name in regions:
        trace_file_path = Path(DATA_PATH) / region_name / trace_idx_file
        assert trace_file_path.exists(), f"Trace file {trace_file_path} does not exist"

        with open(trace_file_path, "r") as f:
            trace = json.load(f)
        all_data.append(trace["data"])
        min_len = min(min_len, len(trace["data"]))
        if metadata_template is None:
            metadata_template = trace["metadata"]
        elif trace["metadata"] != metadata_template:
            logger.warning("Metadata mismatch between regions")
            logger.warning(trace["metadata"])
            logger.warning(metadata_template)

    min_len = int(min_len)

    # Trim all data to the minimum common length
    trimmed_data = [data[:min_len] for data in all_data]
    final_union = []

    for i in range(min_len):
        # A value of 1 means preempted. The union is available (0) if ANY region is available (0).
        # It is preempted (1) only if ALL regions are preempted (1).
        is_all_preempted = all(
            data_point == 1 for data_point in [d[i] for d in trimmed_data]
        )
        final_union.append(1 if is_all_preempted else 0)

    union_trace_content = {"metadata": metadata_template, "data": final_union}

    with open(union_trace_path, "w") as f:
        json.dump(union_trace_content, f)
    logger.info(
        f"Created Union Trace for {len(regions)} regions (index {trace_index}) at {union_trace_path}"
    )
    return str(union_trace_path)


def generate_human_readable_cache_filename(
    strategy: str, env: str, traces: list[str], params: dict
) -> str:
    """Generates a human-readable cache filename based on strategy, env, traces, and key parameters."""
    trace_descs = []
    trace_indices = set()
    for trace_path_str in traces:
        trace_path = Path(trace_path_str)
        trace_descs.append(trace_path.parent.name)
        if trace_path.stem.isdigit():
            trace_indices.add(int(trace_path.stem))

    trace_identifier = "+".join(sorted(list(set(trace_descs))))
    safe_strategy_name = strategy.replace("/", "_")

    # Add key parameters to the filename for uniqueness
    param_str = f"d{params['deadline']}_t{params['task_duration']}_r{params['restart_overhead']}"

    # Describe the set of traces used
    if len(trace_indices) > 1:
        trace_indices_str = f"{len(traces)}traces"
    elif not trace_indices:
        # Handle cases like union traces which have descriptive names but no index
        trace_indices_str = Path(traces[0]).stem
    else:
        # For single traces
        trace_indices_str = f"trace{list(trace_indices)[0]}"

    return f"{safe_strategy_name}_{env}_{trace_identifier}_{trace_indices_str}_{param_str}.json"


def run_simulation(
    strategy: str, env_type: str, trace_paths: List[str], sim_params: dict
) -> float:
    """
    A general-purpose simulation runner with a human-readable, self-describing caching system.
    """
    if not trace_paths:
        logger.warning(
            f"Skipping simulation for {strategy} as no trace paths were provided."
        )
        return np.nan

    # Generate the new, descriptive cache filename instead of an MD5 hash
    cache_filename = generate_human_readable_cache_filename(
        strategy, env_type, trace_paths, sim_params
    )
    cache_file = CACHE_DIR / cache_filename

    # --- Cache Reading Logic ---
    if cache_file.exists():
        logger.info(
            f"Cache HIT for {strategy} on {len(trace_paths)} traces using file: {cache_filename}"
        )
        with open(cache_file, "r") as f:
            return json.load(f)["mean_cost"]

    # --- Cache Miss Logic ---
    logger.info(f"Cache MISS for: {cache_filename}")

    cmd = [
        "python",
        "./main.py",
        f"--strategy={strategy}",
        f"--env={env_type}",
        f"--output-dir={OUTPUT_DIR / 'sim_temp'}",
        f"--task-duration-hours={sim_params['task_duration']}",
        f"--deadline-hours={sim_params['deadline']}",
        f"--restart-overhead-hours={sim_params['restart_overhead']}",
    ]
    if env_type == "trace":
        cmd.append(f"--trace-file={trace_paths[0]}")
    else:
        cmd.extend(["--trace-files"] + trace_paths)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=900
        )
        output = result.stdout + result.stderr
        match = re.search(r"mean:\s*([\d.]+)", output)
        if not match:
            logger.error(
                f"Could not parse cost from output for command: {' '.join(cmd)}"
            )
            logger.error(f"Output was:\n{output}")
            return np.nan

        mean_cost = float(match.group(1))

        # --- Cache Writing Logic ---
        with open(cache_file, "w") as f:
            cache_content = {
                "mean_cost": mean_cost,
                "parameters": {
                    "strategy": strategy,
                    "env": env_type,
                    "traces": sorted(trace_paths),
                    **sim_params,
                },
            }
            json.dump(cache_content, f, indent=2)

        return mean_cost

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(
            f"FATAL: Simulation failed for {strategy}. Command: {' '.join(e.cmd)}\nStderr:\n{e.stderr}"
        )
        return np.nan


def plot_costs(df: pd.DataFrame, num_traces: int, params: dict):
    """Generates and saves a plot of costs vs. number of regions."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        df["num_regions"],
        df["optimal_cost"],
        marker="o",
        linestyle="--",
        label="Optimal (on Union Trace)",
    )
    ax.plot(
        df["num_regions"],
        df["multi_region_cost"],
        marker="s",
        linestyle="-",
        label="Multi-Region Uniform Progress",
    )
    ax.plot(
        df["num_regions"],
        df["multi_region_no_condition2_cost"],
        marker="D",
        linestyle="-",
        label="Multi-Region Uniform Progress (No Cond. 2)",
    )
    ax.plot(
        df["num_regions"],
        df["avg_single_cost"],
        marker="^",
        linestyle=":",
        label="Average Single-Region Uniform Progress",
    )
    ax.plot(
        df["num_regions"],
        df["avg_single_no_condition2_cost"],
        marker="v",
        linestyle=":",
        label="Average Single-Region Uniform Progress (No Cond. 2)",
    )

    ax.set_xlabel("Number of Available Regions")
    ax.set_ylabel("Mean Execution Cost ($)")
    ax.set_title(
        f"Multi-Region Strategy Cost vs. Number of Regions\n"
        f"({num_traces} traces/region, D={params['deadline']}, T={params['task_duration']}, R={params['restart_overhead']})"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.set_xticks(df["num_regions"])  # Ensure ticks are integers for number of regions

    plt.tight_layout()
    plot_filename = f"cost_vs_num_regions_t{num_traces}.png"
    plot_path = OUTPUT_DIR / plot_filename
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-region strategy costs against the number of available regions."
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=10,
        help="Number of traces to use per region for the analysis.",
    )
    parser.add_argument(
        "--task-duration",
        type=float,
        default=TASK_DURATION_HOURS,
        help="Task duration in hours.",
    )
    parser.add_argument(
        "--deadline", type=float, default=DEADLINE_HOURS, help="Job deadline in hours."
    )
    parser.add_argument(
        "--restart-overhead",
        type=float,
        default=RESTART_OVERHEAD_HOURS,
        help="Restart overhead in hours.",
    )
    args = parser.parse_args()

    sim_params = {
        "task_duration": args.task_duration,
        "deadline": args.deadline,
        "restart_overhead": args.restart_overhead,
    }

    results_data = []
    trace_indices_to_run = list(range(args.num_traces))

    for i in range(1, len(ALL_AVAILABLE_REGIONS) + 1):
        num_regions = i
        current_regions = ALL_AVAILABLE_REGIONS[:num_regions]
        logger.info(
            f"\n--- Analyzing with {num_regions} region(s): {current_regions} ---"
        )

        # --- 1. Optimal Cost (Union of all traces for the i regions) ---
        optimal_costs = []
        for trace_idx in trace_indices_to_run:
            try:
                union_trace_path = get_or_create_union_trace(current_regions, trace_idx)
                cost = run_simulation(
                    strategy="quick_optimal",
                    env_type="trace",  # Run on a single union trace
                    trace_paths=[union_trace_path],
                    sim_params=sim_params,
                )
                optimal_costs.append(cost)
            except Exception as e:
                logger.error(
                    f"Failed to calculate optimal cost for trace {trace_idx}: {e}"
                )

        avg_optimal_cost = np.nanmean(optimal_costs) if optimal_costs else np.nan

        # --- 2. Multi-Region Heuristic Cost ---
        if num_regions > 1:
            # This strategy is designed to work with multiple regions (traces) at once.
            # We run one simulation per trace index, giving the strategy all regions for that trace.
            multi_region_costs = []
            for trace_idx in trace_indices_to_run:
                # Get all regional traces for this specific index
                trace_paths_for_idx = [
                    str(Path(DATA_PATH) / r / f"{trace_idx}.json") for r in current_regions
                ]
                trace_paths_for_idx = [p for p in trace_paths_for_idx if Path(p).exists()]

                if len(trace_paths_for_idx) == len(current_regions):
                    cost = run_simulation(
                        strategy="multi_region_rc_cr_threshold",
                        env_type="multi_trace",
                        trace_paths=trace_paths_for_idx,
                        sim_params=sim_params,
                    )
                    multi_region_costs.append(cost)
                else:
                    logger.warning(
                        f"Skipping multi-region simulation for trace index {trace_idx} due to missing files."
                    )
            avg_multi_region_cost = (
                np.nanmean(multi_region_costs) if multi_region_costs else np.nan
            )
        else:
            # When n_regions=1, the multi-region strategy is equivalent to the single-region one.
            # We will calculate the single-region cost later and reuse it.
            avg_multi_region_cost = np.nan

        # --- 2.2. Multi-Region No Condition2 Heuristic Cost ---
        if num_regions > 1:
            # This is the multi-region version of no_condition2 strategy.
            multi_region_no_condition2_costs = []
            for trace_idx in trace_indices_to_run:
                # Get all regional traces for this specific index
                trace_paths_for_idx = [
                    str(Path(DATA_PATH) / r / f"{trace_idx}.json") for r in current_regions
                ]
                trace_paths_for_idx = [p for p in trace_paths_for_idx if Path(p).exists()]

                if len(trace_paths_for_idx) == len(current_regions):
                    cost = run_simulation(
                        strategy="multi_region_rc_cr_quality_bar",
                        env_type="multi_trace",
                        trace_paths=trace_paths_for_idx,
                        sim_params=sim_params,
                    )
                    multi_region_no_condition2_costs.append(cost)
                else:
                    logger.warning(
                        f"Skipping multi-region no_condition2 simulation for trace index {trace_idx} due to missing files."
                    )

            avg_multi_region_no_condition2_cost = (
                np.nanmean(multi_region_no_condition2_costs)
                if multi_region_no_condition2_costs
                else np.nan
            )
        else:
            avg_multi_region_no_condition2_cost = np.nan

        # --- 3. Average of Single-Region Costs ---
        # Here we calculate the average cost of running the *single region* heuristic on each region independently.
        all_single_region_costs = []
        for region in current_regions:
            # Get all traces for this specific region
            region_traces = [
                str(Path(DATA_PATH) / region / f"{idx}.json")
                for idx in trace_indices_to_run
            ]
            region_traces = [p for p in region_traces if Path(p).exists()]

            # The simulation should be run for each trace file and then averaged.
            costs_for_this_region = []
            for trace_path in region_traces:
                cost = run_simulation(
                    strategy="rc_cr_threshold",
                    env_type="trace",
                    trace_paths=[trace_path],
                    sim_params=sim_params,
                )
                costs_for_this_region.append(cost)

            avg_cost_for_region = (
                np.nanmean(costs_for_this_region) if costs_for_this_region else np.nan
            )
            all_single_region_costs.append(avg_cost_for_region)

        avg_single_cost = (
            np.nanmean(all_single_region_costs) if all_single_region_costs else np.nan
        )

        # For n_regions=1, reuse the single-region cost for the multi-region strategy
        if num_regions == 1:
            avg_multi_region_cost = avg_single_cost

        # --- 3.2. Average of Single-Region No Condition2 Costs ---
        # Here we calculate the average cost of running the *single region* no_condition2 heuristic on each region independently.
        all_single_region_no_condition2_costs = []
        for region in current_regions:
            # Get all traces for this specific region
            region_traces = [
                str(Path(DATA_PATH) / region / f"{idx}.json")
                for idx in trace_indices_to_run
            ]
            region_traces = [p for p in region_traces if Path(p).exists()]

            # The simulation should be run for each trace file and then averaged.
            costs_for_this_region = []
            for trace_path in region_traces:
                cost = run_simulation(
                    strategy="rc_cr_threshold_no_condition2",
                    env_type="trace",
                    trace_paths=[trace_path],
                    sim_params=sim_params,
                )
                costs_for_this_region.append(cost)

            avg_cost_for_region = (
                np.nanmean(costs_for_this_region) if costs_for_this_region else np.nan
            )
            all_single_region_no_condition2_costs.append(avg_cost_for_region)

        avg_single_no_condition2_cost = (
            np.nanmean(all_single_region_no_condition2_costs)
            if all_single_region_no_condition2_costs
            else np.nan
        )

        if num_regions == 1:
            avg_multi_region_no_condition2_cost = avg_single_no_condition2_cost

        logger.info(
            f"Results for {num_regions} regions: Optimal=${avg_optimal_cost:.2f}, "
            f"Multi-Region=${avg_multi_region_cost:.2f}, Multi-Region(NoC2)=${avg_multi_region_no_condition2_cost:.2f}, "
            f"Avg-Single=${avg_single_cost:.2f}, Avg-Single(NoC2)=${avg_single_no_condition2_cost:.2f}"
        )

        results_data.append(
            {
                "num_regions": num_regions,
                "optimal_cost": avg_optimal_cost,
                "multi_region_cost": avg_multi_region_cost,
                "multi_region_no_condition2_cost": avg_multi_region_no_condition2_cost,
                "avg_single_cost": avg_single_cost,
                "avg_single_no_condition2_cost": avg_single_no_condition2_cost,
            }
        )

    # Convert results to DataFrame and plot
    if results_data:
        df = pd.DataFrame(results_data)
        print("\n--- Final Results ---")
        print(df)
        plot_costs(df, args.num_traces, sim_params)
    else:
        logger.warning("No results were generated. Cannot create plot.")


if __name__ == "__main__":
    main()
