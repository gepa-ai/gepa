import argparse
import json
import logging
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# --- Configuration ---
DATA_PATH = "data/converted_multi_region_aligned"
OUTPUT_DIR = Path("outputs/checkpoint_size_analysis")
TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2
ENV_START_HOURS = 0
MAX_WORKERS = 8

# Different checkpoint sizes to test (in GB)
CHECKPOINT_SIZES = [10, 50, 100, 200, 500]

# Define display names and colors for strategies
STRATEGY_DISPLAY_NAMES = {
    "multi_region_rc_cr_threshold": "Multi-Region Uniform Progress",
    "multi_region_rc_cr_no_cond2": "Multi-Region Uniform Progress (w/o Cond. 2)",
    "multi_region_rc_cr_randomized": "Multi-Region Uniform Progress (Random)",
    "multi_region_rc_cr_reactive": "Multi-Region Uniform Progress (Reactive)",
    "quick_optimal": "Optimal (on Union)",
    "rc_cr_threshold": "Single-Region Uniform Progress",
}

STRATEGY_COLORS = {
    "multi_region_rc_cr_threshold": "mediumseagreen",
    "multi_region_rc_cr_no_cond2": "darkviolet", 
    "multi_region_rc_cr_randomized": "dodgerblue",
    "multi_region_rc_cr_reactive": "crimson",
    "quick_optimal": "orangered",
    "rc_cr_threshold": "gray",
}

# Test scenarios with different region configurations
TEST_SCENARIOS = [
    {
        "name": "2 Zones (us-east-1)",
        "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1"],
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_no_cond2", 
            "multi_region_rc_cr_randomized",
            "multi_region_rc_cr_reactive",
            "quick_optimal",
        ],
        "compare_single": True,
    },
    {
        "name": "2 Regions (cross-region)",
        "regions": ["us-east-1a_v100_1", "us-west-2a_v100_1"],
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_no_cond2",
            "multi_region_rc_cr_randomized", 
            "multi_region_rc_cr_reactive",
            "quick_optimal",
        ],
        "compare_single": True,
    },
]

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


def get_or_create_union_trace(regions: List[str], trace_index: int) -> str:
    """Creates or retrieves a union trace for a specific set of regions and trace index."""
    trace_idx_file = f"{trace_index}.json"
    regions_str = "+".join(sorted(regions))
    union_trace_filename = f"union_{regions_str}_{trace_index}.json"
    union_trace_path = UNION_TRACES_DIR / union_trace_filename

    if union_trace_path.exists():
        return str(union_trace_path)

    all_data, metadata_template, min_len = [], None, float("inf")

    for region_name in regions:
        trace_file_path = Path(DATA_PATH) / region_name / trace_idx_file
        if not trace_file_path.exists():
            raise FileNotFoundError(
                f"Required trace file does not exist: {trace_file_path}"
            )
        with open(trace_file_path, "r") as f:
            trace = json.load(f)
        all_data.append(trace["data"])
        min_len = min(min_len, len(trace["data"]))
        if metadata_template is None:
            metadata_template = trace["metadata"]

    trimmed_data = [data[: int(min_len)] for data in all_data]
    final_union = [
        1 if all(d[i] == 1 for d in trimmed_data) else 0 for i in range(int(min_len))
    ]
    union_trace_content = {"metadata": metadata_template, "data": final_union}

    with open(union_trace_path, "w") as f:
        json.dump(union_trace_content, f)
    logger.info(f"Created Union Trace for {len(regions)} regions at {union_trace_path}")
    return str(union_trace_path)


def generate_cache_filename(strategy: str, env: str, traces: list[str], checkpoint_size: float) -> str:
    """Generates a cache filename including checkpoint size."""
    trace_descs = []
    for trace_path_str in traces:
        trace_path = Path(trace_path_str)
        if "union" in trace_path.name:
            trace_descs.append(trace_path.stem)
        else:
            trace_descs.append(f"{trace_path.parent.name}_{trace_path.stem}")

    trace_identifier = "+".join(sorted(trace_descs))
    safe_strategy_name = strategy.replace("/", "_").replace("\\", "_")
    filename = f"{safe_strategy_name}_{env}_{trace_identifier}_ckpt{checkpoint_size}gb.json"
    
    # If filename is too long, use hash
    if len(filename) > 200:
        content_hash = hashlib.md5(f"{strategy}_{env}_{trace_identifier}_ckpt{checkpoint_size}gb".encode()).hexdigest()
        short_strategy = safe_strategy_name[-30:] if len(safe_strategy_name) > 30 else safe_strategy_name
        filename = f"{short_strategy}_{env}_{content_hash}.json"
    
    return filename


def run_simulation(strategy: str, env_type: str, trace_paths: List[str], checkpoint_size: float) -> float:
    """Run simulation with specific checkpoint size."""
    cache_filename = generate_cache_filename(strategy, env_type, trace_paths, checkpoint_size)
    cache_file = CACHE_DIR / cache_filename

    if cache_file.exists():
        with open(cache_file, "r") as f:
            cost = json.load(f)["mean_cost"]
        logger.info(f"Cache HIT for {strategy} (ckpt={checkpoint_size}GB) -> ${cost:.2f}")
        return cost

    logger.info(f"Cache MISS for: {cache_filename}")
    cmd = [
        "python",
        "./main.py",
        f"--env={env_type}",
        f"--output-dir={OUTPUT_DIR / 'sim_temp'}",
        f"--task-duration-hours={TASK_DURATION_HOURS}",
        f"--deadline-hours={DEADLINE_HOURS}",
        f"--restart-overhead-hours={RESTART_OVERHEAD_HOURS}",
        f"--env-start-hours={ENV_START_HOURS}",
        f"--checkpoint-size-gb={checkpoint_size}",
        f"--strategy={strategy}",
    ]

    if env_type == "trace":
        cmd.append(f"--trace-file={trace_paths[0]}")
    else:
        cmd.extend(["--trace-files"] + trace_paths)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=900
        )
        match = re.search(r"mean:\s*([\d.]+)", result.stdout + result.stderr)
        if not match:
            raise RuntimeError("Could not parse cost from simulation output.")

        mean_cost = float(match.group(1))
        with open(cache_file, "w") as f:
            json.dump({"mean_cost": mean_cost}, f, indent=2)
        return mean_cost
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        RuntimeError,
    ) as e:
        stderr_msg = getattr(e, 'stderr', 'N/A')
        logger.error(f"Simulation failed for {strategy} (ckpt={checkpoint_size}GB). Stderr: {stderr_msg}")
        return np.nan


def create_simulation_task(scenario, strategy, trace_index, checkpoint_size, task_type, region=None):
    """Creates a dictionary representing a single simulation task."""
    return {
        "scenario_name": scenario["name"],
        "regions_in_scenario": scenario["regions"],
        "strategy": strategy,
        "trace_index": trace_index,
        "checkpoint_size": checkpoint_size,
        "task_type": task_type,
        "region": region,
    }


def execute_simulation_task(task):
    """Executes a task with checkpoint size parameter."""
    try:
        trace_idx = task["trace_index"]
        strategy = task["strategy"]
        checkpoint_size = task["checkpoint_size"]
        cost = np.nan

        if task["task_type"] == "multi_region":
            regions = task["regions_in_scenario"]
            if strategy == "quick_optimal":
                env_type = "trace"
                trace_paths = [get_or_create_union_trace(regions, trace_idx)]
            else:
                env_type = "multi_trace"
                trace_paths = [
                    str(Path(DATA_PATH) / r / f"{trace_idx}.json") for r in regions
                ]
                if not all(Path(p).exists() for p in trace_paths):
                    raise FileNotFoundError("One or more regional traces are missing.")
            cost = run_simulation(strategy, env_type, trace_paths, checkpoint_size)

        elif task["task_type"] == "single_region":
            region = task["region"]
            env_type = "trace"
            trace_path = Path(DATA_PATH) / region / f"{trace_idx}.json"
            if not trace_path.exists():
                raise FileNotFoundError(f"Trace file not found for single region: {trace_path}")
            cost = run_simulation(strategy, env_type, [str(trace_path)], checkpoint_size)

        return {**task, "cost": cost}
    except Exception as e:
        logger.error(f"Failed task {task['scenario_name']}/{task['strategy']}/ckpt={task['checkpoint_size']}GB: {e}")
        return {**task, "cost": np.nan}


def plot_checkpoint_analysis(df: pd.DataFrame, num_traces: int):
    """Generate plots showing the impact of checkpoint size on costs."""
    
    # Filter out NaN costs
    df_clean = df.dropna(subset=['cost'])
    
    # Create subplots for each scenario
    scenarios = df_clean['scenario_name'].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6*len(scenarios), 6))
    if len(scenarios) == 1:
        axes = [axes]
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        scenario_df = df_clean[df_clean['scenario_name'] == scenario]
        
        # Plot multi-region strategies
        multi_df = scenario_df[scenario_df['task_type'] == 'multi_region']
        if not multi_df.empty:
            # Calculate mean cost for each strategy-checkpoint combination
            multi_stats = multi_df.groupby(['strategy', 'checkpoint_size'])['cost'].mean().reset_index()
            
            for strategy in multi_stats['strategy'].unique():
                strategy_data = multi_stats[multi_stats['strategy'] == strategy]
                ax.plot(strategy_data['checkpoint_size'], strategy_data['cost'], 
                       marker='o', linewidth=2, markersize=6,
                       color=STRATEGY_COLORS.get(strategy, 'black'),
                       label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy))
        
        # Add single-region baseline if available
        single_df = scenario_df[scenario_df['task_type'] == 'single_region']
        if not single_df.empty:
            single_stats = single_df.groupby(['checkpoint_size'])['cost'].mean().reset_index()
            ax.plot(single_stats['checkpoint_size'], single_stats['cost'],
                   marker='s', linewidth=2, markersize=6, linestyle='--',
                   color=STRATEGY_COLORS.get('rc_cr_threshold', 'gray'),
                   label=STRATEGY_DISPLAY_NAMES.get('rc_cr_threshold', 'Single-Region'))
        
        ax.set_title(f'{scenario}\n(n={num_traces})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint Size (GB)', fontsize=12)
        ax.set_ylabel('Mean Execution Cost ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set x-axis to show all checkpoint sizes
        ax.set_xticks(CHECKPOINT_SIZES)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"checkpoint_size_analysis_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"📊 Checkpoint analysis plot saved to {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-region strategies across different checkpoint sizes."
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=5,
        help="Number of traces to average over for each run.",
    )
    parser.add_argument(
        "--checkpoint-sizes",
        nargs='+',
        type=float,
        default=CHECKPOINT_SIZES,
        help="List of checkpoint sizes to test (in GB)",
    )
    args = parser.parse_args()

    all_tasks = []
    
    for scenario in TEST_SCENARIOS:
        for checkpoint_size in args.checkpoint_sizes:
            # Multi-region strategy tasks
            for strategy in scenario["strategies"]:
                for i in range(args.num_traces):
                    task = create_simulation_task(
                        scenario, strategy, i, checkpoint_size, "multi_region"
                    )
                    all_tasks.append(task)
            
            # Single-region baseline tasks
            if scenario.get("compare_single"):
                for region in scenario["regions"]:
                    for i in range(args.num_traces):
                        task = create_simulation_task(
                            scenario, "rc_cr_threshold", i, checkpoint_size, 
                            "single_region", region=region
                        )
                        all_tasks.append(task)

    logger.info(f"🚀 Found {len(all_tasks)} simulation tasks across {len(args.checkpoint_sizes)} checkpoint sizes.")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(execute_simulation_task, task): task for task in all_tasks
        }
        for future in as_completed(future_to_task):
            result = future.result()
            if result:
                results.append(result)

    if not results:
        logger.error("No simulation tasks completed successfully. Exiting.")
        return

    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / f"checkpoint_size_results_t{args.num_traces}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ All detailed results saved to: {csv_path}")

    plot_checkpoint_analysis(df, args.num_traces)

    # Print summary
    summary = (
        df.groupby(["scenario_name", "strategy", "checkpoint_size"])["cost"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print("\n--- Checkpoint Size Analysis Summary ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()