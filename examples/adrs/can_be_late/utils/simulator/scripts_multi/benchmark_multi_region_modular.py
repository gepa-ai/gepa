#!/usr/bin/env python3
"""
Modular multi-region benchmark script following Bacterial programming principles.
Each component is self-contained and can be easily copied/modified.
"""

import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List

# Import modular components
# Setup logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('DEBUG') else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from benchmark_components.cache_manager import generate_cache_filename, load_from_cache, save_to_cache
from benchmark_components.simulation_runner import run_single_simulation, check_simulation_errors
from benchmark_components.plot_generator import (
    create_restart_overhead_plot, 
    create_checkpoint_size_plot,
    create_scenario_bar_plot,
    create_deadline_sensitivity_plot,
    create_cost_heatmap,
    create_region_scaling_plot,
    create_scenario_availability_plot,
)
from benchmark_components.error_reporter import print_error_summary, log_simulation_failure
from benchmark_components.scenario_config import EXPERIMENT_SCENARIOS, DEFAULT_PARAMS

try:
    from benchmark_components.skypilot_executor import execute_tasks_with_skypilot
except ImportError:
    logger.warning("SkyPilot not available. Use local execution only.")
    execute_tasks_with_skypilot = None


def get_or_create_union_trace(regions: List[str], trace_index: int, data_path: str) -> str:
    """Creates or retrieves a union trace for a specific set of regions and trace index."""
    # Create union traces directory
    union_traces_dir = Path("outputs/multi_region_scenario_analysis/union_traces")
    union_traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate union trace filename
    regions_str = "+".join(sorted(regions))
    union_trace_filename = f"union_{regions_str}_{trace_index}.json"
    union_trace_path = union_traces_dir / union_trace_filename
    
    # Return if already exists
    if union_trace_path.exists():
        return str(union_trace_path)
    
    # Load all regional traces
    all_data, metadata_template, min_len = [], None, float("inf")
    for region_name in regions:
        trace_file_path = Path(data_path) / region_name / f"{trace_index}.json"
        if not trace_file_path.exists():
            raise FileNotFoundError(f"Required trace file does not exist: {trace_file_path}")
        
        with open(trace_file_path, "r") as f:
            trace = json.load(f)
            all_data.append(trace["data"])
            min_len = min(min_len, len(trace["data"]))
            if metadata_template is None:
                metadata_template = trace["metadata"]
    
    # Create union trace (1 only if ALL regions are preempted, 0 if ANY region is available)
    trimmed_data = [data[:int(min_len)] for data in all_data]
    final_union = [
        1 if all(d[i] == 1 for d in trimmed_data) else 0 for i in range(int(min_len))
    ]
    
    # Save union trace
    union_trace_content = {"metadata": metadata_template, "data": final_union}
    with open(union_trace_path, "w") as f:
        json.dump(union_trace_content, f)
    
    logger.info(f"Created union trace for {len(regions)} regions at {union_trace_path}")
    return str(union_trace_path)


def get_trace_paths_for_task(task: Dict, data_path: str) -> tuple[str, List[str]]:
    """Get environment type and trace paths based on task configuration."""
    if task["task_type"] == "single_region":
        env_type = "trace"
        trace_paths = [str(Path(data_path) / task["region"] / f"{task['trace_index']}.json")]
        return env_type, trace_paths
    
    # Multi-region task
    regions = task["regions_in_scenario"]
    
    # Check if strategy should use union trace
    if task["strategy"] == "quick_optimal" or task.get("trace_mode") == "union":
        env_type = "trace"
        union_trace_path = get_or_create_union_trace(regions, task["trace_index"], data_path)
        trace_paths = [union_trace_path]
    elif task.get("trace_mode") == "single" or task.get("trace_mode") == "best_single":
        env_type = "trace"
        target_region = regions[0]  # Default to first region
        trace_paths = [str(Path(data_path) / target_region / f"{task['trace_index']}.json")]
    else:
        env_type = "multi_trace"
        trace_paths = [
            str(Path(data_path) / r / f"{task['trace_index']}.json") 
            for r in regions
        ]
    
    return env_type, trace_paths


def create_simulation_task(
    scenario: Dict,
    strategy: str,
    trace_index: int,
    task_type: str,
    checkpoint_size: float,
    restart_overhead: float,
    deadline_ratio: float,
    region: Optional[str] = None,
    strategy_file: Optional[str] = None
) -> Dict:
    """Create a single simulation task dictionary."""
    return {
        "scenario_name": scenario["name"],
        "regions_in_scenario": scenario["regions"],
        "num_regions": len(scenario["regions"]),
        "strategy": strategy,
        "trace_index": trace_index,
        "task_type": task_type,
        "region": region,
        "strategy_file": strategy_file,
        "checkpoint_size": checkpoint_size,
        "restart_overhead": restart_overhead,
        "deadline_ratio": deadline_ratio,
    }


def add_trace_mode_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add trace mode baselines: run strategies on union/best_single/average_single trace modes."""
    
    # Filter single-region results for baseline calculations
    single_region_df = df[df["task_type"] == "single_region"].copy()
    
    if single_region_df.empty:
        logger.info("No single-region results found, skipping trace mode baseline calculations")
        return df
    
    # Calculate trace mode baselines for each scenario and strategy combination
    baseline_rows = []
    
    for scenario_name in single_region_df["scenario_name"].unique():
        scenario_data = single_region_df[single_region_df["scenario_name"] == scenario_name]
        
        # Get unique strategies tested in single regions
        strategies_tested = scenario_data["strategy"].unique()
        
        for strategy in strategies_tested:
            strategy_data = scenario_data[scenario_data["strategy"] == strategy]
            
            # Group by region and calculate mean cost for each region
            region_means = strategy_data.groupby("region")["cost"].mean().reset_index()
            
            if not region_means.empty and not region_means["cost"].isna().all():
                # Find best and average single region costs for this strategy
                best_cost = region_means["cost"].min()
                avg_cost = region_means["cost"].mean()
                
                # Get a representative row for copying metadata
                sample_row = strategy_data.iloc[0].copy()
                
                # Create best single region entry for this strategy
                best_row = sample_row.copy()
                best_row.update({
                    "strategy": f"{strategy}_best_single",
                    "task_type": "trace_mode_baseline",
                    "region": None,
                    "cost": best_cost,
                    "migrations": 0,
                    "trace_mode": "best_single",
                    "base_strategy": strategy
                })
                baseline_rows.append(best_row)
                
                # Create average single region entry for this strategy
                avg_row = sample_row.copy()
                avg_row.update({
                    "strategy": f"{strategy}_average_single",
                    "task_type": "trace_mode_baseline",
                    "region": None,
                    "cost": avg_cost,
                    "migrations": 0,
                    "trace_mode": "average_single", 
                    "base_strategy": strategy
                })
                baseline_rows.append(avg_row)
                
                logger.info(f"Added {strategy} trace mode baselines for {scenario_name}: best=${best_cost:.2f}, avg=${avg_cost:.2f}")
    
    # Add baseline rows to the original dataframe
    if baseline_rows:
        baseline_df = pd.DataFrame(baseline_rows)
        df = pd.concat([df, baseline_df], ignore_index=True)
        logger.info(f"Added {len(baseline_rows)} trace mode baseline entries to results")
    
    return df


def execute_simulation_with_cache(task: Dict, cache_dir: Path, params: Dict) -> Dict:
    """Execute simulation with caching support."""
    try:
        # Generate cache key
        cache_key_strategy = task.get("strategy_file") or task["strategy"]
        
        # Generate environment type and trace paths
        env_type, trace_paths = get_trace_paths_for_task(task, params["DATA_PATH"])
        
        # Calculate deadline based on task duration and ratio
        task_duration = params["TASK_DURATION_HOURS"]
        deadline_hours = task_duration * task["deadline_ratio"]
        
        # Check cache
        cache_filename = generate_cache_filename(
            cache_key_strategy, env_type, trace_paths,
            task["checkpoint_size"], task["restart_overhead"],
            deadline_hours
        )
        cache_file = cache_dir / cache_filename
        
        cost = load_from_cache(cache_file)
        if cost is not None:
            logger.info(f"Cache HIT for {cache_key_strategy} -> ${cost:.2f}")
            # For cached results, we don't have migration data
            return {**task, "cost": cost, "migrations": -1}  # -1 indicates no migration data
        
        logger.info(f"Cache MISS for: {cache_filename}")
        
        # Run simulation
        cost, migrations = run_single_simulation(
            task["strategy"],
            env_type,
            trace_paths,
            params["TASK_DURATION_HOURS"],
            deadline_hours,
            task["restart_overhead"],
            task["checkpoint_size"],
            Path(params["OUTPUT_DIR"]) / "sim_temp",
            params["ENV_START_HOURS"],
            task.get("strategy_file")
        )
        
        # Save to cache if successful
        if not pd.isna(cost):
            save_to_cache(cache_file, cost)
        
        return {**task, "cost": cost, "migrations": migrations}
        
    except ValueError as e:
        if "INFEASIBLE:" in str(e):
            # Task is infeasible - extract the detailed message
            error_msg = str(e).replace("INFEASIBLE: ", "")
            log_simulation_failure(task, e)
            return {**task, "cost": float('nan'), "migrations": 0, "error_type": "Task Infeasible", "error_details": error_msg}
        elif "TRACE_INSUFFICIENT:" in str(e):
            # Trace data is insufficient - extract the detailed message
            error_msg = str(e).replace("TRACE_INSUFFICIENT: ", "")
            log_simulation_failure(task, e)
            return {**task, "cost": float('nan'), "migrations": 0, "error_type": "Trace Insufficient", "error_details": error_msg}
        else:
            log_simulation_failure(task, e)
            return {**task, "cost": float('nan'), "migrations": 0, "error_type": "ValueError", "error_details": str(e)}
    except Exception as e:
        log_simulation_failure(task, e)
        return {**task, "cost": float('nan'), "migrations": 0, "error_type": type(e).__name__, "error_details": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Modular multi-region benchmark")
    parser.add_argument("--num-traces", type=int, default=5)
    parser.add_argument("--checkpoint-sizes", nargs='+', type=float, default=[50.0])
    parser.add_argument("--restart-overhead-hours", nargs='+', type=float, default=[0.2])
    parser.add_argument("--deadline-ratios", nargs='+', type=float, default=[1.083], 
                        help="Deadline as ratio of task duration (e.g., 1.083 = 52h/48h)")
    parser.add_argument("--task-duration-hours", type=float, default=DEFAULT_PARAMS["TASK_DURATION_HOURS"],
                        help="Task duration in hours (default: 48)")
    parser.add_argument("--scenarios", nargs='*', help="Specific scenarios to run")
    parser.add_argument("--output-dir", default=DEFAULT_PARAMS["OUTPUT_DIR"])
    parser.add_argument("--skip-infeasible", action='store_true',
                        help="Skip tasks that are known to be infeasible (deadline < task_duration + overhead)")
    
    # SkyPilot option
    parser.add_argument("--use-skypilot", action='store_true',
                        help="Use SkyPilot for parallel execution on cloud (auto-selects best resources)")
    
    args = parser.parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Build task list
    all_tasks = []
    skipped_tasks = 0
    scenarios = EXPERIMENT_SCENARIOS
    if args.scenarios:
        scenarios = [s for s in scenarios if s["name"] in args.scenarios]
    
    task_duration = args.task_duration_hours
    
    for scenario in scenarios:
        for checkpoint_size in args.checkpoint_sizes:
            for restart_overhead in args.restart_overhead_hours:
                for deadline_ratio in args.deadline_ratios:
                    # Check if task is feasible
                    deadline_hours = task_duration * deadline_ratio
                    min_time_needed = task_duration + restart_overhead
                    
                    if args.skip_infeasible and min_time_needed > deadline_hours:
                        logger.info(f"Skipping infeasible configuration: task={task_duration}h + overhead={restart_overhead}h > deadline={deadline_hours}h")
                        skipped_tasks += len(scenario["strategies"]) * args.num_traces
                        if scenario.get("compare_single_region"):
                            skipped_tasks += len(scenario["regions"]) * args.num_traces
                        continue
                    
                    # Multi-region strategies
                    for strategy in scenario["strategies"]:
                        for i in range(args.num_traces):
                            all_tasks.append(create_simulation_task(
                                scenario, strategy, i, "multi_region",
                                checkpoint_size, restart_overhead, deadline_ratio
                            ))
                    
                    # Single-region comparison
                    if scenario.get("compare_single_region"):
                        for region in scenario["regions"]:
                            for i in range(args.num_traces):
                                all_tasks.append(create_simulation_task(
                                    scenario, DEFAULT_PARAMS["SINGLE_REGION_STRATEGY"],
                                    i, "single_region", checkpoint_size, restart_overhead,
                                    deadline_ratio, region=region
                                ))
    
    logger.info(f"🚀 Running {len(all_tasks)} simulation tasks")
    if skipped_tasks > 0:
        logger.info(f"📋 Skipped {skipped_tasks} infeasible tasks (use --skip-infeasible to filter these)")
    
    # Create params dict with actual task duration
    params = DEFAULT_PARAMS.copy()
    params["TASK_DURATION_HOURS"] = args.task_duration_hours
    
    # Execute tasks based on execution mode
    if args.use_skypilot:
        if execute_tasks_with_skypilot is None:
            logger.error("SkyPilot is not available. Please install it with 'pip install skypilot'")
            return
        logger.info(f"🌩️  Using SkyPilot for cloud parallel execution")
        results = execute_tasks_with_skypilot(
            all_tasks,
            cache_dir,
            params,
            auto_down=True  # Auto-terminate clusters after completion
        )
    else:
        # Execute tasks in parallel locally
        results = []
        with ThreadPoolExecutor(max_workers=DEFAULT_PARAMS["MAX_WORKERS"]) as executor:
            futures = {
                executor.submit(execute_simulation_with_cache, task, cache_dir, params): task
                for task in all_tasks
            }
            
            for future in futures:
                result = future.result()
                results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    
    # Add trace mode baseline calculations
    df = add_trace_mode_baselines(df)
    
    csv_path = output_dir / f"scenario_results_t{args.num_traces}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Results saved to: {csv_path}")
    
    # Generate plots
    logger.info("📊 Generating visualizations...")
    
    # 1. Main scenario comparison bar plot (always generate)
    # Get scenario configurations
    scenarios = EXPERIMENT_SCENARIOS
    if args.scenarios:
        scenarios = [s for s in scenarios if s["name"] in args.scenarios]
    create_scenario_bar_plot(df, args.num_traces, output_dir, scenarios)
    
    # 2. Restart overhead analysis (if multiple values)
    if len(args.restart_overhead_hours) > 1:
        create_restart_overhead_plot(df, args.num_traces, args.restart_overhead_hours, output_dir)
    
    # 3. Checkpoint size analysis (if multiple values)
    if len(args.checkpoint_sizes) > 1:
        create_checkpoint_size_plot(df, args.num_traces, args.checkpoint_sizes, output_dir)
    
    # 4. Deadline sensitivity analysis (if multiple deadline ratios)
    if len(args.deadline_ratios) > 1:
        create_deadline_sensitivity_plot(df, args.num_traces, args.deadline_ratios, output_dir)
    
    # 5. Cost heatmap (if both deadline and checkpoint have multiple values)
    if len(args.deadline_ratios) > 1 and len(args.checkpoint_sizes) > 1:
        create_cost_heatmap(df, args.num_traces, args.deadline_ratios, args.checkpoint_sizes, output_dir)
    
    # 6. Region scaling analysis (if there are multiple scenarios with different region counts)
    unique_region_counts = df['num_regions'].unique()
    if len(unique_region_counts) > 1:
        create_region_scaling_plot(df, args.num_traces, output_dir)
    
    # 7. Scenario availability statistics (always generate)
    create_scenario_availability_plot(scenarios, output_dir)
    
    # 8. Migration statistics (if migration data is available)
    if 'migrations' in df.columns:
        pass  # Migration visualization not implemented yet
    
    logger.info("📊 All visualizations generated successfully")
    
    # Print error summary
    print_error_summary(results)


if __name__ == "__main__":
    main()