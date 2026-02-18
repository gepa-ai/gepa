"""Batch worker script that processes multiple tasks on a single SkyPilot cluster."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

# Add parent directory to path to import simulation_runner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation_runner import run_single_simulation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def execute_task(task: Dict, params: Dict) -> Dict:
    """Execute a single simulation task."""
    try:
        # Generate environment type and trace paths
        env_type, trace_paths = get_trace_paths_for_task(task, params["DATA_PATH"])
        
        # Calculate deadline based on task duration and ratio
        task_duration = params["TASK_DURATION_HOURS"]
        deadline_hours = task_duration * task["deadline_ratio"]
        
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
        
        return {**task, "cost": cost, "migrations": migrations}
        
    except ValueError as e:
        if "INFEASIBLE:" in str(e):
            error_msg = str(e).replace("INFEASIBLE: ", "")
            return {**task, "cost": float('nan'), "error_type": "Task Infeasible", "error_details": error_msg}
        elif "TRACE_INSUFFICIENT:" in str(e):
            error_msg = str(e).replace("TRACE_INSUFFICIENT: ", "")
            return {**task, "cost": float('nan'), "error_type": "Trace Insufficient", "error_details": error_msg}
        else:
            return {**task, "cost": float('nan'), "error_type": "ValueError", "error_details": str(e)}
    except Exception as e:
        return {**task, "cost": float('nan'), "error_type": type(e).__name__, "error_details": str(e)}


def process_single_task(args):
    """Process a single task (for multiprocessing)."""
    task, params = args
    return execute_task(task, params)


def main():
    parser = argparse.ArgumentParser(description="Batch worker for SkyPilot execution")
    parser.add_argument("--tasks-file", type=str, required=True,
                        help="JSON file containing list of tasks")
    parser.add_argument("--params-json", type=str, required=True,
                        help="JSON string or file containing parameters")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Parse inputs
    with open(args.tasks_file, 'r') as f:
        tasks = json.load(f)
    
    # Handle params - can be JSON string or file path
    if os.path.exists(args.params_json):
        with open(args.params_json, 'r') as f:
            params = json.load(f)
    else:
        params = json.loads(args.params_json)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(tasks)} tasks in batch mode")
    
    # Determine number of workers (use CPU count but cap at 16 for better utilization)
    num_workers = min(multiprocessing.cpu_count(), 16, len(tasks))
    logger.info(f"Using {num_workers} parallel workers")
    
    # Process tasks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_task, (task, params)): task
            for task in tasks
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                logger.info(f"✅ [{completed}/{len(tasks)}] Completed: {task['scenario_name']} - {task['strategy']} - trace {task['trace_index']}")
            except Exception as e:
                logger.error(f"❌ [{completed}/{len(tasks)}] Failed: {task['scenario_name']} - {task['strategy']} - {e}")
                results.append({
                    **task,
                    "cost": float('nan'),
                    "error_type": "ProcessingError",
                    "error_details": str(e)
                })
    
    # Save all results to individual files
    for i, result in enumerate(results):
        output_file = output_dir / f"result_{i}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Also save a summary file
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_tasks": len(tasks),
            "completed_tasks": len(results),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Batch processing completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()