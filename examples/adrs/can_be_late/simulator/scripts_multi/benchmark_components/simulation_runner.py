"""Simulation execution module - self-contained."""

import subprocess
import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def run_single_simulation(
    strategy: str,
    env_type: str, 
    trace_paths: List[str],
    task_duration_hours: float,
    deadline_hours: float,
    restart_overhead: float,
    checkpoint_size: float,
    output_dir: Path,
    env_start_hours: float = 0,
    strategy_file: Optional[str] = None
) -> Tuple[float, int]:
    """Execute a single simulation and return the cost and number of migrations."""
    
    cmd = [
        "python",
        "main.py",
        f"--env={env_type}",
        f"--output-dir={output_dir}",
        f"--task-duration-hours={task_duration_hours}",
        f"--deadline-hours={deadline_hours}",
        f"--restart-overhead-hours={restart_overhead}",
        f"--env-start-hours={env_start_hours}",
        f"--checkpoint-size-gb={checkpoint_size}",
    ]

    if strategy_file:
        cmd.append(f"--strategy-file={strategy_file}")
    else:
        cmd.append(f"--strategy={strategy}")

    if env_type == "trace":
        cmd.append(f"--trace-file={trace_paths[0]}")
    else:
        cmd.extend(["--trace-files"] + trace_paths)

    try:
        # Run simulation
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract cost from output
        json_file = None
        for line in result.stdout.split("\n"):
            if "Saved to" in line and ".json" in line:
                json_file = line.split("Saved to ")[-1].strip()
                break
        
        # If not found in stdout, look for the expected output file
        if not json_file:
            # Build expected filename based on the parameters
            if env_type == "trace":
                # For single trace, filename format is simpler
                trace_name = os.path.basename(trace_paths[0])
                expected_filename = f"{strategy}-{env_type}-{trace_name}-ddl={deadline_hours}-task=SingleTask({task_duration_hours}h)-over={restart_overhead}"
            else:
                # For multi-trace, join trace names
                trace_names = [os.path.basename(tp) for tp in trace_paths]
                traces_str = ",".join(trace_names)
                expected_filename = f"{strategy}-{env_type}-multi_region_{traces_str}-ddl={deadline_hours}-task=SingleTask({task_duration_hours}h)-over={restart_overhead}"
            
            json_file = output_dir / expected_filename
            if not json_file.exists():
                logger.error(f"Expected output file not found: {json_file}")
                json_file = None

        if json_file:
            # Handle both string and Path objects
            if isinstance(json_file, str):
                json_file_path = json_file
            else:
                json_file_path = str(json_file)
                
            with open(json_file_path, "r") as f:
                data = json.load(f)
                # Try different possible keys for cost
                cost = None
                for key in ["mean_cost", "cost", "total_cost"]:
                    if key in data:
                        cost = float(data[key])
                        break
                
                # Check if costs is an array (new format)
                if cost is None and "costs" in data and isinstance(data["costs"], list) and len(data["costs"]) > 0:
                    cost = float(data["costs"][0])
                
                # Extract migration count
                migrations = 0
                # Check if migrations are directly in the stats (new format)
                if "migrations" in data and isinstance(data["migrations"], list) and len(data["migrations"]) > 0:
                    migrations = data["migrations"][0]  # Single env run
                # Fallback: try to find migration data in the history
                elif "history" in data and data["history"]:
                    # Look for migration count in the last history entry
                    if isinstance(data["history"], list) and len(data["history"]) > 0:
                        last_history = data["history"][0]
                        if isinstance(last_history, list) and len(last_history) > 0:
                            last_entry = last_history[-1]
                            migrations = last_entry.get("MigrationCount", 0)
                # Old format fallbacks
                elif "timeline" in data and isinstance(data["timeline"], list):
                    # Count region switches in timeline
                    prev_region = None
                    for event in data["timeline"]:
                        if "region" in event:
                            if prev_region is not None and event["region"] != prev_region:
                                migrations += 1
                            prev_region = event["region"]
                elif "num_migrations" in data:
                    migrations = data["num_migrations"]
                
                if cost is not None:
                    return cost, migrations
                        
        logger.error(f"Could not extract cost from simulation output")
        return np.nan, 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Simulation failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        
        # Check if it's a task feasibility error
        if "Task infeasible:" in e.stderr:
            # Extract the error message
            for line in e.stderr.split("\n"):
                if "Task infeasible:" in line:
                    raise ValueError(f"INFEASIBLE: {line.strip()}")
        
        # Check if it's a trace data insufficient error
        if "Trace data insufficient:" in e.stderr:
            # Extract the error message
            for line in e.stderr.split("\n"):
                if "Trace data insufficient:" in line:
                    raise ValueError(f"TRACE_INSUFFICIENT: {line.strip()}")
        
        return np.nan, 0
    except Exception as e:
        logger.error(f"Unexpected error in simulation: {e}")
        return np.nan, 0


def check_simulation_errors(results: List[dict]) -> dict:
    """Analyze simulation results and report errors."""
    total_tasks = len(results)
    failed_tasks = sum(1 for r in results if np.isnan(r.get('cost', np.nan)))
    
    errors_by_strategy = {}
    for result in results:
        if np.isnan(result.get('cost', np.nan)):
            strategy = result.get('strategy', 'unknown')
            if strategy not in errors_by_strategy:
                errors_by_strategy[strategy] = 0
            errors_by_strategy[strategy] += 1
    
    return {
        'total_tasks': total_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': (total_tasks - failed_tasks) / total_tasks * 100,
        'errors_by_strategy': errors_by_strategy
    }