"""
Evaluator helpers for the Can't Be Late problem.

This module provides:
- Syntax validation (stage1)
- Simulation execution via subprocess
- CLI segment extraction for detailed LLM feedback
- Fitness function factory for GEPA optimization
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from gepa.optimize_anything import SideInfo

from examples.adrs.can_be_late.trace_config import TRACE_OVERHEADS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Score for failed programs
FAILED_SCORE = -100000.0

# Path to the simulator (relative to this file)
CURRENT_DIR = Path(__file__).resolve().parent
# NOTE: Update this path to point to your simulator installation
SIMULATOR_DIR = CURRENT_DIR / "simulator"

# Evaluation configuration - job duration/deadline matrix
JOB_CONFIGS = [
    {"duration": 48, "deadline": 52},
    {"duration": 48, "deadline": 70},
    {"duration": 48, "deadline": 92},
]

# Restart overhead values to test
CHANGEOVER_DELAYS = TRACE_OVERHEADS


class SimulationFailure(Exception):
    """Exception raised when simulation fails."""

    def __init__(self, error_msg: str, stdout: str, stderr: str):
        self.stdout = stdout
        self.stderr = stderr
        self.error_msg = error_msg
        super().__init__(f"{error_msg}\nSTDOUT: {stdout}\nSTDERR: {stderr}")


def evaluate_stage1(program_path: str) -> dict:
    """
    Stage 1: Quick syntax and import check.

    Args:
        program_path: Path to the strategy Python file

    Returns:
        Dict with 'runs_successfully' (0.0 or 1.0) and optional 'error', 'score'
    """
    try:
        with open(program_path, "r") as f:
            code = f.read()

        # Try to compile the code
        compile(code, program_path, "exec")

        # Basic validation - check for required class structure
        if "class" not in code or "Strategy" not in code:
            return {
                "runs_successfully": 0.0,
                "error": "No Strategy class found",
                "score": FAILED_SCORE,
            }

        if "_step" not in code:
            return {
                "runs_successfully": 0.0,
                "error": "No _step method found",
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


# ============================================================================
# CLI Segment Extraction Functions
# These extract detailed timeline information from simulations for LLM feedback
# ============================================================================


def _cluster_type_to_name(value) -> str | None:
    """Align assorted cluster type encodings to canonical names."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            value = int(raw)
        else:
            upper = raw.upper()
            if upper in ("SPOT",):
                return "SPOT"
            if upper in ("ON_DEMAND", "ONDEMAND", "OD"):
                return "ON_DEMAND"
            if upper in ("NONE", "0", "NOPE"):
                return None
            if "SPOT" in upper:
                return "SPOT"
            if "ON_DEMAND" in upper or "ONDEMAND" in upper:
                return "ON_DEMAND"
            return None
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv == 2:
            return "SPOT"
        if iv == 3:
            return "ON_DEMAND"
        return None
    return None


def _find_instance_segments(
    history: list[dict[str, Any]]
) -> dict[int, list[tuple[int, int, str, float, bool]]]:
    """Reproduce the timeline segmentation logic for continuous runs."""
    segments: dict[int, list[tuple[int, int, str, float, bool]]] = defaultdict(list)
    current_segment: dict[int, dict[str, Any]] = {}

    for tick_idx, tick_data in enumerate(history):
        task_done = tick_data.get("Task/Done(seconds)", 0.0)
        task_target = tick_data.get("Task/Target(seconds)", 1.0)
        progress = (task_done / task_target * 100.0) if task_target > 0 else 0.0

        restart_overhead_remaining = tick_data.get("Strategy/RemainingRestartOverhead(seconds)", 0.0) or 0.0

        raw_active = tick_data.get("ActiveInstances") or {}
        active_instances = raw_active if isinstance(raw_active, dict) else {}

        if not active_instances:
            fallback = (
                _cluster_type_to_name(tick_data.get("ClusterType"))
                or _cluster_type_to_name(tick_data.get("RequestType"))
                or _cluster_type_to_name(tick_data.get("Strategy/ClusterType"))
            )
            if fallback in ("SPOT", "ON_DEMAND"):
                active_instances = {"0": fallback}

        active_regions = set()
        for region_str, inst_type_str in active_instances.items():
            region = int(region_str)
            active_regions.add(region)

            if region not in current_segment:
                current_segment[region] = {
                    "start": tick_idx,
                    "type": inst_type_str,
                    "progress": progress,
                    "had_overhead": restart_overhead_remaining > 0,
                }
            else:
                seg = current_segment[region]
                if seg["type"] != inst_type_str:
                    segments[region].append(
                        (
                            seg["start"],
                            max(seg["start"], tick_idx - 1),
                            seg["type"],
                            seg["progress"],
                            seg.get("had_overhead", False),
                        )
                    )
                    current_segment[region] = {
                        "start": tick_idx,
                        "type": inst_type_str,
                        "progress": progress,
                        "had_overhead": restart_overhead_remaining > 0,
                    }
                else:
                    seg["progress"] = progress
                    if restart_overhead_remaining > 0:
                        seg["had_overhead"] = True

        ended_regions = set(current_segment.keys()) - active_regions
        for region in ended_regions:
            seg = current_segment[region]
            segments[region].append(
                (
                    seg["start"],
                    tick_idx - 1,
                    seg["type"],
                    seg["progress"],
                    seg.get("had_overhead", False),
                )
            )
            del current_segment[region]

    for region, seg in current_segment.items():
        segments[region].append(
            (
                seg["start"],
                len(history) - 1,
                seg["type"],
                seg["progress"],
                seg.get("had_overhead", False),
            )
        )

    return dict(segments)


def _extract_spot_availability(trace_file: str, gap_hours: float) -> str:
    """Extract spot availability pattern from trace file."""
    if not trace_file or not os.path.exists(trace_file):
        return "N/A"

    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        trace_availability = trace_data.get("data", [])

        # In trace data: 0 = spot available, 1 = spot unavailable/preempted
        spot_availability_segments = []
        current_state = None
        start_tick = 0

        for i, availability in enumerate(trace_availability):
            has_spot = (availability == 0)

            if current_state is None:
                current_state = has_spot
                start_tick = i
            elif current_state != has_spot:
                start_h = start_tick * gap_hours
                end_h = (i - 1) * gap_hours
                state_str = "S" if current_state else "X"
                spot_availability_segments.append(f"{start_h:.1f}-{end_h:.1f}:{state_str}")
                current_state = has_spot
                start_tick = i

                if len(spot_availability_segments) >= 10:
                    spot_availability_segments.append("...")
                    break

        if len(spot_availability_segments) < 10 and current_state is not None:
            start_h = start_tick * gap_hours
            end_h = min(len(trace_availability) * gap_hours, start_h + 100)
            state_str = "S" if current_state else "X"
            spot_availability_segments.append(f"{start_h:.1f}-{end_h:.1f}:{state_str}")

        return " | ".join(spot_availability_segments)
    except Exception:
        return "N/A"


def _generate_cli_segments_summary(
    stats: dict[str, Any], trace_file: str | None = None
) -> dict[str, Any]:
    """Produce a compact CLI-friendly description of instance usage."""
    history_batches = stats.get("history") or []
    if not history_batches:
        return {}

    history = history_batches[0]
    if not history:
        return {}

    segments = _find_instance_segments(history)
    if not segments:
        return {}

    # Get gap_hours for conversion
    metadata = stats.get("env", {}).get("metadata", {})
    gap_seconds = metadata.get("gap_seconds") or stats.get("env", {}).get("gap_seconds")
    if not gap_seconds and len(history) > 1:
        gap_seconds = (history[1].get("Elapsed", 0) or 0) - (history[0].get("Elapsed", 0) or 0)
    gap_hours = (gap_seconds / 3600.0) if gap_seconds else 0.0

    # Extract spot availability pattern
    spot_pattern = _extract_spot_availability(trace_file, gap_hours) if trace_file else "N/A"

    args = stats.get("args", {})
    deadline = args.get("deadline_hours", 0.0)
    if isinstance(deadline, list):
        deadline_hours = deadline[0] if deadline else 0.0
    else:
        deadline_hours = float(deadline) if deadline is not None else 0.0

    task_hours_val = args.get("task_duration_hours", 0.0)
    if isinstance(task_hours_val, list):
        task_hours = task_hours_val[0] if task_hours_val else 0.0
    else:
        task_hours = float(task_hours_val) if task_hours_val is not None else 0.0

    final_tick = history[-1]
    task_done_hours = final_tick.get("Task/Done(seconds)", 0.0) / 3600.0
    final_progress = min((task_done_hours / task_hours) * 100.0, 100.0) if task_hours else 0.0

    timeline_events: list[str] = []
    spot_count = 0
    ondemand_count = 0
    migration_count = 0
    restart_count = 0
    total_runtime = 0.0

    def _is_migration(region_idx: int, start_tick: int) -> bool:
        for other_region, other_segs in segments.items():
            if other_region == region_idx:
                continue
            for other_seg in other_segs:
                other_end = other_seg[1]
                if other_end < start_tick and (start_tick - other_end) <= 5:
                    return True
        return False

    for region, segs in sorted(segments.items()):
        for seg_idx, seg in enumerate(segs):
            start_tick, end_tick, inst_type, progress, had_overhead = seg
            duration_hours = max(0, end_tick - start_tick + 1) * gap_hours
            total_runtime += duration_hours

            inst_norm = _cluster_type_to_name(inst_type)
            if inst_norm == "SPOT":
                spot_count += 1
            elif inst_norm == "ON_DEMAND":
                ondemand_count += 1

            start_h = start_tick * gap_hours
            end_h = (end_tick + 1) * gap_hours
            type_abbr = "S" if inst_norm == "SPOT" else ("OD" if inst_norm == "ON_DEMAND" else "NA")
            annotation_parts = [f"{start_h:.1f}-{end_h:.1f}:{type_abbr}@R{region}[{progress:.0f}%]"]
            if had_overhead:
                annotation_parts.append("overhead")

            is_migration = _is_migration(region, start_tick)
            if is_migration:
                migration_count += 1
                annotation_parts.append("migration")
            elif seg_idx > 0:
                restart_count += 1
                annotation_parts.append("restart")

            timeline_events.append(" ".join(annotation_parts).strip())

    costs = stats.get("costs") or []
    avg_cost = sum(costs) / len(costs) if costs else 0.0

    return {
        "timeline_events": timeline_events,
        "spot_availability": spot_pattern,
        "spot_segments": spot_count,
        "ondemand_segments": ondemand_count,
        "migration_count": migration_count,
        "restart_count": restart_count,
        "total_runtime_hours": total_runtime,
        "deadline_hours": deadline_hours,
        "task_hours": task_hours,
        "final_progress_pct": final_progress,
        "avg_cost": avg_cost,
        "gap_hours": gap_hours,
    }


def _load_simulation_stats(output_dir: str) -> tuple[dict[str, Any] | None, str | None]:
    """Read the simulation JSON artifact emitted by the simulator."""
    try:
        for path in sorted(Path(output_dir).iterdir()):
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), str(path)
    except Exception:
        return None, None
    return None, None


# ============================================================================
# Simulation Execution
# ============================================================================


def run_single_simulation(
    program_path: str, trace_file: str, config: dict
) -> tuple[bool, float, str, dict[str, Any]]:
    """
    Run a single simulation.

    Args:
        program_path: Path to the strategy Python file
        trace_file: Path to the trace JSON file
        config: Dict with 'duration', 'deadline', 'overhead'

    Returns:
        Tuple of (success, cost, error_msg, detailed_info)
    """
    trace_file = os.path.abspath(trace_file)
    output_dir = tempfile.mkdtemp(prefix="cant_be_late_run_")

    cmd = [
        sys.executable,
        str(SIMULATOR_DIR / "main.py"),
        f"--strategy-file={program_path}",
        "--env=trace",
        f"--trace-file={trace_file}",
        f"--task-duration-hours={config['duration']}",
        f"--deadline-hours={config['deadline']}",
        f"--restart-overhead-hours={config['overhead']}",
        "--silent",
        f"--output-dir={output_dir}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(SIMULATOR_DIR),
        )

        if result.returncode != 0:
            error_msg = f"Run failed for {os.path.basename(trace_file)}"
            detailed_info = {
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            shutil.rmtree(output_dir, ignore_errors=True)
            return False, 0.0, str(SimulationFailure(error_msg, result.stdout, result.stderr)), detailed_info

        if "mean:" not in result.stdout:
            error_msg = f"No 'mean:' found in output for {os.path.basename(trace_file)}"
            detailed_info = {
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            shutil.rmtree(output_dir, ignore_errors=True)
            return False, 0.0, str(SimulationFailure(error_msg, result.stdout, result.stderr)), detailed_info

        # Parse cost from output
        for line in result.stdout.splitlines():
            if "mean:" in line:
                try:
                    cost_str = line.split("mean:")[1].split(";")[0].strip()
                    cost = float(cost_str)

                    # Extract detailed CLI segments from simulation output
                    detailed_info: dict[str, Any] = {
                        "cost": cost,
                        "trace_file": os.path.basename(trace_file),
                        "config": config,
                    }

                    # Try to load simulation stats for CLI segment extraction
                    stats, _ = _load_simulation_stats(output_dir)
                    if stats:
                        cli_summary = _generate_cli_segments_summary(stats, trace_file)
                        if cli_summary:
                            detailed_info["cli_segments"] = cli_summary

                    shutil.rmtree(output_dir, ignore_errors=True)
                    return True, cost, "", detailed_info
                except Exception as e:
                    error_msg = f"Failed to parse cost from line: {line}\nError: {e}"
                    detailed_info = {"stdout": result.stdout, "stderr": result.stderr}
                    shutil.rmtree(output_dir, ignore_errors=True)
                    return False, 0.0, str(SimulationFailure(error_msg, result.stdout, result.stderr)), detailed_info

        shutil.rmtree(output_dir, ignore_errors=True)
        return False, 0.0, "Could not find cost in output", {"stdout": result.stdout, "stderr": result.stderr}

    except subprocess.TimeoutExpired as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, 0.0, f"Timeout on trace {os.path.basename(trace_file)}", {"timeout": True}
    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, 0.0, f"Error on trace {os.path.basename(trace_file)}: {e}", {"error": str(e)}


# ============================================================================
# Fitness Function
# ============================================================================


def create_fitness_function(timeout: int = 300):
    """
    Create fitness function for GEPA optimization.

    The fitness function evaluates a candidate strategy on a batch of trace samples,
    running simulations and returning scores with diagnostic information.

    Args:
        timeout: Timeout in seconds for each simulation

    Returns:
        Fitness function compatible with GEPA's optimize_anything API
    """

    def fitness_fn(
        candidate: dict[str, str], batch: Sequence[Any], **kwargs
    ) -> list[tuple[float, Any, SideInfo]]:
        """
        Evaluate a candidate strategy on a batch of trace samples.

        Args:
            candidate: Dict with "program" key containing strategy code
            batch: Sequence of sample dicts with 'trace_file' and 'config'

        Returns:
            List of (score, output, side_info) tuples, one per batch item
        """
        program_code = candidate["program"]

        print(f"Evaluating candidate: \n\n{program_code}", flush=True)
        results = []

        # Write program to temp file
        tmpdir = tempfile.mkdtemp(prefix="cant_be_late_eval_")
        program_path = os.path.join(tmpdir, "strategy.py")

        try:
            with open(program_path, "w", encoding="utf-8") as f:
                f.write(program_code)

            # Stage 1: Syntax check
            stage1_result = evaluate_stage1(program_path)
            if stage1_result.get("runs_successfully", 0.0) < 1.0:
                # Syntax error - return failed score for all batch items
                error_msg = stage1_result.get("error", "Syntax validation failed")
                for sample in batch:
                    side_info: SideInfo = {
                        "scores": {"cost": FAILED_SCORE},
                        "Input": {
                            "trace_file": sample.get("trace_file", "unknown"),
                            "config": sample.get("config", {}),
                        },
                        "Error": error_msg,
                        "stage": "stage1",
                    }
                    output = {"error": error_msg, "stage": "stage1"}
                    results.append((FAILED_SCORE, output, side_info))
                return results

            # Stage 2: Run simulations
            for sample in batch:
                trace_file = sample.get("trace_file")
                config = sample.get("config")

                if not trace_file or not config:
                    # Invalid sample
                    side_info = {
                        "scores": {"cost": FAILED_SCORE},
                        "Input": {"trace_file": trace_file, "config": config},
                        "Error": "Invalid sample: missing trace_file or config",
                    }
                    results.append((FAILED_SCORE, {"error": "Invalid sample"}, side_info))
                    continue

                # Run simulation
                success, cost, error_msg, detailed_info = run_single_simulation(
                    program_path, trace_file, config
                )

                if success:
                    # Score is negative cost (lower cost = higher score)
                    score = -cost

                    # Extract CLI segments for detailed feedback
                    cli_segments = detailed_info.get("cli_segments", {})
                    spot_availability = cli_segments.get("spot_availability", "N/A")
                    timeline_events = cli_segments.get("timeline_events", [])

                    # Build timeline string for LLM
                    if timeline_events:
                        timeline_str = " | ".join(timeline_events[:12])  # Limit for prompt size
                        if len(timeline_events) > 12:
                            timeline_str += " | ..."
                    else:
                        timeline_str = "N/A"

                    side_info = {
                        "scores": {"cost": score},
                        "Input": {
                            "trace_file": os.path.basename(trace_file),
                            "duration": f"{config['duration']}h",
                            "deadline": f"{config['deadline']}h",
                            "overhead": f"{config['overhead']}h",
                            "spot_availability": spot_availability,
                        },
                        "Output": {
                            "cost": f"${cost:.2f}",
                            "timeline": timeline_str,
                            "segments": f"SPOT={cli_segments.get('spot_segments', 0)}, ON_DEMAND={cli_segments.get('ondemand_segments', 0)}, restarts={cli_segments.get('restart_count', 0)}",
                        },
                    }
                    output = {
                        "trace_file": trace_file,
                        "config": config,
                        "cost": cost,
                        "score": score,
                        "cli_segments": cli_segments,
                    }
                else:
                    score = FAILED_SCORE
                    side_info = {
                        "scores": {"cost": score},
                        "Input": {
                            "trace_file": os.path.basename(trace_file) if trace_file else "unknown",
                            "duration": f"{config.get('duration', 'N/A')}h",
                            "deadline": f"{config.get('deadline', 'N/A')}h",
                            "overhead": f"{config.get('overhead', 'N/A')}h",
                        },
                        "Error": error_msg,
                    }
                    output = {
                        "trace_file": trace_file,
                        "config": config,
                        "error": error_msg,
                    }

                results.append((score, output, side_info))

        finally:
            # Cleanup temp directory
            shutil.rmtree(tmpdir, ignore_errors=True)

        return results

    return fitness_fn


if __name__ == "__main__":
    # Quick test of the evaluator
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "program_path",
        type=str,
        nargs="?",
        default="initial_strategy.py",
        help="Path to the strategy to evaluate",
    )
    args = parser.parse_args()

    print(f"Testing evaluator with program: {args.program_path}")

    print("\nStage 1 (syntax check):")
    result1 = evaluate_stage1(args.program_path)
    print(json.dumps(result1, indent=2))
