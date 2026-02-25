"""Run spot-instance scheduling simulations and build GEPA side-info dicts.

Exposes four functions used by ``evaluate`` in main.py:
  - get_program_path(code)        – write candidate to temp file (cached)
  - syntax_is_valid(program_path) – True when the program compiles and looks correct
  - run_simulation(...)           – execute the simulator subprocess
  - build_side_info(...)          – format results for GEPA's reflective feedback
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
from typing import Any

from gepa.optimize_anything import SideInfo

logger = logging.getLogger(__name__)

FAILED_SCORE = -100_000.0

_SIMULATOR_DIR = Path(__file__).resolve().parent / "simulator"

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
        "Input": {
            "trace_file": os.path.basename(example.get("trace_file", "?")),
            "config": example.get("config", {}),
        },
        "Error": _program_cache["syntax_error"] or "Syntax validation failed",
    }


def _evict_cached_program() -> None:
    if _program_cache["tmpdir"]:
        shutil.rmtree(_program_cache["tmpdir"], ignore_errors=True)
    _program_cache.update(code=None, path=None, tmpdir=None, syntax_ok=None, syntax_error=None)


def _write_program_to_cache(code: str) -> None:
    tmpdir = tempfile.mkdtemp(prefix="cant_be_late_eval_")
    path = os.path.join(tmpdir, "strategy.py")
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
        if "class" not in code or "Strategy" not in code:
            return False, "No Strategy class found"
        if "_step" not in code:
            return False, "No _step method found"
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Simulation execution
# ---------------------------------------------------------------------------


def run_simulation(
    program_path: str,
    trace_file: str,
    config: dict[str, Any],
    timeout: int = 300,
) -> tuple[bool, float, str, dict[str, Any]]:
    """Run the scheduler simulator and return (success, cost, error_msg, details).

    Args:
        program_path: Path to the candidate strategy file.
        trace_file: Path to the spot-availability trace JSON.
        config: Dict with ``duration``, ``deadline``, and ``overhead`` (hours).
        timeout: Subprocess timeout in seconds.

    Returns:
        ``(success, cost, error_message, details)`` where ``details`` contains
        rich timeline information for LLM reflection feedback.
    """
    output_dir = tempfile.mkdtemp(prefix="cant_be_late_run_")
    cmd = _build_simulator_command(program_path, trace_file, config, output_dir)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(_SIMULATOR_DIR))

        if proc.returncode != 0 or "mean:" not in proc.stdout:
            shutil.rmtree(output_dir, ignore_errors=True)
            return False, 0.0, _extract_error(proc), {}

        cost = _parse_cost_from_output(proc.stdout)
        details = _extract_simulation_details(output_dir, trace_file, config)
        shutil.rmtree(output_dir, ignore_errors=True)
        return True, cost, "", details

    except subprocess.TimeoutExpired:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, 0.0, f"Timeout on trace {os.path.basename(trace_file)}", {}
    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, 0.0, f"Error: {e}", {}


def _build_simulator_command(
    program_path: str,
    trace_file: str,
    config: dict[str, Any],
    output_dir: str,
) -> list[str]:
    return [
        sys.executable,
        str(_SIMULATOR_DIR / "main.py"),
        f"--strategy-file={program_path}",
        "--env=trace",
        f"--trace-file={os.path.abspath(trace_file)}",
        f"--task-duration-hours={config['duration']}",
        f"--deadline-hours={config['deadline']}",
        f"--restart-overhead-hours={config['overhead']}",
        "--silent",
        f"--output-dir={output_dir}",
    ]


def _extract_error(proc: subprocess.CompletedProcess) -> str:  # type: ignore[type-arg]
    return (proc.stderr or proc.stdout or "Simulation failed").strip()[:500]


def _parse_cost_from_output(stdout: str) -> float:
    for line in stdout.splitlines():
        if "mean:" in line:
            return float(line.split("mean:")[1].split(";")[0].strip())
    raise ValueError("No 'mean:' line found in simulator output")


def _extract_simulation_details(
    output_dir: str,
    trace_file: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    stats = _load_simulation_stats(output_dir)
    if stats is None:
        return {}
    return _build_cli_segments_summary(stats, trace_file)


# ---------------------------------------------------------------------------
# Side-info builders for GEPA reflection feedback
# ---------------------------------------------------------------------------


def simulation_failure_info(error_msg: str, example: dict[str, Any]) -> SideInfo:
    """Build a SideInfo dict describing a simulation execution failure."""
    config = example.get("config", {})
    return {
        "scores": {"cost": FAILED_SCORE},
        "Input": {
            "trace_file": os.path.basename(example.get("trace_file", "?")),
            "duration": f"{config.get('duration', '?')}h",
            "deadline": f"{config.get('deadline', '?')}h",
            "overhead": f"{config.get('overhead', '?')}h",
        },
        "Error": error_msg,
    }


def simulation_success_info(
    score: float,
    example: dict[str, Any],
    details: dict[str, Any],
) -> SideInfo:
    """Build a SideInfo dict with timeline and cost breakdown for LLM feedback."""
    config = example.get("config", {})
    cost = -score
    timeline_events = details.get("timeline_events", [])
    timeline_str = " | ".join(timeline_events[:12]) if timeline_events else "N/A"
    if len(timeline_events) > 12:
        timeline_str += " | ..."

    return {
        "scores": {"cost": score},
        "Input": {
            "trace_file": os.path.basename(example.get("trace_file", "?")),
            "duration": f"{config.get('duration', '?')}h",
            "deadline": f"{config.get('deadline', '?')}h",
            "overhead": f"{config.get('overhead', '?')}h",
            "spot_availability": details.get("spot_availability", "N/A"),
        },
        "Output": {
            "cost": f"${cost:.2f}",
            "timeline": timeline_str,
            "segments": (
                f"SPOT={details.get('spot_segments', 0)}, "
                f"ON_DEMAND={details.get('ondemand_segments', 0)}, "
                f"restarts={details.get('restart_count', 0)}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Internal: parse the simulator's JSON output into a human-readable summary
# ---------------------------------------------------------------------------


def _load_simulation_stats(output_dir: str) -> dict[str, Any] | None:
    try:
        for path in sorted(Path(output_dir).iterdir()):
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception:
        pass
    return None


def _build_cli_segments_summary(stats: dict[str, Any], trace_file: str) -> dict[str, Any]:
    """Distil raw simulator JSON into a compact dict suitable for LLM feedback."""
    history_batches = stats.get("history") or []
    if not history_batches or not history_batches[0]:
        return {}
    history = history_batches[0]

    gap_hours = _infer_gap_hours(stats, history)
    segments = _find_instance_segments(history)
    if not segments:
        return {}

    spot_availability = _extract_spot_availability(trace_file, gap_hours)
    timeline_events, spot_count, ondemand_count, restart_count = _build_timeline_events(
        segments, gap_hours
    )

    costs = stats.get("costs") or []
    avg_cost = sum(costs) / len(costs) if costs else 0.0

    return {
        "timeline_events": timeline_events,
        "spot_availability": spot_availability,
        "spot_segments": spot_count,
        "ondemand_segments": ondemand_count,
        "restart_count": restart_count,
        "avg_cost": avg_cost,
    }


def _infer_gap_hours(stats: dict[str, Any], history: list[dict[str, Any]]) -> float:
    metadata = stats.get("env", {}).get("metadata", {})
    gap_seconds = metadata.get("gap_seconds") or stats.get("env", {}).get("gap_seconds")
    if not gap_seconds and len(history) > 1:
        gap_seconds = (history[1].get("Elapsed", 0) or 0) - (history[0].get("Elapsed", 0) or 0)
    return (gap_seconds / 3600.0) if gap_seconds else 0.0


def _build_timeline_events(
    segments: dict[int, list[tuple]],
    gap_hours: float,
) -> tuple[list[str], int, int, int]:
    events: list[str] = []
    spot_count = 0
    ondemand_count = 0
    restart_count = 0

    for region, segs in sorted(segments.items()):
        for seg_idx, (start_tick, end_tick, inst_type, progress, had_overhead) in enumerate(segs):
            inst_norm = _normalize_cluster_type(inst_type)
            if inst_norm == "SPOT":
                spot_count += 1
            elif inst_norm == "ON_DEMAND":
                ondemand_count += 1

            start_h = start_tick * gap_hours
            end_h = (end_tick + 1) * gap_hours
            type_abbr = "S" if inst_norm == "SPOT" else ("OD" if inst_norm == "ON_DEMAND" else "NA")
            parts = [f"{start_h:.1f}-{end_h:.1f}:{type_abbr}@R{region}[{progress:.0f}%]"]
            if had_overhead:
                parts.append("overhead")
            if seg_idx > 0:
                restart_count += 1
                parts.append("restart")
            events.append(" ".join(parts))

    return events, spot_count, ondemand_count, restart_count


def _find_instance_segments(
    history: list[dict[str, Any]],
) -> dict[int, list[tuple[int, int, str, float, bool]]]:
    """Identify contiguous runs of the same instance type per region."""
    segments: dict[int, list[tuple[int, int, str, float, bool]]] = defaultdict(list)
    current: dict[int, dict[str, Any]] = {}

    for tick_idx, tick in enumerate(history):
        task_done = tick.get("Task/Done(seconds)", 0.0)
        task_target = tick.get("Task/Target(seconds)", 1.0)
        progress = (task_done / task_target * 100.0) if task_target > 0 else 0.0
        has_overhead = (tick.get("Strategy/RemainingRestartOverhead(seconds)", 0.0) or 0.0) > 0

        raw_active = tick.get("ActiveInstances") or {}
        active_instances = raw_active if isinstance(raw_active, dict) else {}

        if not active_instances:
            fallback = (
                _normalize_cluster_type(tick.get("ClusterType"))
                or _normalize_cluster_type(tick.get("RequestType"))
                or _normalize_cluster_type(tick.get("Strategy/ClusterType"))
            )
            if fallback in ("SPOT", "ON_DEMAND"):
                active_instances = {"0": fallback}

        active_regions: set[int] = set()
        for region_str, inst_type in active_instances.items():
            region = int(region_str)
            active_regions.add(region)
            if region not in current:
                current[region] = {"start": tick_idx, "type": inst_type, "progress": progress, "overhead": has_overhead}
            elif current[region]["type"] != inst_type:
                seg = current[region]
                segments[region].append((seg["start"], max(seg["start"], tick_idx - 1), seg["type"], seg["progress"], seg["overhead"]))
                current[region] = {"start": tick_idx, "type": inst_type, "progress": progress, "overhead": has_overhead}
            else:
                current[region]["progress"] = progress
                if has_overhead:
                    current[region]["overhead"] = True

        for region in set(current) - active_regions:
            seg = current.pop(region)
            segments[region].append((seg["start"], tick_idx - 1, seg["type"], seg["progress"], seg["overhead"]))

    for region, seg in current.items():
        segments[region].append((seg["start"], len(history) - 1, seg["type"], seg["progress"], seg["overhead"]))

    return dict(segments)


def _extract_spot_availability(trace_file: str, gap_hours: float) -> str:
    """Summarise the spot availability pattern from the trace file as a string."""
    if not trace_file or not os.path.exists(trace_file):
        return "N/A"
    try:
        with open(trace_file) as f:
            data = json.load(f).get("data", [])
        parts: list[str] = []
        current_state = None
        start_tick = 0
        for i, val in enumerate(data):
            has_spot = (val == 0)
            if current_state is None:
                current_state, start_tick = has_spot, i
            elif current_state != has_spot:
                parts.append(f"{start_tick * gap_hours:.1f}-{(i - 1) * gap_hours:.1f}:{'S' if current_state else 'X'}")
                current_state, start_tick = has_spot, i
                if len(parts) >= 10:
                    parts.append("...")
                    break
        if len(parts) < 10 and current_state is not None:
            parts.append(f"{start_tick * gap_hours:.1f}-{len(data) * gap_hours:.1f}:{'S' if current_state else 'X'}")
        return " | ".join(parts)
    except Exception:
        return "N/A"


def _normalize_cluster_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper in ("SPOT",) or "SPOT" in upper:
            return "SPOT"
        if upper in ("ON_DEMAND", "ONDEMAND", "OD") or "ON_DEMAND" in upper:
            return "ON_DEMAND"
        if upper.isdigit():
            return _normalize_cluster_type(int(upper))
        return None
    if isinstance(value, (int, float)):
        return "SPOT" if int(value) == 2 else ("ON_DEMAND" if int(value) == 3 else None)
    return None
