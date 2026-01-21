"""
Trace dataset loading for Can't Be Late optimization.

This module provides functions to load and split trace files into
train/val/test sets with proper configuration expansion.
"""

import math
import random
from pathlib import Path
from typing import Any

from examples.adrs.can_be_late.trace_config import (
    LEGACY_ENV_PATHS,
    LEGACY_TRACE_TARGET,
    TRACE_OVERHEADS,
    TRACE_SAMPLE_IDS,
)

# Evaluation configuration - job duration/deadline matrix
JOB_CONFIGS = [
    {"duration": 48, "deadline": 52},
    {"duration": 48, "deadline": 70},
    {"duration": 48, "deadline": 92},
]


def _is_random_start_trace(path: Path) -> bool:
    """Check if path is a valid random_start trace file."""
    return path.is_file() and path.suffix == ".json" and "traces" in path.parts


def _list_all_traces(root: Path) -> list[str]:
    """List all trace files under root."""
    trace_paths = {
        str(path.resolve())
        for path in root.glob("**/traces/random_start/*.json")
        if _is_random_start_trace(path)
    }
    if not trace_paths:
        raise FileNotFoundError(f"No trace files found under {root}")
    return sorted(trace_paths)


def _canonical_trace_path(
    root: Path,
    env: str,
    trace_id: str,
    prefer_overheads: list[float] | None = None,
) -> str | None:
    """Return a canonical path for (env, trace_id), preferring given overheads."""
    if prefer_overheads is None:
        prefer_overheads = [0.20, 0.02, 0.40]
    for ov in prefer_overheads:
        p = (
            root
            / f"ddl=search+task=48+overhead={ov:.2f}"
            / "real"
            / env
            / "traces"
            / "random_start"
            / f"{trace_id}.json"
        )
        if p.is_file():
            return str(p.resolve())
    return None


def _list_all_unique_traces(root: Path, env_paths: list[str]) -> list[str]:
    """List unique trace files by (env, id) using canonical paths."""
    unique: list[str] = []
    for env in env_paths:
        ids: set[str] = set()
        for ov in TRACE_OVERHEADS:
            d = root / f"ddl=search+task=48+overhead={ov:.2f}" / "real" / env / "traces" / "random_start"
            if not d.is_dir():
                continue
            for path in d.glob("*.json"):
                ids.add(path.stem)
        for tid in sorted(ids, key=lambda s: int(s) if s.isdigit() else s):
            canon = _canonical_trace_path(root, env, tid)
            if canon:
                unique.append(canon)
    return unique


def _legacy_select_test_unique_traces(
    root: Path,
    trace_target: int = LEGACY_TRACE_TARGET,
    env_paths: list[str] = LEGACY_ENV_PATHS,
) -> list[str]:
    """Select test traces matching legacy evaluator selection."""
    selected: list[str] = []
    for env in env_paths:
        for tid_int in TRACE_SAMPLE_IDS:
            tid = str(tid_int)
            # Fixed: always use 0.20 overhead folder
            p = root / f"ddl=search+task=48+overhead=0.20" / "real" / env / "traces" / "random_start" / f"{tid}.json"
            if p.is_file():
                selected.append(str(p.resolve()))
    return selected


def _build_trace_configs(trace_files: list[str]) -> list[dict[str, Any]]:
    """Expand each trace path into per-scenario samples."""
    if not trace_files:
        return []

    samples: list[dict[str, Any]] = []
    for trace_path in trace_files:
        for job_cfg in JOB_CONFIGS:
            for overhead in TRACE_OVERHEADS:
                samples.append(
                    {
                        "trace_file": trace_path,
                        "config": {
                            "duration": job_cfg["duration"],
                            "deadline": job_cfg["deadline"],
                            "overhead": overhead,
                        },
                    }
                )
    return samples


def _build_trace_configs_for_test(trace_files: list[str]) -> list[dict[str, Any]]:
    """Build test samples as a clean grid: 3 overhead × 3 deadline."""
    if not trace_files:
        return []

    samples: list[dict[str, Any]] = []
    for trace_path in trace_files:
        for ov in TRACE_OVERHEADS:
            for job_cfg in JOB_CONFIGS:
                samples.append(
                    {
                        "trace_file": trace_path,
                        "config": {
                            "duration": job_cfg["duration"],
                            "deadline": job_cfg["deadline"],
                            "overhead": ov,
                        },
                    }
                )
    return samples


def load_trace_dataset(
    dataset_root: str,
    seed: int = 0,
    max_traces_per_split: int | None = None,
    use_legacy_style_test: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Build train/val/test splits with legacy-compatible test selection.

    Args:
        dataset_root: Root directory containing trace files
        seed: Random seed for shuffling
        max_traces_per_split: Maximum samples per split (for quick testing)
        use_legacy_style_test: Use legacy test trace selection

    Returns:
        Dict with 'train', 'val', 'test' keys, each containing list of samples
    """
    root_path = Path(dataset_root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    # Build env list for unique enumeration
    envs: list[str] = []
    for ov in TRACE_OVERHEADS:
        base = root_path / f"ddl=search+task=48+overhead={ov:.2f}" / "real"
        if not base.is_dir():
            continue
        for d in base.iterdir():
            # Skip duplicate directories with .json suffix
            if d.name.endswith('.json'):
                continue
            if (d / "traces" / "random_start").is_dir():
                envs.append(d.name)
    envs = sorted(set(envs))

    # Legacy test selection
    if use_legacy_style_test:
        test_traces = _legacy_select_test_unique_traces(
            root=root_path,
            trace_target=LEGACY_TRACE_TARGET,
            env_paths=LEGACY_ENV_PATHS,
        )
    else:
        test_traces = []

    # Build test samples
    test_samples = _build_trace_configs_for_test(test_traces)

    # Get all unique traces and exclude test traces for train/val
    all_unique = _list_all_unique_traces(root_path, envs)
    test_trace_set = set(test_traces)
    remaining_traces = [t for t in all_unique if t not in test_trace_set]

    # Expand remaining traces to samples
    remaining_samples = _build_trace_configs(remaining_traces)

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(remaining_samples)

    # Limit total samples
    total_remaining = len(remaining_samples)
    max_trainval_samples = 2000

    if total_remaining > max_trainval_samples:
        remaining_samples = remaining_samples[:max_trainval_samples]
        total_remaining = len(remaining_samples)

    # Split evenly for balanced train/val
    val_count = total_remaining // 2
    train_samples = remaining_samples[val_count:]
    val_samples = remaining_samples[:val_count]

    # Apply max_traces_per_split limit
    if max_traces_per_split is not None:
        train_samples = train_samples[:max_traces_per_split]
        val_samples = val_samples[:max_traces_per_split]
        test_samples = test_samples[:max_traces_per_split]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

