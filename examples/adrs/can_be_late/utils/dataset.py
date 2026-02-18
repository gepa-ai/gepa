"""Load and split spot-availability trace files for the Can't Be Late problem."""

import math
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Trace collection constants
# ---------------------------------------------------------------------------

# Overhead fractions present in the reference dataset
TRACE_OVERHEADS: list[float] = [0.02, 0.20, 0.40]

# Environments used for the held-out test split
_TEST_ENVS = [
    "us-west-2a_k80_1",
    "us-west-2b_k80_1",
    "us-west-2a_v100_1",
    "us-west-2b_v100_1",
]

# Fixed trace IDs that form the test split (legacy-compatible)
_TEST_TRACE_IDS: list[int] = [
    0, 8, 9, 20, 21, 33, 42, 51, 61, 70, 99, 107, 117, 126, 135,
    145, 154, 163, 172, 182, 191, 219, 228, 238, 247, 256, 266, 275, 284, 294,
]

# Job configurations used to expand each trace into multiple evaluation scenarios
_JOB_CONFIGS = [
    {"duration": 48, "deadline": 52},
    {"duration": 48, "deadline": 70},
    {"duration": 48, "deadline": 92},
]

_MAX_TRAINVAL_SAMPLES = 2000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_trace_dataset(
    dataset_root: str,
    seed: int = 0,
    max_traces_per_split: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return train / val / test splits of trace-based evaluation samples.

    Each sample is a dict with keys ``trace_file`` and ``config`` (a dict
    with ``duration``, ``deadline``, and ``overhead`` in hours).

    Args:
        dataset_root: Root directory containing the ``ddl=…`` trace folders.
        seed: Random seed used to shuffle train/val samples.
        max_traces_per_split: Cap each split at this many samples (useful for
            quick smoke-test runs).
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    test_samples = _build_test_samples(root)
    train_samples, val_samples = _build_trainval_samples(root, test_samples, seed)

    if max_traces_per_split is not None:
        train_samples = train_samples[:max_traces_per_split]
        val_samples = val_samples[:max_traces_per_split]
        test_samples = test_samples[:max_traces_per_split]

    return {"train": train_samples, "val": val_samples, "test": test_samples}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_test_samples(root: Path) -> list[dict[str, Any]]:
    """Select the fixed test traces and expand them into evaluation samples."""
    test_traces: list[str] = []
    for env in _TEST_ENVS:
        for trace_id in _TEST_TRACE_IDS:
            path = (
                root
                / "ddl=search+task=48+overhead=0.20"
                / "real"
                / env
                / "traces"
                / "random_start"
                / f"{trace_id}.json"
            )
            if path.is_file():
                test_traces.append(str(path.resolve()))
    return _expand_traces_to_samples(test_traces)


def _build_trainval_samples(
    root: Path,
    test_samples: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect all non-test traces, shuffle, then split evenly into train/val."""
    test_trace_paths = {s["trace_file"] for s in test_samples}
    all_traces = _list_all_unique_traces(root)
    remaining_traces = [t for t in all_traces if t not in test_trace_paths]

    all_samples = _expand_traces_to_samples(remaining_traces)

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    if len(all_samples) > _MAX_TRAINVAL_SAMPLES:
        all_samples = all_samples[:_MAX_TRAINVAL_SAMPLES]

    midpoint = len(all_samples) // 2
    val_samples = all_samples[:midpoint]
    train_samples = all_samples[midpoint:]
    return train_samples, val_samples


def _list_all_unique_traces(root: Path) -> list[str]:
    """Return one canonical path per (environment, trace_id) pair."""
    envs = _discover_envs(root)
    unique: list[str] = []
    for env in envs:
        trace_ids = _discover_trace_ids(root, env)
        for tid in sorted(trace_ids, key=lambda s: int(s) if s.isdigit() else s):
            canonical = _canonical_trace_path(root, env, tid)
            if canonical:
                unique.append(canonical)
    return unique


def _discover_envs(root: Path) -> list[str]:
    envs: set[str] = set()
    for overhead in TRACE_OVERHEADS:
        base = root / f"ddl=search+task=48+overhead={overhead:.2f}" / "real"
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.name.endswith(".json") and (d / "traces" / "random_start").is_dir():
                envs.add(d.name)
    return sorted(envs)


def _discover_trace_ids(root: Path, env: str) -> set[str]:
    ids: set[str] = set()
    for overhead in TRACE_OVERHEADS:
        trace_dir = root / f"ddl=search+task=48+overhead={overhead:.2f}" / "real" / env / "traces" / "random_start"
        if trace_dir.is_dir():
            ids.update(p.stem for p in trace_dir.glob("*.json"))
    return ids


def _canonical_trace_path(root: Path, env: str, trace_id: str) -> str | None:
    """Return the first existing trace file path for (env, trace_id)."""
    for overhead in [0.20, 0.02, 0.40]:
        path = (
            root
            / f"ddl=search+task=48+overhead={overhead:.2f}"
            / "real"
            / env
            / "traces"
            / "random_start"
            / f"{trace_id}.json"
        )
        if path.is_file():
            return str(path.resolve())
    return None


def _expand_traces_to_samples(trace_files: list[str]) -> list[dict[str, Any]]:
    """Cross-product each trace with job configs and overhead values."""
    samples: list[dict[str, Any]] = []
    for trace_path in trace_files:
        for job_cfg in _JOB_CONFIGS:
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
