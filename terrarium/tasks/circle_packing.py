"""Circle packing task: pack 26 circles in a unit square to maximize sum of radii.

This is a single-task problem (no dataset). The candidate is Python code.
"""

from __future__ import annotations

from typing import Any

from terrarium.registry import register_task
from terrarium.task import Task

DESCRIPTION = """\
Pack 26 non-overlapping circles inside a unit square [0,1]x[0,1].
The candidate is Python code defining `main(timeout, current_best_solution)` that
returns {'circles': np.ndarray(26,3), 'all_scores': [float]}.
Score = sum of all circle radii (higher is better).
"""

INITIAL_CANDIDATE = '''\
import numpy as np

def main(timeout, current_best_solution):
    """Circle packing: returns dict with 'circles' (n,3) and 'all_scores'."""
    n = 26

    if current_best_solution is not None:
        circles = current_best_solution.copy()
    else:
        centers = np.zeros((n, 2))
        centers[0] = [0.5, 0.5]

        # Ring of 8 around center
        angles = 2 * np.pi * np.arange(8) / 8
        centers[1:9] = np.column_stack([0.5 + 0.3 * np.cos(angles), 0.5 + 0.3 * np.sin(angles)])

        # Outer ring for remaining 17
        angles = 2 * np.pi * np.arange(17) / 17
        centers[9:] = np.column_stack([0.5 + 0.7 * np.cos(angles), 0.5 + 0.7 * np.sin(angles)])

        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        circles = np.column_stack([centers, radii])

    return {'circles': circles, 'all_scores': [float(circles[:, 2].sum())]}


def compute_max_radii(centers):
    """Compute maximum radii that don't overlap and stay in unit square."""
    n = len(centers)
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
'''


def evaluate(candidate: str) -> tuple[float, dict[str, Any]]:
    """Evaluate circle packing code. Returns (sum_radii, info)."""
    from examples.circle_packing.utils import execute_code, compute_multiple_metrics

    result = execute_code(candidate, timeout=600, current_best_solution=None)
    score = result.get("validation_details", {}).get("sum_radii", 0.0)
    all_scores = result.get("all_scores", [0.0])
    metrics = compute_multiple_metrics(all_scores) if all_scores else {}

    return score, {
        "score": score,
        "metrics": metrics,
        "circles": result.get("circles"),
        "stdout": result.get("stdout", ""),
        "error": result.get("error"),
        "traceback": result.get("traceback"),
        "validation_details": result.get("validation_details"),
    }


TASK = register_task(Task(
    name="circle_packing",
    description=DESCRIPTION,
    initial_candidate=INITIAL_CANDIDATE,
    eval_fn=evaluate,
    metadata={
        "type": "single_task",
        "candidate_type": "code",
        "objective": "Maximize sum of circle radii for 26 circles in a unit square.",
    },
))
