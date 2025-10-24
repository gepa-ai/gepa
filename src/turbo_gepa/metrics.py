"""
Metrics collection for TurboGEPA dashboard.

Simple, lightweight metrics extracted from orchestrator state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


@dataclass
class Metrics:
    """Snapshot of orchestrator metrics at a point in time."""

    timestamp: float
    round: int
    evaluations: int
    best_quality: float
    avg_quality: float
    pareto_size: int
    qd_size: int
    total_candidates: int
    rung_activity: Dict[str, int]  # rung_key -> inflight count
    max_rounds: int | None = None
    max_evaluations: int | None = None


def extract_metrics(orchestrator: Orchestrator) -> Metrics:
    """
    Extract current metrics from orchestrator state.

    This is a pure read operation - no orchestrator state is modified.
    """
    # Get archive statistics
    pareto = orchestrator.archive.pareto_entries()
    qd_elites = orchestrator.archive.sample_qd(limit=1000)  # Get all QD elites
    # Total unique candidates = pareto + QD grid
    total = len(orchestrator.archive.pareto) + len(orchestrator.archive.qd_grid)

    # Calculate best and average quality from pareto frontier
    promote_objective = orchestrator.config.promote_objective
    if pareto:
        best_quality = max(
            e.result.objectives.get(promote_objective, 0.0) for e in pareto
        )
        avg_quality = sum(
            e.result.objectives.get(promote_objective, 0.0) for e in pareto
        ) / len(pareto)
    else:
        best_quality = 0.0
        avg_quality = 0.0

    # Get rung activity (inflight counts per rung)
    rung_activity = orchestrator._inflight_by_rung.copy()

    return Metrics(
        timestamp=time.time(),
        round=orchestrator.rounds_completed,
        evaluations=orchestrator.evaluations_run,
        best_quality=best_quality,
        avg_quality=avg_quality,
        pareto_size=len(pareto),
        qd_size=len(qd_elites),
        total_candidates=total,
        rung_activity=rung_activity,
        max_rounds=orchestrator.max_rounds,
        max_evaluations=orchestrator.max_evaluations,
    )
