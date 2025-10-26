"""
Metrics collection for TurboGEPA dashboard.

Simple, lightweight metrics extracted from orchestrator state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Any

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
    mutations_requested: int = 0
    mutations_generated: int = 0
    mutations_enqueued: int = 0
    mutations_promoted: int = 0
    unique_parents: int = 0
    unique_children: int = 0
    evolution_edges: int = 0
    lineage_data: List[Dict[str, Any]] = field(default_factory=list)  # List of {fingerprint, generation, quality, status}


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

    evo = orchestrator.evolution_snapshot()

    # Get lineage data for visualization
    lineage_data = orchestrator.get_candidate_lineage_data()

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
        mutations_requested=evo["mutations_requested"],
        mutations_generated=evo["mutations_generated"],
        mutations_enqueued=evo["mutations_enqueued"],
        mutations_promoted=evo["mutations_promoted"],
        unique_parents=evo["unique_parents"],
        unique_children=evo["unique_children"],
        evolution_edges=evo["evolution_edges"],
        lineage_data=lineage_data,
    )
