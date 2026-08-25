"""Glean-specific integration with GEPA's low-level optimization engine."""

from __future__ import annotations

from typing import Any, cast

from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import FrontierType
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.strategies.eval_policy import EvaluationPolicy
from gepa.utils import MaxMetricCallsStopper
from glean_gepa.adapter_types import SingleModelALDataInst, TeacherStudentALDataInst
from glean_gepa.evolutionary_proposer import EvolutionaryProposer
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter


def optimize(
    *,
    seed_candidate: dict[str, str],
    trainset: list[SingleModelALDataInst] | list[TeacherStudentALDataInst],
    valset: list[SingleModelALDataInst] | list[TeacherStudentALDataInst],
    adapter: SingleModelAdapter | TeacherStudentAdapter,
    proposer: EvolutionaryProposer,
    logger: LoggerProtocol,
    experiment_tracker: ExperimentTracker,
    max_metric_calls: int,
    run_dir: str | None,
    frontier_type: FrontierType,
    val_evaluation_policy: EvaluationPolicy | None = None,
) -> GEPAResult:
    """Run the Glean proposer while keeping custom wiring outside ``gepa.api``."""
    del trainset  # The proposer owns its training loader; the engine only needs validation data.
    engine = GEPAEngine(
        adapter=cast(Any, adapter),
        run_dir=run_dir,
        valset=cast(Any, valset),
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=0,
        reflective_proposer=proposer,  # type: ignore[arg-type]
        merge_proposer=None,
        frontier_type=frontier_type,
        logger=logger,
        experiment_tracker=experiment_tracker,
        stop_callback=MaxMetricCallsStopper(max_metric_calls),
        val_evaluation_policy=val_evaluation_policy,
    )

    with experiment_tracker:
        state = engine.run()
    return GEPAResult.from_state(state, run_dir=run_dir, seed=0)


__all__ = ["optimize"]
