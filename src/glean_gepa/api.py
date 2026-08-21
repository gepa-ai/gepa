"""Glean-specific integration with GEPA's low-level optimization engine."""

from __future__ import annotations

from typing import Any

from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import FrontierType
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.utils import MaxMetricCallsStopper
from glean_gepa.al_adapter import ALDataInst, AssistantALAdapter
from glean_gepa.evolutionary_proposer import EvolutionaryProposer


class _EngineProposerBridge:
    """Normalize Glean proposals across GEPA engine revisions."""

    def __init__(self, proposer: EvolutionaryProposer):
        self._proposer = proposer
        self.trainset = proposer.trainset

    def propose(self, state: Any) -> Any:
        proposal = self._proposer.propose(state)
        if hasattr(GEPAEngine, "_run_reflective_batch"):
            return [proposal] if proposal is not None else []
        return proposal


def optimize(
    *,
    seed_candidate: dict[str, str],
    trainset: list[ALDataInst],
    valset: list[ALDataInst],
    adapter: AssistantALAdapter,
    proposer: EvolutionaryProposer,
    logger: LoggerProtocol,
    experiment_tracker: ExperimentTracker,
    max_metric_calls: int,
    run_dir: str | None,
    frontier_type: FrontierType,
) -> GEPAResult:
    """Run the Glean proposer while keeping custom wiring outside ``gepa.api``."""
    del trainset  # The proposer owns its training loader; the engine only needs validation data.
    engine = GEPAEngine(
        adapter=adapter,
        run_dir=run_dir,
        valset=valset,
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=0,
        reflective_proposer=_EngineProposerBridge(proposer),  # type: ignore[arg-type]
        merge_proposer=None,
        frontier_type=frontier_type,
        logger=logger,
        experiment_tracker=experiment_tracker,
        stop_callback=MaxMetricCallsStopper(max_metric_calls),
    )

    with experiment_tracker:
        state = engine.run()
    return GEPAResult.from_state(state, run_dir=run_dir, seed=0)


__all__ = ["optimize"]
