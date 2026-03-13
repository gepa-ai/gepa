"""GEPA Adapter for Glean AL optimization.

This adapter bridges between GEPA's standard interface and AssistantALAdapter.
"""

from typing import Any, Mapping, Sequence

from gepa.adapters.glean_adapter.al_adapter import (
    AssistantALAdapter,
    ModuleSpec,
)
from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class GleanGEPAAdapter(GEPAAdapter[dict[str, Any], None, dict[str, Any]]):
    """Adapter that bridges GEPA's interface to AssistantALAdapter.

    GEPA expects:
    - evaluate(List[DataInst], dict[str, str]) -> EvaluationBatch[Trajectory, RolloutOutput]
    - DataInst can be any type
    - Type params: [DataInst, Trajectory, RolloutOutput]

    AssistantALAdapter expects:
    - evaluate(Candidate, List[Example]) -> CandidateEval

    This adapter converts between the two interfaces.
    """

    def __init__(
        self,
        al_adapter: AssistantALAdapter,
        model: str,
        module_specs: dict[str, ModuleSpec],
        global_token_cap: int,
        baseline_prompt_hash: str,
    ):
        self.al_adapter = al_adapter
        self.model = model
        self.module_specs = module_specs
        self.global_token_cap = global_token_cap
        self.baseline_prompt_hash = baseline_prompt_hash

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[None, dict[str, Any]]:
        """Evaluate candidate on batch (which is actually eval set metadata).

        In Glean's case:
        - batch contains eval set metadata (not individual queries)
        - We always evaluate on the full eval set via AssistantALAdapter

        Args:
            batch: List containing eval set metadata
            candidate: Flattened candidate dict[str, str]
            capture_traces: Whether to capture execution traces

        Returns:
            EvaluationBatch with outputs, scores, and no trajectories
            (EvolutionaryProposer handles its own reflection)
        """
        # Evaluate using simplified adapter interface
        cand_eval = self.al_adapter.evaluate(
            batch, candidate, capture_traces=capture_traces
        )

        # Convert CandidateEval to EvaluationBatch
        # For Glean, we have one "output" per eval set run
        outputs = [{"summary": cand_eval.summary}]
        scores = [cand_eval.summary.quality]  # Use quality as overall score
        # No trajectories - EvolutionaryProposer handles reflection internally

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[None, dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Create reflective dataset for component updates.

        Not used with EvolutionaryProposer - it handles its own reflection.
        """
        # Return empty reflective dataset - EvolutionaryProposer doesn't use this
        return {comp: [] for comp in components_to_update}
