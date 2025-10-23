# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass
from typing import Any, Generic

from gepa.core.adapter import RolloutOutput
from gepa.core.state import ProgramIdx, ValId, ValScores


@dataclass(frozen=True)
class GEPAResult(Generic[RolloutOutput]):
    """
    Immutable snapshot of a GEPA run with convenience accessors.

    - candidates: list of proposed candidates (component_name -> component_text)
    - parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
    - val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
    - val_subscores: per-candidate per-instance scores on the validation set (len == num_val_instances)
    - val_aggregate_subscores: optional per-candidate aggregate subscores across objectives
    - per_val_instance_best_candidates: for each val instance t, a set of candidate indices achieving the current best score on t
    - per_objective_best_candidates: optional per-objective set of candidate indices achieving best aggregate subscore
    - discovery_eval_counts: number of metric calls accumulated up to the discovery of each candidate

    Optional fields:
    - best_outputs_valset: per-task best outputs on the validation set. [task_idx -> [(program_idx_1, output_1), (program_idx_2, output_2), ...]]

    Run-level metadata:
    - total_metric_calls: total number of metric calls made across the run
    - num_full_val_evals: number of full validation evaluations performed
    - run_dir: where artifacts were written (if any)
    - seed: RNG seed for reproducibility (if known)
    - tracked_scores: optional tracked aggregate scores (if different from val_aggregate_scores)

    Convenience:
    - best_idx: candidate index with the highest val_aggregate_scores
    - best_candidate: the program text mapping for best_idx
    - non_dominated_indices(): candidate indices that are not dominated across per-instance pareto fronts
    - lineage(idx): parent chain from base to idx
    - diff(parent_idx, child_idx, only_changed=True): component-wise diff between two candidates
    - best_k(k): top-k candidates by aggregate val score
    - instance_winners(t): set of candidates on the pareto front for val instance t
    - to_dict(...), save_json(...): serialization helpers
    """

    # Core data
    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[ValScores]
    per_val_instance_best_candidates: dict[ValId, set[ProgramIdx]]
    discovery_eval_counts: list[int]
    val_aggregate_subscores: list[dict[str, float]] | None = None
    per_objective_best_candidates: dict[str, set[ProgramIdx]] | None = None
    objective_pareto_front: dict[str, float] | None = None

    # Optional data
    best_outputs_valset: dict[ValId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None

    # Run metadata (optional)
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    # -------- Convenience properties --------
    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def num_val_instances(self) -> int:
        return len(self.per_val_instance_best_candidates)

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda i: scores[i])

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]

    def to_dict(self) -> dict[str, Any]:
        cands = [dict(cand.items()) for cand in self.candidates]

        return dict(
            candidates=cands,
            parents=self.parents,
            val_aggregate_scores=self.val_aggregate_scores,
            val_subscores=self.val_subscores,
            best_outputs_valset=self.best_outputs_valset,
            per_val_instance_best_candidates={k: list(v) for k, v in self.per_val_instance_best_candidates.items()},
            val_aggregate_subscores=self.val_aggregate_subscores,
            per_objective_best_candidates=(
                {k: list(v) for k, v in self.per_objective_best_candidates.items()}
                if self.per_objective_best_candidates is not None
                else None
            ),
            objective_pareto_front=self.objective_pareto_front,
            discovery_eval_counts=self.discovery_eval_counts,
            total_metric_calls=self.total_metric_calls,
            num_full_val_evals=self.num_full_val_evals,
            run_dir=self.run_dir,
            seed=self.seed,
            best_idx=self.best_idx,
        )

    @staticmethod
    def from_state(state: Any, run_dir: str | None = None, seed: int | None = None) -> "GEPAResult":
        """
        Build a GEPAResult from a GEPAState.
        """
        objective_scores_list = [dict(scores) for scores in state.prog_candidate_objective_scores]
        has_objective_scores = any(obj for obj in objective_scores_list)
        per_objective_best = {
            objective: set(front) for objective, front in state.program_at_pareto_front_objectives.items()
        }
        objective_front = dict(state.objective_pareto_front)

        return GEPAResult(
            candidates=list(state.program_candidates),
            parents=list(state.parent_program_for_candidate),
            val_aggregate_scores=list(state.program_full_scores_val_set),
            best_outputs_valset=getattr(state, "best_outputs_valset", None),
            val_subscores=[dict(scores) for scores in state.prog_candidate_val_subscores],
            per_val_instance_best_candidates={
                val_id: set(front) for val_id, front in state.program_at_pareto_front_valset.items()
            },
            val_aggregate_subscores=(objective_scores_list if has_objective_scores else None),
            per_objective_best_candidates=(per_objective_best if per_objective_best else None),
            objective_pareto_front=objective_front if objective_front else None,
            discovery_eval_counts=list(state.num_metric_calls_by_discovery),
            total_metric_calls=getattr(state, "total_num_evals", None),
            num_full_val_evals=getattr(state, "num_full_ds_evals", None),
            run_dir=run_dir,
            seed=seed,
        )
