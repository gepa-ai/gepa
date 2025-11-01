# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic

from gepa.core.adapter import RolloutOutput
from gepa.core.data_loader import DataId
from gepa.core.state import ProgramIdx

if TYPE_CHECKING:
    from gepa.core.state import GEPAState


@dataclass(frozen=True)
class GEPAResult(Generic[RolloutOutput, DataId]):
    """
    Immutable snapshot of a GEPA run with convenience accessors.

    - candidates: list of proposed candidates (component_name -> component_text)
    - parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
    - val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
    - val_subscores: per-candidate mapping from validation id to score on the validation set (sparse dict)
    - per_val_instance_best_candidates: for each val instance t, a set of candidate indices achieving the current best score on t
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
    val_subscores: list[dict[DataId, float]]
    per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
    discovery_eval_counts: list[int]

    # Optional data
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None

    # Run metadata (optional)
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 2

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

        return {
            "candidates": cands,
            "parents": self.parents,
            "val_aggregate_scores": self.val_aggregate_scores,
            "val_subscores": self.val_subscores,
            "best_outputs_valset": self.best_outputs_valset,
            "per_val_instance_best_candidates": {
                val_id: list(front) for val_id, front in self.per_val_instance_best_candidates.items()
            },
            "discovery_eval_counts": self.discovery_eval_counts,
            "total_metric_calls": self.total_metric_calls,
            "num_full_val_evals": self.num_full_val_evals,
            "run_dir": self.run_dir,
            "seed": self.seed,
            "best_idx": self.best_idx,
            "validation_schema_version": GEPAResult._VALIDATION_SCHEMA_VERSION,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GEPAResult[RolloutOutput, DataId]":
        version = d.get("validation_schema_version") or 0
        if version > GEPAResult._VALIDATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported GEPAResult validation schema version {version}; "
                f"max supported is {GEPAResult._VALIDATION_SCHEMA_VERSION}"
            )

        if version <= 1:
            return GEPAResult._migrate_from_dict_v0(d)

        return GEPAResult._from_dict_v2(d)

    @staticmethod
    def _common_kwargs_from_dict(d: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidates": [dict(candidate) for candidate in d.get("candidates", [])],
            "parents": [list(parent_row) for parent_row in d.get("parents", [])],
            "val_aggregate_scores": list(d.get("val_aggregate_scores", [])),
            "discovery_eval_counts": list(d.get("discovery_eval_counts", [])),
            "total_metric_calls": d.get("total_metric_calls"),
            "num_full_val_evals": d.get("num_full_val_evals"),
            "run_dir": d.get("run_dir"),
            "seed": d.get("seed"),
        }

    @staticmethod
    def _migrate_from_dict_v0(d: dict[str, Any]) -> "GEPAResult[RolloutOutput, DataId]":
        kwargs = GEPAResult._common_kwargs_from_dict(d)
        kwargs["val_subscores"] = [
            {idx: score for idx, score in enumerate(scores)} for scores in d.get("val_subscores", [])
        ]
        kwargs["per_val_instance_best_candidates"] = {
            idx: set(front) for idx, front in enumerate(d.get("per_val_instance_best_candidates", []))
        }

        best_outputs_valset = d.get("best_outputs_valset")
        if best_outputs_valset is not None:
            kwargs["best_outputs_valset"] = {
                idx: [(program_idx, output) for program_idx, output in outputs]
                for idx, outputs in enumerate(best_outputs_valset)
            }
        else:
            kwargs["best_outputs_valset"] = None
        return GEPAResult(**kwargs)

    @staticmethod
    def _from_dict_v2(d: dict[str, Any]) -> "GEPAResult[RolloutOutput, DataId]":
        kwargs = GEPAResult._common_kwargs_from_dict(d)
        kwargs["val_subscores"] = [dict(scores) for scores in d.get("val_subscores", [])]
        per_val_instance_best_candidates_data = d.get("per_val_instance_best_candidates", {})
        kwargs["per_val_instance_best_candidates"] = {
            val_id: set(candidates_on_front)
            for val_id, candidates_on_front in per_val_instance_best_candidates_data.items()
        }

        best_outputs_valset = d.get("best_outputs_valset")
        if best_outputs_valset is not None:
            kwargs["best_outputs_valset"] = {
                val_id: [(program_idx, output) for program_idx, output in outputs]
                for val_id, outputs in best_outputs_valset.items()
            }
        else:
            kwargs["best_outputs_valset"] = None

        return GEPAResult(**kwargs)

    @staticmethod
    def from_state(
        state: "GEPAState[RolloutOutput, DataId]",
        run_dir: str | None = None,
        seed: int | None = None,
    ) -> "GEPAResult[RolloutOutput, DataId]":
        """Build a GEPAResult from a GEPAState."""

        return GEPAResult(
            candidates=list(state.program_candidates),
            parents=list(state.parent_program_for_candidate),
            val_aggregate_scores=list(state.program_full_scores_val_set),
            best_outputs_valset=getattr(state, "best_outputs_valset", None),
            val_subscores=[dict(scores) for scores in state.prog_candidate_val_subscores],
            per_val_instance_best_candidates={
                val_id: set(front) for val_id, front in state.program_at_pareto_front_valset.items()
            },
            discovery_eval_counts=list(state.num_metric_calls_by_discovery),
            total_metric_calls=getattr(state, "total_num_evals", None),
            num_full_val_evals=getattr(state, "num_full_ds_evals", None),
            run_dir=run_dir,
            seed=seed,
        )
