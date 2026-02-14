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
    """Immutable snapshot returned by :func:`~gepa.optimize_anything.optimize_anything`.

    Key attributes:
        best_candidate: The optimized parameter(s) — ``dict[str, str]`` or plain
            ``str`` when ``seed_candidate`` was a string.
        best_idx: Index of the highest-scoring candidate.
        val_aggregate_scores: Per-candidate average validation score (higher is better).
        candidates: All candidates explored during optimization.
        parents: Lineage — ``parents[i]`` is a list of parent indices for candidate ``i``.
        per_val_instance_best_candidates: Pareto frontier — per validation example,
            the set of candidate indices achieving the best score.
        best_refiner_prompt: The refiner prompt from the best candidate (if refiner was enabled).

    Serialization:
        ``to_dict()`` / ``from_dict()`` for JSON-safe round-tripping.

    Example::

        result = optimize_anything(...)
        print(result.best_candidate)
        print(result.val_aggregate_scores[result.best_idx])
    """

    # Core data
    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[DataId, float]]
    per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
    discovery_eval_counts: list[int]
    val_aggregate_subscores: list[dict[str, float]] | None = None
    per_objective_best_candidates: dict[str, set[ProgramIdx]] | None = None
    objective_pareto_front: dict[str, float] | None = None

    # Optional data
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None

    # Run metadata (optional)
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    # When set, best_candidate unwraps the dict to return a plain str.
    # This is the internal dict key used to wrap str seed_candidates.
    _str_candidate_key: str | None = None

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
    def best_candidate(self) -> str | dict[str, str]:
        """Return the best candidate.

        When ``optimize_anything`` was called with a ``str`` seed_candidate,
        returns the plain ``str`` value.  Otherwise returns the full
        ``dict[str, str]`` parameter mapping.
        """
        cand = self.candidates[self.best_idx]
        if self._str_candidate_key is not None and self._str_candidate_key in cand:
            return cand[self._str_candidate_key]
        return cand

    @property
    def best_refiner_prompt(self) -> str | None:
        """Return the refiner prompt from the best candidate, or ``None`` if
        the refiner was not enabled."""
        return self.candidates[self.best_idx].get("refiner_prompt")

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
            "val_aggregate_subscores": self.val_aggregate_subscores,
            "per_objective_best_candidates": (
                {k: list(v) for k, v in self.per_objective_best_candidates.items()}
                if self.per_objective_best_candidates is not None
                else None
            ),
            "objective_pareto_front": self.objective_pareto_front,
            "discovery_eval_counts": self.discovery_eval_counts,
            "total_metric_calls": self.total_metric_calls,
            "num_full_val_evals": self.num_full_val_evals,
            "run_dir": self.run_dir,
            "seed": self.seed,
            "_str_candidate_key": self._str_candidate_key,
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
            "_str_candidate_key": d.get("_str_candidate_key"),
        }

    @staticmethod
    def _migrate_from_dict_v0(d: dict[str, Any]) -> "GEPAResult[RolloutOutput, DataId]":
        kwargs = GEPAResult._common_kwargs_from_dict(d)
        kwargs["val_subscores"] = [dict(enumerate(scores)) for scores in d.get("val_subscores", [])]
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

        val_aggregate_subscores = d.get("val_aggregate_subscores")
        kwargs["val_aggregate_subscores"] = (
            [dict(scores) for scores in val_aggregate_subscores] if val_aggregate_subscores is not None else None
        )

        per_objective_best_candidates = d.get("per_objective_best_candidates")
        if per_objective_best_candidates is not None:
            kwargs["per_objective_best_candidates"] = {
                objective: set(program_indices) for objective, program_indices in per_objective_best_candidates.items()
            }
        else:
            kwargs["per_objective_best_candidates"] = None

        objective_pareto_front = d.get("objective_pareto_front")
        kwargs["objective_pareto_front"] = dict(objective_pareto_front) if objective_pareto_front is not None else None

        return GEPAResult(**kwargs)

    @staticmethod
    def from_state(
        state: "GEPAState[RolloutOutput, DataId]",
        run_dir: str | None = None,
        seed: int | None = None,
        str_candidate_key: str | None = None,
    ) -> "GEPAResult[RolloutOutput, DataId]":
        """Build a GEPAResult from a GEPAState.

        Args:
            str_candidate_key: When set, ``best_candidate`` unwraps the internal
                dict to return the plain ``str`` value stored under this key.
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
            _str_candidate_key=str_candidate_key,
        )
