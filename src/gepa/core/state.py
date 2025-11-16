# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar

from gepa.core.adapter import RolloutOutput as RolloutOutputType
from gepa.core.data_loader import DataId as DataIdType
from gepa.gepa_utils import json_default
from gepa.logging.logger import LoggerProtocol

# TypeVars for Generic classes
RolloutOutput = TypeVar("RolloutOutput")
ValId = TypeVar("ValId")

# Types for GEPAState
ProgramIdx = int
ValScores: TypeAlias = dict[ValId, float]
ValOutputs: TypeAlias = dict[ValId, RolloutOutput]
ObjectiveScores: TypeAlias = dict[str, float]
ValObjectiveScores: TypeAlias = dict[ValId, ObjectiveScores]


@dataclass(slots=True)
class ValsetEvaluation(Generic[RolloutOutput, ValId]):
    outputs_by_val_id: ValOutputs
    scores_by_val_id: ValScores
    objective_scores_by_val_id: ValObjectiveScores | None = None


class GEPAState(Generic[RolloutOutput, ValId]):
    """Persistent optimizer state tracking candidates, sparse validation coverage, and objective frontiers."""

    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 3

    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]
    prog_candidate_val_subscores: list[ValScores]
    prog_candidate_objective_scores: list[ObjectiveScores]

    pareto_front_valset: ValScores
    program_at_pareto_front_valset: dict[ValId, set[ProgramIdx]]
    objective_pareto_front: ObjectiveScores
    program_at_pareto_front_objectives: dict[str, set[ProgramIdx]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list[dict[str, Any]]
    best_outputs_valset: dict[ValId, list[tuple[ProgramIdx, RolloutOutput]]] | None

    validation_schema_version: int

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: (
            tuple[ValOutputs, ValScores] | tuple[ValOutputs, ValScores, dict[ValId, ObjectiveScores] | None]
        ),
        track_best_outputs: bool = False,
    ):
        base_evaluation = self._normalize_base_eval_output(base_valset_eval_output)

        self.program_candidates = [dict(seed_candidate)]
        self.prog_candidate_val_subscores = [dict(base_evaluation.scores_by_val_id)]

        base_objective_aggregates = self._aggregate_objective_scores(base_evaluation.objective_scores_by_val_id)
        self.prog_candidate_objective_scores = [base_objective_aggregates]

        self.pareto_front_valset = {val_id: score for val_id, score in base_evaluation.scores_by_val_id.items()}
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in base_evaluation.scores_by_val_id.keys()}
        self.objective_pareto_front = dict(base_objective_aggregates)
        self.program_at_pareto_front_objectives = {objective: {0} for objective in base_objective_aggregates.keys()}

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        if track_best_outputs:
            self.best_outputs_valset = {
                val_id: [(0, output)] for val_id, output in base_evaluation.outputs_by_val_id.items()
            }
        else:
            self.best_outputs_valset = None

        self.full_program_trace = []
        self.validation_schema_version = self._VALIDATION_SCHEMA_VERSION

    def is_consistent(self) -> bool:
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        assert len(self.program_candidates) == len(self.prog_candidate_val_subscores)
        assert len(self.program_candidates) == len(self.prog_candidate_objective_scores)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        assert len(self.pareto_front_valset) == len(self.program_at_pareto_front_valset)
        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())
        assert set(self.objective_pareto_front.keys()) == set(self.program_at_pareto_front_objectives.keys())

        for front in self.program_at_pareto_front_valset.values():
            for prog_idx in front:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in valset pareto front exceeds number of program candidates"
                )

        return True

    def save(self, run_dir: str | None, *, use_cloudpickle: bool = False) -> None:
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            if use_cloudpickle:
                import cloudpickle as pickle  # type: ignore[import-not-found]
            else:
                import pickle
            serialized = dict(self.__dict__.items())
            serialized["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION
            pickle.dump(serialized, f)

    @staticmethod
    def load(run_dir: str) -> "GEPAState[RolloutOutputType, DataIdType]":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle

            data = pickle.load(f)

        # handle schema migration
        version = data.get("validation_schema_version")
        if version is None or version < 2:
            GEPAState._migrate_from_legacy_state_v0(data)
            version = data.get("validation_schema_version")
        if version is None or version < GEPAState._VALIDATION_SCHEMA_VERSION:
            GEPAState._upgrade_state_dict(data)

        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(data)

        state.validation_schema_version = GEPAState._VALIDATION_SCHEMA_VERSION
        assert len(state.program_candidates) == len(state.prog_candidate_val_subscores)
        assert len(state.program_candidates) == len(state.prog_candidate_objective_scores)
        assert len(state.program_candidates) == len(state.num_metric_calls_by_discovery)
        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)
        assert set(state.pareto_front_valset.keys()) == set(state.program_at_pareto_front_valset.keys())
        assert set(state.objective_pareto_front.keys()) == set(state.program_at_pareto_front_objectives.keys())
        return state

    @staticmethod
    def _migrate_from_legacy_state_v0(d: dict[str, Any]) -> None:
        assert isinstance(d, dict)
        assert "prog_candidate_val_subscores" in d
        assert isinstance(d["prog_candidate_val_subscores"], list)
        assert all(isinstance(scores, list) for scores in d["prog_candidate_val_subscores"])
        legacy_scores: list[list[float]] = d.pop("prog_candidate_val_subscores", [])
        d["prog_candidate_val_subscores"] = [
            {idx: score for idx, score in enumerate(scores)} for scores in legacy_scores
        ]

        pareto_front = d.get("pareto_front_valset")
        if isinstance(pareto_front, list):
            d["pareto_front_valset"] = {idx: score for idx, score in enumerate(pareto_front)}

        program_at_front = d.get("program_at_pareto_front_valset")
        if isinstance(program_at_front, list):
            d["program_at_pareto_front_valset"] = {idx: set(front) for idx, front in enumerate(program_at_front)}

        best_outputs = d.get("best_outputs_valset")
        if isinstance(best_outputs, list):
            d["best_outputs_valset"] = {idx: list(outputs) for idx, outputs in enumerate(best_outputs)}

        d["validation_schema_version"] = 2

    @staticmethod
    def _upgrade_state_dict(d: dict[str, Any]) -> None:
        num_candidates = len(d.get("program_candidates", []))
        if "prog_candidate_objective_scores" not in d:
            d["prog_candidate_objective_scores"] = [{} for _ in range(num_candidates)]
        if "objective_pareto_front" not in d:
            d["objective_pareto_front"] = {}
        if "program_at_pareto_front_objectives" not in d:
            d["program_at_pareto_front_objectives"] = {}
        d["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION

    @staticmethod
    def _normalize_base_eval_output(
        base_valset_eval_output: (
            tuple[ValOutputs, ValScores] | tuple[ValOutputs, ValScores, dict[ValId, ObjectiveScores] | None]
        ),
    ) -> ValsetEvaluation:
        if len(base_valset_eval_output) == 3:
            outputs, scores, objective_scores = base_valset_eval_output
            return ValsetEvaluation(
                outputs_by_val_id=dict(outputs),
                scores_by_val_id=dict(scores),
                objective_scores_by_val_id=(
                    dict((val_id, dict(obj_scores)) for val_id, obj_scores in objective_scores.items())
                    if objective_scores is not None
                    else None
                ),
            )
        if len(base_valset_eval_output) == 2:
            outputs, scores = base_valset_eval_output
            return ValsetEvaluation(
                outputs_by_val_id=dict(outputs),
                scores_by_val_id=dict(scores),
                objective_scores_by_val_id=None,
            )
        raise ValueError("Unexpected base_valset_eval_output tuple size")

    @staticmethod
    def _aggregate_objective_scores(
        val_objective_scores: ValObjectiveScores | None,
    ) -> ObjectiveScores:
        if not val_objective_scores:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for objective_dict in val_objective_scores.values():
            for objective, score in objective_dict.items():
                totals[objective] = totals.get(objective, 0.0) + score
                counts[objective] = counts.get(objective, 0) + 1
        return {
            objective: totals[objective] / counts[objective] for objective in totals.keys() if counts[objective] > 0
        }

    def get_program_average_val_subset(self, program_idx: int) -> tuple[float, int]:
        # TODO: This should be only used/handled by the val_evaluation_policy, and never used directly.
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        num_samples = len(scores)
        avg = sum(scores.values()) / num_samples
        return avg, num_samples

    # Backwards-compatible alias for legacy callers.
    def get_program_average(self, program_idx: int) -> tuple[float, int]:
        return self.get_program_average_val_subset(program_idx)

    @property
    def valset_evaluations(self) -> dict[ValId, list[ProgramIdx]]:
        """
        Valset examples by id and programs that have evaluated them. Keys include only validation
        ids that have been scored at least once.
        """
        result: dict[ValId, list[ProgramIdx]] = defaultdict(list)
        for program_idx, val_scores in enumerate(self.prog_candidate_val_subscores):
            for val_id in val_scores.keys():
                result[val_id].append(program_idx)
        return result

    @property
    def program_full_scores_val_set(self) -> list[float]:
        # TODO: This should be using the val_evaluation_policy instead of the get_program_average_val_subset method to calculate the scores.
        return [
            self.get_program_average_val_subset(program_idx)[0]
            for program_idx in range(len(self.prog_candidate_val_subscores))
        ]

    @property
    def per_program_tracked_scores(self) -> list[float]:
        return [
            self.get_program_average_val_subset(program_idx)[0]
            for program_idx in range(len(self.prog_candidate_val_subscores))
        ]

    def _update_objective_pareto_front(self, objective_scores: ObjectiveScores, program_idx: ProgramIdx) -> None:
        if not objective_scores:
            return
        for objective, score in objective_scores.items():
            prev_score = self.objective_pareto_front.get(objective, float("-inf"))
            if score > prev_score:
                self.objective_pareto_front[objective] = score
                self.program_at_pareto_front_objectives[objective] = {program_idx}
            elif score == prev_score:
                front = self.program_at_pareto_front_objectives.setdefault(objective, set())
                front.add(program_idx)

    def _update_pareto_front_for_val_id(
        self,
        val_id: ValId,
        score: float,
        program_idx: ProgramIdx,
        outputs: ValOutputs | None,
        run_dir: str | None,
        iteration: int,
    ) -> None:
        prev_score = self.pareto_front_valset.get(val_id, float("-inf"))
        if score > prev_score:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}
            output = outputs.get(val_id) if outputs is not None else None
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]
                if run_dir is not None:
                    task_dir = os.path.join(run_dir, "generated_best_outputs_valset", f"task_{val_id}")
                    os.makedirs(task_dir, exist_ok=True)
                    with open(
                        os.path.join(task_dir, f"iter_{iteration}_prog_{program_idx}.json"),
                        "w",
                    ) as fout:
                        json.dump(output, fout, indent=4, default=json_default)
        elif score == prev_score:
            pareto_front = self.program_at_pareto_front_valset.setdefault(val_id, set())
            pareto_front.add(program_idx)
            output = outputs.get(val_id) if outputs is not None else None
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id].append((program_idx, output))

    def update_state_with_new_program(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_evaluation: ValsetEvaluation,
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
    ) -> ProgramIdx:
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(dict(new_program))
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)

        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(list(parent_program_idx))

        valset_scores = dict(valset_evaluation.scores_by_val_id)
        self.prog_candidate_val_subscores.append(valset_scores)
        objective_scores = self._aggregate_objective_scores(valset_evaluation.objective_scores_by_val_id)
        self.prog_candidate_objective_scores.append(objective_scores)

        for val_id, score in valset_scores.items():
            self._update_pareto_front_for_val_id(
                val_id,
                score,
                new_program_idx,
                valset_evaluation.outputs_by_val_id,
                run_dir,
                self.i + 1,
            )

        self._update_objective_pareto_front(objective_scores, new_program_idx)

        return new_program_idx

    def get_pareto_front_mapping(self, frontier_type: str) -> dict[Any, set[ProgramIdx]]:
        if frontier_type == "instance":
            return {val_id: set(front) for val_id, front in self.program_at_pareto_front_valset.items()}
        if frontier_type == "objective":
            return {objective: set(front) for objective, front in self.program_at_pareto_front_objectives.items()}
        if frontier_type == "hybrid":
            combined: dict[Any, set[ProgramIdx]] = {
                val_id: set(front) for val_id, front in self.program_at_pareto_front_valset.items()
            }
            for objective, front in self.program_at_pareto_front_objectives.items():
                combined[f"objective::{objective}"] = set(front)
            return combined
        raise ValueError(f"Unknown frontier_type: {frontier_type}")


def write_eval_scores_to_directory(scores: ValScores, output_dir: str) -> None:
    for val_id, score in scores.items():
        task_dir = os.path.join(output_dir, f"task_{val_id}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(score, f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger: LoggerProtocol,
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[
        [dict[str, str]],
        tuple[ValOutputs, ValScores] | tuple[ValOutputs, ValScores, dict[ValId, ObjectiveScores] | None],
    ],
    track_best_outputs: bool = False,
) -> GEPAState[RolloutOutputType, DataIdType]:
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
    else:
        num_evals_run = 0

        eval_result = valset_evaluator(seed_candidate)
        if len(eval_result) == 3:
            seed_val_outputs, seed_val_scores, seed_objective_scores = eval_result
        elif len(eval_result) == 2:
            seed_val_outputs, seed_val_scores = eval_result
            seed_objective_scores = None
        else:
            raise ValueError("Unexpected valset_evaluator return value length")

        seed_valset_evaluation = GEPAState._normalize_base_eval_output(
            (seed_val_outputs, seed_val_scores, seed_objective_scores)
        )

        if run_dir is not None:
            write_eval_scores_to_directory(
                seed_valset_evaluation.scores_by_val_id,
                os.path.join(run_dir, "generated_best_outputs_valset"),
            )
        num_evals_run += len(seed_valset_evaluation.scores_by_val_id)

        gepa_state = GEPAState(
            seed_candidate,
            (
                seed_valset_evaluation.outputs_by_val_id,
                seed_valset_evaluation.scores_by_val_id,
                seed_valset_evaluation.objective_scores_by_val_id,
            ),
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
