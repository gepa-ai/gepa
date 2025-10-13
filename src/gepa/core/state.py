# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from collections import defaultdict
from typing import Any, Callable, ClassVar, Generic, Hashable, TypeAlias, TypeVar

from gepa.core.adapter import RolloutOutput
from gepa.gepa_utils import json_default

# Types for GEPAState
ProgramIdx = int
ValId = TypeVar("ValId", bound=Hashable)
"""Opaque identifier for valset examples"""
ValScores: TypeAlias = dict[ValId, float]
ValOutputs: TypeAlias = dict[ValId, RolloutOutput]
ObjectiveScores: TypeAlias = dict[str, float]


class GEPAState(Generic[RolloutOutput, ValId]):
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

    full_program_trace: list
    best_outputs_valset: dict[ValId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None

    validation_schema_version: int

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: tuple[ValOutputs, ValScores, dict[ValId, ObjectiveScores] | None],
        track_best_outputs: bool = False,
    ):
        base_outputs, base_scores, base_objective_scores = base_valset_eval_output
        self.program_candidates = [seed_candidate]
        self.prog_candidate_val_subscores = [base_scores]
        base_objective_aggregates = self._aggregate_objective_scores(base_objective_scores)
        self.prog_candidate_objective_scores = [base_objective_aggregates]

        self.pareto_front_valset = {val_id: score for val_id, score in base_scores.items()}
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in base_scores.keys()}
        self.objective_pareto_front = dict(base_objective_aggregates)
        self.program_at_pareto_front_objectives = {
            objective: {0} for objective in base_objective_aggregates.keys()
        }

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        self.best_outputs_valset = (
            {val_id: [(0, output)] for val_id, output in base_outputs.items()} if track_best_outputs else None
        )

        self.full_program_trace = []
        self.validation_schema_version = self._VALIDATION_SCHEMA_VERSION

    def is_consistent(self):
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        assert len(self.program_candidates) == len(self.prog_candidate_val_subscores)
        assert len(self.program_candidates) == len(self.prog_candidate_objective_scores)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        for front in self.program_at_pareto_front_valset.values():
            for prog_idx in front:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in valset pareto front exceeds number of program candidates"
                )

        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())
        assert set(self.objective_pareto_front.keys()) == set(self.program_at_pareto_front_objectives.keys())

        return True

    def save(self, run_dir: str | None):
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            import pickle

            d = dict(self.__dict__.items())
            d["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION
            pickle.dump(d, f)

    @staticmethod
    def load(run_dir: str) -> "GEPAState":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle

            d = pickle.load(f)

        version = d.get("validation_schema_version")
        if version is None:
            GEPAState._migrate_legacy_state_dict(d)

        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)

        state.validation_schema_version = GEPAState._VALIDATION_SCHEMA_VERSION
        assert set(state.pareto_front_valset.keys()) == set(state.program_at_pareto_front_valset.keys())
        assert len(state.program_candidates) == len(state.prog_candidate_val_subscores)
        assert len(state.program_candidates) == len(state.num_metric_calls_by_discovery)
        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        return state

    @staticmethod
    def _migrate_legacy_state_dict(d: dict[str, Any]) -> None:
        legacy_scores: list[list[float]] = d.pop("prog_candidate_val_subscores", [])
        # convert to sparse val subscores
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

        if "prog_candidate_objective_scores" not in d:
            num_candidates = len(d.get("program_candidates", []))
            d["prog_candidate_objective_scores"] = [{} for _ in range(num_candidates)]
        if "objective_pareto_front" not in d:
            d["objective_pareto_front"] = {}
        if "program_at_pareto_front_objectives" not in d:
            d["program_at_pareto_front_objectives"] = {}

        d["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION

    def get_program_average(self, program_idx: int) -> tuple[float, int]:
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        num_samples = len(scores)
        avg = sum(scores.values()) / num_samples
        return avg, num_samples

    @property
    def valset_evaluations(self) -> dict[ValId, list[ProgramIdx]]:
        """
        Valset examples by id and programs that have evaluated them. Keys consist of all known
        valset ids
        """
        result = defaultdict(list)
        for program_idx, val_scores in enumerate(self.prog_candidate_val_subscores):
            for val_id in val_scores.keys():
                result[val_id].append(program_idx)
        return result

    @property
    def program_full_scores_val_set(self) -> list[float]:
        return [self.get_program_average(program_idx)[0] for program_idx in range(len(self.prog_candidate_val_subscores))]

    @property
    def per_program_tracked_scores(self) -> list[float]:
        # NOTE(aria42): This same as valset program average scores, but this was already the case
        return [self.get_program_average(program_idx)[0] for program_idx in range(len(self.prog_candidate_val_subscores))]

    @staticmethod
    def _aggregate_objective_scores(
        val_objective_scores: dict[ValId, ObjectiveScores] | None,
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
            objective: totals[objective] / counts[objective]
            for objective in totals.keys()
            if counts[objective] > 0
        }

    def _update_objective_pareto_front(
        self, objective_scores: ObjectiveScores, program_idx: ProgramIdx
    ) -> None:
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
                    with open(os.path.join(task_dir, f"iter_{iteration}_prog_{program_idx}.json"), "w") as fout:
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
        valset_subscores: ValScores,
        valset_objective_subscores: dict[ValId, ObjectiveScores] | None,
        valset_outputs: ValOutputs | None,
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
    ) -> tuple[ProgramIdx, ProgramIdx]:
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)

        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(list(parent_program_idx))

        self.prog_candidate_val_subscores.append(dict(valset_subscores))
        objective_scores = self._aggregate_objective_scores(valset_objective_subscores)
        self.prog_candidate_objective_scores.append(objective_scores)

        for val_id, score in valset_subscores.items():
            self._update_pareto_front_for_val_id(val_id, score, new_program_idx, valset_outputs, run_dir, self.i + 1)

        self._update_objective_pareto_front(objective_scores, new_program_idx)

        linear_pareto_front_program_idx = self._best_program_idx()
        return new_program_idx, linear_pareto_front_program_idx

    def _best_program_idx(self) -> ProgramIdx:
        best_idx = 0
        best_avg, best_cov = self.get_program_average(0)
        for idx in range(1, len(self.program_candidates)):
            avg, cov = self.get_program_average(idx)
            if avg > best_avg or (avg == best_avg and cov > best_cov):
                best_idx = idx
                best_avg, best_cov = avg, cov
        return best_idx

    def get_pareto_front_mapping(self, pareto_frontier_type: str) -> dict[Any, set[ProgramIdx]]:
        if pareto_frontier_type == "instance":
            return {val_id: set(front) for val_id, front in self.program_at_pareto_front_valset.items()}
        if pareto_frontier_type == "objective":
            return {
                objective: set(front)
                for objective, front in self.program_at_pareto_front_objectives.items()
            }
        if pareto_frontier_type == "hybrid":
            combined: dict[Any, set[ProgramIdx]] = {
                val_id: set(front) for val_id, front in self.program_at_pareto_front_valset.items()
            }
            for objective, front in self.program_at_pareto_front_objectives.items():
                combined[f"objective::{objective}"] = set(front)
            return combined
        raise ValueError(f"Unknown pareto_frontier_type: {pareto_frontier_type}")


def write_eval_scores_to_directory(scores: ValScores, output_dir: str):
    for val_id, score in scores.items():
        task_dir = os.path.join(output_dir, f"task_{val_id}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(score, f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger,
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[[dict[str, str]], tuple[ValOutputs, ValScores]],
    track_best_outputs: bool = False,
):
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
    else:
        num_evals_run = 0

        seed_val_outputs, seed_val_scores, seed_objective_scores = valset_evaluator(seed_candidate)
        if run_dir is not None:
            write_eval_scores_to_directory(seed_val_scores, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(seed_val_scores)

        gepa_state = GEPAState(
            seed_candidate,
            (seed_val_outputs, seed_val_scores, seed_objective_scores),
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
