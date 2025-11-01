# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any, ClassVar, Generic

from gepa.core.adapter import RolloutOutput
from gepa.core.data_loader import DataId
from gepa.gepa_utils import json_default
from gepa.logging.logger import LoggerProtocol

# Types for GEPAState
ProgramIdx = int
"""Opaque identifier for program candidates."""


class GEPAState(Generic[RolloutOutput, DataId]):
    """Persistent optimizer state tracking candidates, sparse validation coverage, and execution metadata."""

    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 2

    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]
    prog_candidate_val_subscores: list[dict[DataId, float]]

    pareto_front_valset: dict[DataId, float]
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list[dict[str, Any]]
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None

    validation_schema_version: int

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: tuple[dict[DataId, RolloutOutput], dict[DataId, float]],
        track_best_outputs: bool = False,
    ):
        base_outputs, base_scores = base_valset_eval_output
        self.program_candidates = [seed_candidate]
        self.prog_candidate_val_subscores = [base_scores]

        self.pareto_front_valset = {val_id: score for val_id, score in base_scores.items()}
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in base_scores.keys()}

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        if track_best_outputs:
            self.best_outputs_valset = {
                val_id: [(0, output)] for val_id, output in base_outputs.items()
            }
        else:
            self.best_outputs_valset = None

        self.full_program_trace = []
        self.validation_schema_version = self._VALIDATION_SCHEMA_VERSION

    def is_consistent(self) -> bool:
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)

        assert len(self.prog_candidate_val_subscores) == len(self.program_candidates)
        assert len(self.pareto_front_valset) == len(self.program_at_pareto_front_valset)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        for front in self.program_at_pareto_front_valset.values():
            for prog_idx in front:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in valset pareto front exceeds number of program candidates"
                )

        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())

        return True

    def save(self, run_dir: str | None, *, use_cloudpickle: bool = False) -> None:
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            if use_cloudpickle:
                import cloudpickle as pickle  # pragma: no cover - optional dependency
            else:
                import pickle
            serialized = dict(self.__dict__.items())
            serialized["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION
            pickle.dump(serialized, f)

    @staticmethod
    def load(run_dir: str) -> "GEPAState[RolloutOutput, DataId]":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle

            data = pickle.load(f)

        # handle schema migration
        version = data.get("validation_schema_version")
        if version is None or version == 1:
            GEPAState._migrate_from_legacy_state_v0(data)

        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(data)

        assert len(state.program_candidates) == len(state.program_full_scores_val_set)
        assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)

        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        return state

    @staticmethod
    def _migrate_from_legacy_state_v0(d: dict[str, Any]) -> None:
        assert isinstance(d, dict)
        assert "prog_candidate_val_subscores" in d
        assert isinstance(d["prog_candidate_val_subscores"], list)
        assert all(isinstance(scores, list) for scores in d["prog_candidate_val_subscores"])
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

        d["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION

    def get_program_average_val_subset(self, program_idx: int) -> tuple[float, int]:
        # TODO: This should be only used/handled by the val_evaluation_policy, and never used directly.
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        num_samples = len(scores)
        avg = sum(scores.values()) / num_samples
        return avg, num_samples

    @property
    def valset_evaluations(self) -> dict[DataId, list[ProgramIdx]]:
        """
        Valset examples by id and programs that have evaluated them. Keys include only validation
        ids that have been scored at least once.
        """
        result: dict[DataId, list[ProgramIdx]] = defaultdict(list)
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

    def _update_pareto_front_for_val_id(
        self,
        val_id: DataId,
        score: float,
        program_idx: ProgramIdx,
        output: RolloutOutput | None,
        run_dir: str | None,
        iteration: int,
    ) -> None:
        prev_score = self.pareto_front_valset.get(val_id, float("-inf"))
        if score > prev_score:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]
                if run_dir is not None:
                    task_dir = os.path.join(run_dir, "generated_best_outputs_valset", f"task_{val_id}")
                    os.makedirs(task_dir, exist_ok=True)
                    with open(os.path.join(task_dir, f"iter_{iteration}_prog_{program_idx}.json"), "w") as fout:
                        json.dump(output, fout, indent=4, default=json_default)
        elif score == prev_score:
            assert self.program_at_pareto_front_valset.get(val_id), (
                f"Program at pareto front for val_id {val_id} should be non-empty"
            )
            pareto_front = self.program_at_pareto_front_valset[val_id]
            pareto_front.add(program_idx)
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id].append((program_idx, output))

    def update_state_with_new_program(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_subscores: dict[DataId, float],
        valset_outputs: dict[DataId, RolloutOutput] | None,
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
    ) -> ProgramIdx:
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)

        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(list(parent_program_idx))

        self.prog_candidate_val_subscores.append(valset_subscores)
        for val_id, score in valset_subscores.items():
            valset_output = valset_outputs.get(val_id) if valset_outputs else None
            self._update_pareto_front_for_val_id(val_id, score, new_program_idx, valset_output, run_dir, self.i + 1)
        return new_program_idx


def write_eval_scores_to_directory(scores: dict[DataId, float], output_dir: str) -> None:
    for val_id, score in scores.items():
        task_dir = os.path.join(output_dir, f"task_{val_id}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(score, f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger: LoggerProtocol,
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[[dict[str, str]], tuple[dict[DataId, RolloutOutput], dict[DataId, float]]],
    track_best_outputs: bool = False,
) -> GEPAState[RolloutOutput, DataId]:
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
    else:
        num_evals_run = 0

        seed_val_outputs, seed_val_scores = valset_evaluator(seed_candidate)
        if run_dir is not None:
            write_eval_scores_to_directory(seed_val_scores, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(seed_val_scores)

        gepa_state = GEPAState(
            seed_candidate,
            (seed_val_outputs, seed_val_scores),
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
