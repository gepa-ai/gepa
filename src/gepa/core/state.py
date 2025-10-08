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


class GEPAState(Generic[RolloutOutput, ValId]):
    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 2

    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]

    program_full_scores_val_set: list[float]

    pareto_front_valset: ValScores
    program_at_pareto_front_valset: dict[ValId, set[ProgramIdx]]

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
        base_valset_eval_output: tuple[ValOutputs, ValScores],
        track_best_outputs: bool = False,
    ):
        base_outputs, base_scores = base_valset_eval_output
        self.program_candidates = [seed_candidate]
        self.program_full_scores_val_set = [valset_base_score]

        self.pareto_front_valset = {val_id: score for val_id, score in base_scores.items()}
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in base_scores.keys()}

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

    def save(self, run_dir: str | None, *, use_cloudpickle: bool = False):
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            if use_cloudpickle:
                import cloudpickle as pickle
            else:
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

        assert len(state.program_candidates) == len(state.program_full_scores_val_set)
        assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)

        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        return state

    def update_state_with_new_program(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_subscores: ValScores,
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

        self.prog_candidate_val_subscores.append(valset_subscores)
        self.program_full_scores_val_set.append(valset_score)
        for task_idx, (old_score, new_score) in enumerate(zip(self.pareto_front_valset, valset_subscores, strict=False)):
            if new_score > old_score:
                self.pareto_front_valset[task_idx] = new_score
                self.program_at_pareto_front_valset[task_idx] = {new_program_idx}

                if self.best_outputs_valset is not None:
                    self.best_outputs_valset[task_idx] = [(new_program_idx, valset_outputs[task_idx])]

                if run_dir is not None:
                    os.makedirs(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}"), exist_ok=True)
                    with open(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}", f"iter_{self.i+1}_prog_{new_program_idx}.json"), "w") as f:
                        json.dump(valset_outputs[task_idx], f, indent=4, default=json_default)
            elif new_score == old_score:
                self.program_at_pareto_front_valset[task_idx].add(new_program_idx)
                if self.best_outputs_valset is not None:
                    self.best_outputs_valset[task_idx].append((new_program_idx, valset_outputs[task_idx]))

        assert len(valset_subscores) == len(self.program_at_pareto_front_valset)

        self.per_program_tracked_scores = self.program_full_scores_val_set

        linear_pareto_front_program_idx = idxmax(self.per_program_tracked_scores)

        return new_program_idx, linear_pareto_front_program_idx

def write_eval_output_to_directory(
    eval_out: tuple[list[RolloutOutput], list[float]],
    output_dir: str
):
    for task_idx, _score in enumerate(eval_out[1]):
        os.makedirs(os.path.join(output_dir, f"task_{task_idx}"), exist_ok=True)
        with open(os.path.join(output_dir, f"task_{task_idx}", f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(eval_out[1][task_idx], f, indent=4, default=json_default)

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

        seed_val_outputs, seed_val_scores = valset_evaluator(seed_candidate)
        if run_dir is not None:
            write_eval_output_to_directory(valset_out, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(valset_out[1])

        gepa_state = GEPAState(
            seed_candidate,
            (seed_val_outputs, seed_val_scores),
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
