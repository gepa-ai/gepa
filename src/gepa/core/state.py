# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from typing import Any, Callable, Generic

from gepa.core.adapter import RolloutOutput
from gepa.gepa_utils import idxmax, json_default


class GEPAState(Generic[RolloutOutput]):
    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[int | None]]

    program_full_scores_val_set: list[float]

    program_at_pareto_front_valset: list[set[int]]

    prog_candidate_val_subscores: list[list[float]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list

    per_program_tracked_scores: list[float]

    best_outputs_valset: list[tuple[int, list[RolloutOutput]]] | None = None

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: tuple[list[RolloutOutput], list[float]],
        track_best_outputs: bool = False,
    ):
        valset_base_score = sum(base_valset_eval_output[1]) / len(base_valset_eval_output[1])
        base_valset_pareto_front = list(base_valset_eval_output[1])

        self.program_candidates = [seed_candidate]
        self.program_full_scores_val_set = [valset_base_score]

        self.per_program_tracked_scores = [valset_base_score]

        self.pareto_front_valset = base_valset_pareto_front
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = [{0} for _ in range(len(base_valset_pareto_front))]

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.prog_candidate_val_subscores = [base_valset_eval_output[1]]
        self.num_metric_calls_by_discovery = [0]

        if track_best_outputs:
            # [(program_idx_1, output_1), (program_idx_2, output_2), ...]
            self.best_outputs_valset = [[(0, output)] for output in base_valset_eval_output[0]]

        self.full_program_trace = []

    def is_consistent(self):
        assert len(self.program_candidates) == len(self.program_full_scores_val_set)
        assert len(self.program_candidates) == len(self.per_program_tracked_scores)
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)

        assert len(self.prog_candidate_val_subscores) == len(self.program_candidates)
        assert len(self.pareto_front_valset) == len(self.program_at_pareto_front_valset)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        for prog_list in self.program_at_pareto_front_valset:
            for prog_idx in prog_list:
                assert prog_idx < len(self.program_candidates), "Program index in valset pareto front exceeds number of program candidates"

        return True

    def save(self, run_dir: str | None):
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            import pickle
            d = dict(self.__dict__.items())
            pickle.dump(d, f)

    @staticmethod
    def load(run_dir: str) -> "GEPAState":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle
            d = pickle.load(f)
        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)

        assert len(state.program_candidates) == len(state.program_full_scores_val_set)
        assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)

        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)

        return state

    def update_state_with_new_program(
        self,
        parent_program_idx: list[int],
        new_program: dict[str, str],
        valset_score: float,
        valset_outputs: Any,
        valset_subscores: list[float],
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int
    ):
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)
        # Find the highest predictor id from the parent programs
        max_predictor_id = max([self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx])
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
    valset_evaluator: Callable[[dict[str, str]], tuple[list[RolloutOutput], list[float]]],
    track_best_outputs: bool = False,
):
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
    else:
        num_evals_run = 0

        valset_out = valset_evaluator(seed_candidate)
        if run_dir is not None:
            write_eval_output_to_directory(valset_out, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(valset_out[1])

        gepa_state = GEPAState(
            seed_candidate,
            valset_out,
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
