# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import traceback
from typing import Any, Callable, Generic, Sequence

from gepa.core.adapter import DataInst, EvaluationBatch, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState, ValsetEvaluation, initialize_gepa_state
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import (
    ReflectiveMutationProposer,
)
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """
    Orchestrates the optimization loop. It uses pluggable ProposeNewCandidate strategies.
    """

    def __init__(
        self,
        run_dir: str | None,
        evaluator: Callable[
            [list[DataInst], dict[str, str]],
            EvaluationBatch[Trajectory, RolloutOutput]
            | tuple[list[RolloutOutput], list[float]]
            | tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None],
        ],
        valset: list[DataInst] | DataLoader[DataId, DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        frontier_type: str,
        # Logging
        logger: Any,
        experiment_tracker: Any,
        # Optional parameters
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        # Budget and Stop Condition
        stop_callback: Callable[[Any], bool] | None = None,
        val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | None = None,
    ):
        self.logger = logger
        self.run_dir = run_dir

        # Graceful stopping mechanism
        self._stop_requested = False

        # Set up stopping mechanism
        self.stop_callback = stop_callback
        self.evaluator = evaluator
        self.valset = ensure_loader(valset)
        self.seed_candidate = seed_candidate

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer
        self.frontier_type = frontier_type

        # Merge scheduling flags (mirroring previous behavior)
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar
        self.use_cloudpickle = use_cloudpickle

        self.raise_on_exception = raise_on_exception
        self.val_evaluation_policy = val_evaluation_policy or FullEvaluationPolicy()

    @staticmethod
    def _coerce_evaluation_result(
        evaluation_result: (
            EvaluationBatch[Trajectory, RolloutOutput]
            | tuple[list[RolloutOutput], list[float]]
            | tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]
        ),
    ) -> tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]:
        if isinstance(evaluation_result, EvaluationBatch):
            return (
                evaluation_result.outputs,
                evaluation_result.scores,
                evaluation_result.objective_scores,
            )
        if len(evaluation_result) == 3:
            outputs, scores, objective_scores = evaluation_result
            return outputs, scores, objective_scores
        if len(evaluation_result) == 2:
            outputs, scores = evaluation_result
            return outputs, scores, None
        raise ValueError("Unexpected evaluator return signature")

    def _evaluate_on_valset(self, program: dict[str, str], state: GEPAState) -> ValsetEvaluation[RolloutOutput, DataId]:
        assert self.valset is not None

        val_ids = self.val_evaluation_policy.get_eval_batch(self.valset, state)
        batch = self.valset.fetch(val_ids)
        outputs, scores, objective_scores = self._coerce_evaluation_result(self.evaluator(batch, program))
        assert len(outputs) == len(val_ids), "Eval outputs should match length of selected validation indices"
        assert len(scores) == len(val_ids), "Eval scores should match length of selected validation indices"

        outputs_by_val_idx = dict(zip(val_ids, outputs, strict=False))
        scores_by_val_idx = dict(zip(val_ids, scores, strict=False))
        objective_by_val_idx = (
            dict(zip(val_ids, objective_scores, strict=False)) if objective_scores is not None else None
        )
        return ValsetEvaluation(
            outputs_by_val_id=outputs_by_val_idx,
            scores_by_val_id=scores_by_val_idx,
            objective_scores_by_val_id=objective_by_val_idx,
        )

    def _get_pareto_front_programs(self, state: GEPAState) -> dict[Any, set[int]]:
        return state.get_pareto_front_mapping(self.frontier_type)

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState,
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        valset_evaluation = self._evaluate_on_valset(new_program, state)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_evaluation.scores_by_val_id)

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_evaluation=valset_evaluation,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )
        state.full_program_trace[-1]["new_program_idx"] = new_program_idx
        state.full_program_trace[-1]["evaluated_val_indices"] = sorted(valset_evaluation.scores_by_val_id.keys())

        valset_score = self.val_evaluation_policy.get_valset_score(new_program_idx, state)

        linear_pareto_front_program_idx = self.val_evaluation_policy.get_best_program(state)
        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {state.i + 1}: Found a better program on the valset with score {valset_score}.")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            new_program_idx=new_program_idx,
            valset_evaluation=valset_evaluation,
            objective_scores=state.prog_candidate_objective_scores[new_program_idx],
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            valset_size=len(self.valset),
            val_evaluation_policy=self.val_evaluation_policy,
        )
        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> GEPAState:
        # Check tqdm availability if progress bar is enabled
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError("tqdm must be installed when display_progress_bar is enabled")

            # Check if stop_callback contains MaxMetricCallsStopper
            total_calls = None
            if hasattr(self.stop_callback, "max_metric_calls"):
                # Direct MaxMetricCallsStopper
                total_calls = self.stop_callback.max_metric_calls
            elif hasattr(self.stop_callback, "stoppers"):
                # CompositeStopper - iterate to find MaxMetricCallsStopper
                for stopper in self.stop_callback.stoppers:
                    if hasattr(stopper, "max_metric_calls"):
                        total_calls = stopper.max_metric_calls
                        break

            if total_calls is not None:
                progress_bar = tqdm(total=total_calls, desc="GEPA Optimization", unit="rollouts")
            else:
                progress_bar = tqdm(desc="GEPA Optimization", unit="rollouts")
            progress_bar.update(0)
            last_pbar_val = 0

        # Prepare valset
        if self.valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        def valset_evaluator(program: dict[str, str]):
            all_ids = list(self.valset.all_ids())
            eval_result = self.evaluator(self.valset.fetch(all_ids), program)
            outputs, scores, objective_scores = self._coerce_evaluation_result(eval_result)
            outputs_dict = dict(zip(all_ids, outputs, strict=False))
            scores_dict = dict(zip(all_ids, scores, strict=False))
            objective_scores_dict = (
                dict(zip(all_ids, objective_scores, strict=False)) if objective_scores is not None else None
            )
            return outputs_dict, scores_dict, objective_scores_dict

        # Initialize state
        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            valset_evaluator=valset_evaluator,
            track_best_outputs=self.track_best_outputs,
        )

        # Log base program score
        base_val_avg, base_val_coverage = state.get_program_average_val_subset(0)
        self.experiment_tracker.log_metrics(
            {
                "base_program_full_valset_score": base_val_avg,
                "base_program_val_coverage": base_val_coverage,
                "iteration": state.i + 1,
            },
            step=state.i + 1,
        )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {base_val_avg} "
            f"over {base_val_coverage} / {len(self.valset)} examples"
        )

        # Merge scheduling
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        # Main loop
        while not self._should_stop(state):
            if self.display_progress_bar:
                delta = state.total_num_evals - last_pbar_val
                progress_bar.update(delta)
                last_pbar_val = state.total_num_evals

            assert state.is_consistent()
            try:
                state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
                state.i += 1
                state.full_program_trace.append({"i": state.i})

                # 1) Attempt merge first if scheduled and last iter found new program
                if self.merge_proposer is not None and self.merge_proposer.use_merge:
                    if self.merge_proposer.merges_due > 0 and self.merge_proposer.last_iter_found_new_program:
                        proposal = self.merge_proposer.propose(state)
                        self.merge_proposer.last_iter_found_new_program = False  # old behavior

                        if proposal is not None and proposal.tag == "merge":
                            parent_sums = proposal.subsample_scores_before or [
                                float("-inf"),
                                float("-inf"),
                            ]
                            new_sum = sum(proposal.subsample_scores_after or [])

                            if new_sum >= max(parent_sums):
                                # ACCEPTED: consume one merge attempt and record it
                                self._run_full_eval_and_add(
                                    new_program=proposal.candidate,
                                    state=state,
                                    parent_program_idx=proposal.parent_program_ids,
                                )
                                self.merge_proposer.merges_due -= 1
                                self.merge_proposer.total_merges_tested += 1
                                continue  # skip reflective this iteration
                            else:
                                # REJECTED: do NOT consume merges_due or total_merges_tested
                                self.logger.log(
                                    f"Iteration {state.i + 1}: New program subsample score {new_sum} "
                                    f"is worse than both parents {parent_sums}, skipping merge"
                                )
                                # Skip reflective this iteration (old behavior)
                                continue

                    # Old behavior: regardless of whether we attempted, clear the flag before reflective
                    self.merge_proposer.last_iter_found_new_program = False

                # 2) Reflective mutation proposer
                proposal = self.reflective_proposer.propose(state)
                if proposal is None:
                    self.logger.log(f"Iteration {state.i + 1}: Reflective mutation did not propose a new candidate")
                    continue

                # Acceptance: require strict improvement on subsample
                old_sum = sum(proposal.subsample_scores_before or [])
                new_sum = sum(proposal.subsample_scores_after or [])
                if new_sum <= old_sum:
                    self.logger.log(
                        f"Iteration {state.i + 1}: New subsample score {new_sum} is not better than old score {old_sum}, skipping"
                    )
                    continue
                else:
                    self.logger.log(
                        f"Iteration {state.i + 1}: New subsample score {new_sum} is better than old score {old_sum}. Continue to full eval and add to candidate pool."
                    )

                # Accept: full eval + add
                self._run_full_eval_and_add(
                    new_program=proposal.candidate,
                    state=state,
                    parent_program_idx=proposal.parent_program_ids,
                )

                # Schedule merge attempts like original behavior
                if self.merge_proposer is not None:
                    self.merge_proposer.last_iter_found_new_program = True
                    if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
                        self.merge_proposer.merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {state.i + 1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                if self.raise_on_exception:
                    raise e
                else:
                    continue

        # Close progress bar if it exists
        if self.display_progress_bar:
            progress_bar.close()

        state.save(self.run_dir)
        return state

    def _should_stop(self, state: GEPAState) -> bool:
        """Check if the optimization should stop."""
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback(state):
            return True
        return False

    def request_stop(self):
        """Manually request the optimization to stop gracefully."""
        self.logger.log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
