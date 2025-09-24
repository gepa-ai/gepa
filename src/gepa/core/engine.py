# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import traceback
from typing import Any, Callable, Generic

from gepa.core.state import GEPAState, initialize_gepa_state
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

from .adapter import DataInst, RolloutOutput, Trajectory

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GEPAEngine(Generic[DataInst, Trajectory, RolloutOutput]):
    """
    Orchestrates the optimization loop. It uses pluggable ProposeNewCandidate strategies.
    """

    def __init__(
        self,
        run_dir: str | None,
        evaluator: Callable[[list[DataInst], dict[str, str]], tuple[list[RolloutOutput], list[float]]],
        valset: list[DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        # Logging
        logger: Any,
        experiment_tracker: Any,
        # Optional parameters
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        # Budget and Stop Condition
        stop_callback: Callable[[Any], bool] | None = None,
    ):
        self.logger = logger
        self.run_dir = run_dir

        # Graceful stopping mechanism
        self._stop_requested = False

        # Set up stopping mechanism
        self.stop_callback = stop_callback
        self.evaluator = evaluator
        self.valset = valset
        self.seed_candidate = seed_candidate

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer

        # Merge scheduling flags (mirroring previous behavior)
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar

        self.raise_on_exception = raise_on_exception

    def _val_evaluator(self) -> Callable[[dict[str, str]], tuple[list[RolloutOutput], list[float]]]:
        assert self.valset is not None
        return lambda prog: self.evaluator(self.valset, prog)

    def _get_pareto_front_programs(self, state: GEPAState) -> list:
        return state.program_at_pareto_front_valset

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState,
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        valset_outputs, valset_subscores = self._val_evaluator()(new_program)
        valset_score = sum(valset_subscores) / len(valset_subscores)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )
        state.full_program_trace[-1]["new_program_idx"] = new_program_idx

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {state.i + 1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            valset_score=valset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
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

        # Initialize state
        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            valset_evaluator=self._val_evaluator(),
            track_best_outputs=self.track_best_outputs,
        )

        assert len(state.pareto_front_valset) == len(self.valset)

        # Log base program score
        self.experiment_tracker.log_metrics(
            {
                "base_program_full_valset_score": state.program_full_scores_val_set[0],
                "iteration": state.i + 1,
            },
            step=state.i + 1,
        )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {state.program_full_scores_val_set[0]}"
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
                state.save(self.run_dir)
                state.i += 1
                state.full_program_trace.append({"i": state.i})

                # 1) Attempt merge first if scheduled and last iter found new program
                if self.merge_proposer is not None and self.merge_proposer.use_merge:
                    if self.merge_proposer.merges_due > 0 and self.merge_proposer.last_iter_found_new_program:
                        proposal = self.merge_proposer.propose(state)
                        self.merge_proposer.last_iter_found_new_program = False  # old behavior

                        if proposal is not None and proposal.tag == "merge":
                            parent_sums = proposal.subsample_scores_before or [float("-inf"), float("-inf")]
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
                    self.logger.log(f"Iteration {state.i + 1}: New subsample score {new_sum} is not better than old score {old_sum}, skipping")
                    continue
                else:
                    self.logger.log(f"Iteration {state.i + 1}: New subsample score {new_sum} is better than old score {old_sum}. Continue to full eval and add to candidate pool.")

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
