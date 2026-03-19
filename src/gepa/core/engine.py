# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import os
import traceback
from collections.abc import Sequence
from typing import Any, Generic

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    BudgetUpdatedEvent,
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    ErrorEvent,
    GEPACallback,
    IterationEndEvent,
    IterationStartEvent,
    MergeAcceptedEvent,
    MergeAttemptedEvent,
    MergeRejectedEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ParetoFrontUpdatedEvent,
    StateSavedEvent,
    ValsetEvaluatedEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import EvaluationCache, FrontierType, GEPAState, ValsetEvaluation, initialize_gepa_state
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import (
    ReflectiveMutationProposer,
)
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.utils import StopperProtocol

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Orchestrates the optimization loop using pluggable candidate proposers."""

    def __init__(
        self,
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        run_dir: str | None,
        valset: list[DataInst] | DataLoader[DataId, DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float | None,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        frontier_type: FrontierType,
        # Logging
        logger: LoggerProtocol,
        experiment_tracker: ExperimentTracker,
        # Callbacks
        callbacks: list[GEPACallback] | None = None,
        # Optional parameters
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        # Budget and Stop Condition
        stop_callback: StopperProtocol | None = None,
        val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | None = None,
        # Evaluation caching (stored in state, passed here for initialization)
        evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None,
    ):
        self.logger = logger
        self.run_dir = run_dir
        self.callbacks = callbacks

        # Graceful stopping mechanism
        self._stop_requested = False

        # Set up stopping mechanism
        self.stop_callback = stop_callback
        self.adapter = adapter

        # Store cache reference for state initialization (actual cache lives in GEPAState)
        self._initial_evaluation_cache = evaluation_cache

        def evaluator(
            batch: list[DataInst], program: dict[str, str]
        ) -> tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]:
            eval_result = adapter.evaluate(batch, program, capture_traces=False)
            return eval_result.outputs, eval_result.scores, eval_result.objective_scores

        self.evaluator = evaluator

        self.valset = ensure_loader(valset) if valset is not None else None
        self.seed_candidate = seed_candidate

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer
        self.frontier_type: FrontierType = frontier_type

        # Merge scheduling flags (mirroring previous behavior)
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar
        self.use_cloudpickle = use_cloudpickle

        self.raise_on_exception = raise_on_exception
        self.val_evaluation_policy: EvaluationPolicy[DataId, DataInst] = (
            val_evaluation_policy if val_evaluation_policy is not None else FullEvaluationPolicy()
        )

    def _evaluate_on_valset(
        self,
        program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
    ) -> ValsetEvaluation[RolloutOutput, DataId]:
        valset = self.valset
        assert valset is not None

        val_ids = self.val_evaluation_policy.get_eval_batch(valset, state)

        outputs_by_val_idx, scores_by_val_idx, objective_by_val_idx, num_actual_evals = state.cached_evaluate_full(
            program, list(val_ids), valset.fetch, self.evaluator
        )
        state.increment_evals(num_actual_evals)

        return ValsetEvaluation(
            outputs_by_val_id=outputs_by_val_idx,
            scores_by_val_id=scores_by_val_idx,
            objective_scores_by_val_id=objective_by_val_idx,
        )

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals
        valset_evaluation = self._evaluate_on_valset(new_program, state)
        state.num_full_ds_evals += 1

        # Snapshot Pareto front before update
        front_before = state.get_pareto_front_mapping()
        candidates_before: set[int] = set()
        for program_set in front_before.values():
            candidates_before.update(program_set)

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_evaluation=valset_evaluation,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )

        # Compute best program immediately after state update (before callbacks)
        # to ensure is_best_program reflects the updated Pareto front
        valset_score = self.val_evaluation_policy.get_valset_score(new_program_idx, state)
        linear_pareto_front_program_idx = self.val_evaluation_policy.get_best_program(state)
        is_best_program = new_program_idx == linear_pareto_front_program_idx

        # Snapshot Pareto front after update and notify callback
        front_after = state.get_pareto_front_mapping()
        candidates_after: set[int] = set()
        for program_set in front_after.values():
            candidates_after.update(program_set)

        new_front = sorted(candidates_after)
        displaced_candidates = sorted(candidates_before - candidates_after)

        notify_callbacks(
            self.callbacks,
            "on_pareto_front_updated",
            ParetoFrontUpdatedEvent(
                iteration=state.i + 1,
                new_front=new_front,
                displaced_candidates=displaced_candidates,
            ),
        )

        state.full_program_trace[-1]["new_program_idx"] = new_program_idx
        state.full_program_trace[-1]["evaluated_val_indices"] = sorted(valset_evaluation.scores_by_val_id.keys())

        if is_best_program:
            self.logger.log(f"Iteration {state.i + 1}: Found a better program on the valset with score {valset_score}.")

        valset = self.valset
        assert valset is not None

        notify_callbacks(
            self.callbacks,
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=state.i + 1,
                candidate_idx=new_program_idx,
                candidate=new_program,
                scores_by_val_id=dict(valset_evaluation.scores_by_val_id),
                average_score=valset_score,
                num_examples_evaluated=len(valset_evaluation.scores_by_val_id),
                total_valset_size=len(valset),
                parent_ids=parent_program_idx,
                is_best_program=is_best_program,
                outputs_by_val_id=(
                    dict(valset_evaluation.outputs_by_val_id) if valset_evaluation.outputs_by_val_id else None
                ),
            ),
        )

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            new_program_idx=new_program_idx,
            valset_evaluation=valset_evaluation,
            objective_scores=state.prog_candidate_objective_scores[new_program_idx],
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            valset_size=len(valset),
            val_evaluation_policy=self.val_evaluation_policy,
        )

        # Log candidate table row with instructions and metadata
        component_names = sorted(new_program.keys())
        columns = ["iteration", "candidate_idx", "parent_ids", "valset_score", "is_best"] + [
            f"text:{name}" for name in component_names
        ]
        row = [
            state.i + 1,
            new_program_idx,
            str(parent_program_idx),
            valset_score,
            is_best_program,
        ] + [new_program[name] for name in component_names]
        self.experiment_tracker.log_table("candidates", columns=columns, data=[row])

        # Update candidate tree visualization
        self._log_candidate_tree(state)

        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        # Check tqdm availability if progress bar is enabled
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError("tqdm must be installed when display_progress_bar is enabled")

            # Check if stop_callback contains MaxMetricCallsStopper
            total_calls: int | None = None
            stop_cb = self.stop_callback
            if stop_cb is not None:
                max_calls_attr = getattr(stop_cb, "max_metric_calls", None)
                if isinstance(max_calls_attr, int):
                    # Direct MaxMetricCallsStopper
                    total_calls = max_calls_attr
                else:
                    stoppers = getattr(stop_cb, "stoppers", None)
                    if stoppers is not None:
                        # CompositeStopper - iterate to find MaxMetricCallsStopper
                        for stopper in stoppers:
                            stopper_max = getattr(stopper, "max_metric_calls", None)
                            if isinstance(stopper_max, int):
                                total_calls = stopper_max
                                break

            if total_calls is not None:
                progress_bar = tqdm(total=total_calls, desc="GEPA Optimization", unit="rollouts")
            else:
                progress_bar = tqdm(desc="GEPA Optimization", unit="rollouts")
            progress_bar.update(0)

        # Prepare valset
        valset = self.valset
        if valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        def valset_evaluator(
            program: dict[str, str],
        ) -> ValsetEvaluation[RolloutOutput, DataId]:
            all_ids = list(valset.all_ids())
            outputs, scores, objective_scores = self.evaluator(valset.fetch(all_ids), program)
            outputs_dict = dict(zip(all_ids, outputs, strict=False))
            scores_dict = dict(zip(all_ids, scores, strict=False))
            objective_scores_dict = (
                dict(zip(all_ids, objective_scores, strict=False)) if objective_scores is not None else None
            )
            return ValsetEvaluation(
                outputs_by_val_id=outputs_dict,
                scores_by_val_id=scores_dict,
                objective_scores_by_val_id=objective_scores_dict,
            )

        # Notify callbacks of optimization start (before seed valset eval)
        notify_callbacks(
            self.callbacks,
            "on_optimization_start",
            OptimizationStartEvent(
                seed_candidate=self.seed_candidate,
                trainset_size=len(self.reflective_proposer.trainset),
                valset_size=len(valset),
                config={
                    "perfect_score": self.perfect_score,
                    "seed": self.seed,
                    "track_best_outputs": self.track_best_outputs,
                },
            ),
        )

        # Evaluate seed candidate on valset (after on_optimization_start callback)
        seed_valset_evaluation = valset_evaluator(self.seed_candidate)

        # Initialize state with pre-computed seed evaluation
        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            seed_valset_evaluation=seed_valset_evaluation,
            track_best_outputs=self.track_best_outputs,
            frontier_type=self.frontier_type,
            evaluation_cache=self._initial_evaluation_cache,
        )

        # Log run configuration
        self.experiment_tracker.log_config(
            {
                "seed": self.seed,
                "perfect_score": self.perfect_score,
                "frontier_type": self.frontier_type,
                "track_best_outputs": self.track_best_outputs,
                "use_cloudpickle": self.use_cloudpickle,
                "raise_on_exception": self.raise_on_exception,
                "trainset_size": len(self.reflective_proposer.trainset),
                "valset_size": len(valset),
                "seed_candidate_components": sorted(self.seed_candidate.keys()),
                "val_evaluation_policy": type(self.val_evaluation_policy).__name__,
                "has_merge_proposer": self.merge_proposer is not None,
                "run_dir": self.run_dir,
            }
        )

        # Log base program score using the same metric names as subsequent iterations
        # so they appear on the same charts in wandb/mlflow
        base_val_avg, base_val_coverage = state.get_program_average_val_subset(0)
        pareto_scores = list(state.pareto_front_valset.values())
        base_pareto_avg = sum(pareto_scores) / len(pareto_scores) if pareto_scores else base_val_avg
        self.experiment_tracker.log_metrics(
            {
                "val_program_average": base_val_avg,
                "best_score_on_valset": base_val_avg,
                "val_evaluated_count_new_program": base_val_coverage,
                "val_total_count": len(valset),
                "total_metric_calls": state.total_num_evals,
                "valset_pareto_front_agg": base_pareto_avg,
                "new_program_idx": 0,
                "linear_pareto_front_program_idx": 0,
                "best_program_as_per_agg_score_valset": 0,
            },
            step=state.i + 1,
        )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {base_val_avg} "
            f"over {base_val_coverage} / {len(valset)} examples"
        )

        # Notify callbacks of seed candidate's initial valset evaluation (iteration 0)
        # This provides the baseline performance before any optimization
        seed_scores = state.prog_candidate_val_subscores[0]
        notify_callbacks(
            self.callbacks,
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=0,
                candidate_idx=0,
                candidate=self.seed_candidate,
                scores_by_val_id=dict(seed_scores),
                average_score=base_val_avg,
                num_examples_evaluated=len(seed_scores),
                total_valset_size=len(valset),
                parent_ids=[],
                is_best_program=True,  # Seed is always best at iteration 0
                outputs_by_val_id=None,  # Outputs not tracked at initialization unless track_best_outputs=True
            ),
        )

        # Register budget hook to fire on_budget_updated callback in real-time
        def budget_hook(new_total: int, delta: int) -> None:
            notify_callbacks(
                self.callbacks,
                "on_budget_updated",
                BudgetUpdatedEvent(
                    iteration=state.i + 1,
                    metric_calls_used=new_total,
                    metric_calls_delta=delta,
                    metric_calls_remaining=self._get_remaining_budget(state),
                ),
            )

        state.add_budget_hook(budget_hook)

        # Merge scheduling
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        # Main loop
        last_pbar_val = 0
        while not self._should_stop(state):
            if self.display_progress_bar and progress_bar is not None:
                delta = state.total_num_evals - last_pbar_val
                progress_bar.update(delta)
                last_pbar_val = state.total_num_evals

            assert state.is_consistent()
            proposal_accepted = False
            iteration_started = False
            try:
                state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
                notify_callbacks(
                    self.callbacks,
                    "on_state_saved",
                    StateSavedEvent(
                        iteration=state.i + 1,
                        run_dir=self.run_dir,
                    ),
                )

                state.i += 1
                state.full_program_trace.append({"i": state.i})

                # Notify callbacks of iteration start
                notify_callbacks(
                    self.callbacks,
                    "on_iteration_start",
                    IterationStartEvent(
                        iteration=state.i + 1,
                        state=state,
                        trainset_loader=self.reflective_proposer.trainset,
                    ),
                )
                iteration_started = True

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

                            # Notify merge attempted
                            notify_callbacks(
                                self.callbacks,
                                "on_merge_attempted",
                                MergeAttemptedEvent(
                                    iteration=state.i + 1,
                                    parent_ids=proposal.parent_program_ids,
                                    merged_candidate=proposal.candidate,
                                ),
                            )

                            if new_sum >= max(parent_sums):
                                # ACCEPTED: consume one merge attempt and record it
                                new_idx, _ = self._run_full_eval_and_add(
                                    new_program=proposal.candidate,
                                    state=state,
                                    parent_program_idx=proposal.parent_program_ids,
                                )
                                self.merge_proposer.merges_due -= 1
                                self.merge_proposer.total_merges_tested += 1
                                proposal_accepted = True

                                # Notify merge accepted
                                notify_callbacks(
                                    self.callbacks,
                                    "on_merge_accepted",
                                    MergeAcceptedEvent(
                                        iteration=state.i + 1,
                                        new_candidate_idx=new_idx,
                                        parent_ids=proposal.parent_program_ids,
                                    ),
                                )
                                notify_callbacks(
                                    self.callbacks,
                                    "on_candidate_accepted",
                                    CandidateAcceptedEvent(
                                        iteration=state.i + 1,
                                        new_candidate_idx=new_idx,
                                        new_score=new_sum,
                                        parent_ids=proposal.parent_program_ids,
                                    ),
                                )
                                continue  # skip reflective this iteration
                            else:
                                # REJECTED: do NOT consume merges_due or total_merges_tested
                                self.logger.log(
                                    f"Iteration {state.i + 1}: New program subsample score {new_sum} "
                                    f"is worse than both parents {parent_sums}, skipping merge"
                                )
                                # Notify merge rejected
                                notify_callbacks(
                                    self.callbacks,
                                    "on_merge_rejected",
                                    MergeRejectedEvent(
                                        iteration=state.i + 1,
                                        parent_ids=proposal.parent_program_ids,
                                        reason=f"Merged score {new_sum} worse than both parents {parent_sums}",
                                    ),
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
                    # Log rejected proposal LM call to experiment tracker
                    self._log_proposal_lm_calls(state.i + 1, proposal, candidate_idx=-1)
                    # Notify candidate rejected
                    notify_callbacks(
                        self.callbacks,
                        "on_candidate_rejected",
                        CandidateRejectedEvent(
                            iteration=state.i + 1,
                            old_score=old_sum,
                            new_score=new_sum,
                            reason=f"New subsample score {new_sum} not better than old score {old_sum}",
                        ),
                    )
                    continue
                else:
                    self.logger.log(
                        f"Iteration {state.i + 1}: New subsample score {new_sum} is better than old score {old_sum}. Continue to full eval and add to candidate pool."
                    )

                # Accept: full eval + add
                new_idx, _ = self._run_full_eval_and_add(
                    new_program=proposal.candidate,
                    state=state,
                    parent_program_idx=proposal.parent_program_ids,
                )
                proposal_accepted = True

                # Log accepted proposal LM call to experiment tracker (candidate_idx now known)
                self._log_proposal_lm_calls(state.i + 1, proposal, candidate_idx=new_idx)

                # Notify candidate accepted
                notify_callbacks(
                    self.callbacks,
                    "on_candidate_accepted",
                    CandidateAcceptedEvent(
                        iteration=state.i + 1,
                        new_candidate_idx=new_idx,
                        new_score=new_sum,
                        parent_ids=proposal.parent_program_ids,
                    ),
                )

                # Schedule merge attempts like original behavior
                if self.merge_proposer is not None:
                    self.merge_proposer.last_iter_found_new_program = True
                    if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
                        self.merge_proposer.merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {state.i + 1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                # Notify error callback
                notify_callbacks(
                    self.callbacks,
                    "on_error",
                    ErrorEvent(
                        iteration=state.i + 1,
                        exception=e,
                        will_continue=not self.raise_on_exception,
                    ),
                )
                if self.raise_on_exception:
                    raise e
                else:
                    continue
            finally:
                # Notify iteration end only if the iteration actually started
                # (i.e., on_iteration_start was called successfully)
                if iteration_started:
                    notify_callbacks(
                        self.callbacks,
                        "on_iteration_end",
                        IterationEndEvent(
                            iteration=state.i + 1,
                            state=state,
                            proposal_accepted=proposal_accepted,
                        ),
                    )

        # Close progress bar if it exists
        if self.display_progress_bar and progress_bar is not None:
            progress_bar.close()

        state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)

        # Notify optimization end
        best_candidate_idx = self.val_evaluation_policy.get_best_program(state)
        notify_callbacks(
            self.callbacks,
            "on_optimization_end",
            OptimizationEndEvent(
                best_candidate_idx=best_candidate_idx,
                total_iterations=state.i,
                total_metric_calls=state.total_num_evals,
                final_state=state,
            ),
        )

        # Log final summary: seed candidate, best candidate, and all candidates table
        best_candidate = state.program_candidates[best_candidate_idx]
        best_score = self.val_evaluation_policy.get_valset_score(best_candidate_idx, state)
        summary: dict[str, Any] = {
            "best_candidate_idx": best_candidate_idx,
            "best_valset_score": best_score,
            "total_iterations": state.i,
            "total_candidates": len(state.program_candidates),
        }
        for name in sorted(self.seed_candidate.keys()):
            summary[f"seed/{name}"] = self.seed_candidate[name]
            summary[f"best/{name}"] = best_candidate[name]
        self.experiment_tracker.log_summary(summary)

        return state

    def _log_proposal_lm_calls(
        self,
        iteration: int,
        proposal: Any,
        candidate_idx: int,
    ) -> None:
        """Log per-component LM prompt / raw-output from a proposal to the experiment tracker.

        Appends one row per component to the ``"proposals"`` table.
        ``candidate_idx`` is the assigned index for accepted proposals,
        or ``-1`` for rejected ones — making it easy to join with the
        ``"candidates"`` table in WandB / MLflow.
        """
        metadata = proposal.metadata or {}
        components = {
            k.split(":", 1)[1]
            for k in metadata
            if k.startswith("prompt:") or k.startswith("raw_lm_output:")
        }
        if not components:
            return

        status = "accepted" if candidate_idx >= 0 else "rejected"
        subsample_before = sum(proposal.subsample_scores_before or [])
        subsample_after = sum(proposal.subsample_scores_after or [])
        parent_ids_str = str(proposal.parent_program_ids)

        rows = []
        for comp in sorted(components):
            prompt = metadata.get(f"prompt:{comp}", "")
            raw_output = metadata.get(f"raw_lm_output:{comp}", "")
            proposed_text = proposal.candidate.get(comp, "")
            rows.append([
                iteration,
                comp,
                status,
                candidate_idx,
                parent_ids_str,
                subsample_before,
                subsample_after,
                prompt if isinstance(prompt, str) else str(prompt),
                raw_output,
                proposed_text,
            ])

        self.experiment_tracker.log_table(
            "proposals",
            columns=[
                "iteration",
                "component",
                "status",
                "candidate_idx",
                "parent_ids",
                "subsample_score_before",
                "subsample_score_after",
                "prompt",
                "raw_lm_output",
                "proposed_text",
            ],
            data=rows,
        )

    def _log_candidate_tree(self, state: GEPAState[RolloutOutput, DataId]) -> None:
        """Generate and log the candidate tree visualization."""
        try:
            from gepa.visualization import candidate_tree_html

            html_content = candidate_tree_html(state)
            self.experiment_tracker.log_html(html_content, key="candidate_tree")
            if self.run_dir is not None:
                tree_path = os.path.join(self.run_dir, "candidate_tree.html")
                with open(tree_path, "w") as f:
                    f.write(html_content)
        except Exception as e:
            self.logger.log(f"Warning: Failed to generate candidate tree visualization: {e}")

    def _should_stop(self, state: GEPAState[RolloutOutput, DataId]) -> bool:
        """Check if the optimization should stop."""
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback(state):
            return True
        return False

    def _get_remaining_budget(self, state: GEPAState[RolloutOutput, DataId]) -> int | None:
        """Get remaining metric calls budget, or None if unlimited."""
        stop_cb = self.stop_callback
        if stop_cb is None:
            return None

        max_calls = getattr(stop_cb, "max_metric_calls", None)
        if isinstance(max_calls, int):
            return max(0, max_calls - state.total_num_evals)

        # Check for CompositeStopper
        stoppers = getattr(stop_cb, "stoppers", None)
        if stoppers is not None:
            for stopper in stoppers:
                stopper_max = getattr(stopper, "max_metric_calls", None)
                if isinstance(stopper_max, int):
                    return max(0, stopper_max - state.total_num_evals)

        return None

    def request_stop(self) -> None:
        """Manually request the optimization to stop gracefully."""
        self.logger.log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
