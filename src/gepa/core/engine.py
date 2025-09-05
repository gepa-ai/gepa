# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import signal
import traceback
from typing import Any, Callable, Generic

from gepa.core.state import GEPAState, initialize_gepa_state
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.logging.wandb_utils import initialize_wandb
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
        max_metric_calls: int | None,
        perfect_score: float,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        # Logging
        logger: Any,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        # Graceful stopping
        stop_callback: Callable[[], bool] | None = None,
        enable_signal_handling: bool = True,
    ):
        # Budget constraint: max_metric_calls must be set
        assert max_metric_calls is not None, "max_metric_calls must be set"

        self.logger = logger
        self.run_dir = run_dir
        
        # Graceful stopping mechanism
        self.stop_callback = stop_callback
        self.enable_signal_handling = enable_signal_handling
        self._stop_requested = False
        self._original_signal_handlers = {}
        
        # Set up signal handling for graceful shutdown
        if self.enable_signal_handling:
            self._setup_signal_handlers()
        self.evaluator = evaluator
        self.valset = valset
        self.seed_candidate = seed_candidate

        self.max_metric_calls = max_metric_calls

        self.perfect_score = perfect_score
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.seed = seed
        self.wandb_init_kwargs = wandb_init_kwargs or {}

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

        valset_multi_scores = None
        if hasattr(valset_outputs, 'multi_scores') and valset_outputs.multi_scores is not None:
            valset_multi_scores = valset_outputs.multi_scores
        elif isinstance(valset_outputs, list) and len(valset_outputs) > 0:
            # for extracting multi_scores from the first output if it's an evaluation batch
            if hasattr(valset_outputs[0], 'multi_scores') and valset_outputs[0].multi_scores is not None:
                valset_multi_scores = valset_outputs[0].multi_scores

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            valset_multi_scores=valset_multi_scores,
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
            # new_instruction="Merged or Reflective program",
            use_wandb=self.use_wandb,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
        )
        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> GEPAState:
        if self.use_wandb:
            initialize_wandb(wandb_api_key=self.wandb_api_key, wandb_init_kwargs=self.wandb_init_kwargs)

        # Check tqdm availability if progress bar is enabled
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError("tqdm must be installed when display_progress_bar is enabled")
            # Initialize progress bar
            progress_bar = tqdm(total=self.max_metric_calls, desc="GEPA Optimization", unit="rollouts")
            progress_bar.update(0)
            last_pbar_val = 0

        # Prepare valset
        if self.valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        # Initialize state (keeps your previous semantics)
        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            valset_evaluator=self._val_evaluator(),
            track_best_outputs=self.track_best_outputs,
        )

        assert len(state.pareto_front_valset) == len(self.valset)

        if self.use_wandb:
            import wandb  # type: ignore

            wandb.log(
                {
                    "base_program_full_valset_score": state.program_full_scores_val_set[0],
                    "iteration": state.i + 1,
                }
            )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {state.program_full_scores_val_set[0]}"
        )

        # Merge scheduling
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        # Main loop
        while state.total_num_evals < self.max_metric_calls:
            # Check for stop request
            if self._should_stop():
                self.logger.log(f"Iteration {state.i + 1}: Stop requested. Saving state and exiting gracefully...")
                break
                
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
                        # Old behavior: clear the flag as soon as we attempt a merge
                        self.merge_proposer.last_iter_found_new_program = False

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

                                # Skip reflective this iteration (old behavior)
                                continue
                            else:
                                # REJECTED: do NOT consume merges_due or total_merges_tested
                                self.logger.log(
                                    f"Iteration {state.i + 1}: New program subsample score {new_sum} is worse than both parents {parent_sums}, skipping merge"
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
                    self.logger.log(f"Iteration {state.i + 1}: New subsample score is not better, skipping")
                    continue

                # Accept: full eval + add
                self._run_full_eval_and_add(
                    new_program=proposal.candidate, state=state, parent_program_idx=proposal.parent_program_ids
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

        # Restore original signal handlers
        self._restore_signal_handlers()
        
        state.save(self.run_dir)
        return state

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.log(f"Received signal {signum}. Initiating graceful shutdown...")
            self._stop_requested = True
            
        # Store original handlers and set new ones
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                self._original_signal_handlers[sig] = signal.signal(sig, signal_handler)
            except (OSError, ValueError):
                # Signal not available on this platform
                pass

    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_signal_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass

    def _should_stop(self) -> bool:
        """Check if the optimization should stop."""
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback():
            return True
        return False

    def request_stop(self):
        """Manually request the optimization to stop gracefully."""
        self.logger.log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
