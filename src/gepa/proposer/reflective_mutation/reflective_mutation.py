# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature


@dataclass
class ProposalContext:
    """Pre-sampled context for a single proposal worker.

    Created by :meth:`ReflectiveMutationProposer.prepare_proposal` (sequential),
    then consumed by :meth:`ReflectiveMutationProposer.execute_proposal` (parallel-safe).
    """

    iteration: int
    curr_prog_id: int
    curr_prog: dict[str, str]
    curr_prog_score: float
    subsample_ids: list
    minibatch: list
    parent_ids: list[int]
    is_seed_candidate: bool


@dataclass
class ProposalOutput:
    """Result from :meth:`ReflectiveMutationProposer.execute_proposal`.

    Contains the proposal plus deferred state updates that must be applied
    sequentially via :meth:`ReflectiveMutationProposer.apply_proposal_output`.
    """

    proposal: CandidateProposal | None
    total_evals: int
    trace_data: dict[str, Any] = field(default_factory=dict)
    cache_entry: tuple | None = None


class ReflectiveMutationProposer(ProposeNewCandidate[DataId]):
    """Implements the reflective mutation flow.

    Supports parallel execution: call :meth:`prepare_proposal` sequentially,
    then :meth:`execute_proposal` from multiple threads, then
    :meth:`apply_proposal_output` sequentially.
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float | None,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        custom_candidate_proposer: ProposalFn | None = None,
        callbacks: list[GEPACallback] | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm
        self.custom_candidate_proposer = custom_candidate_proposer
        self.callbacks = callbacks
        self._lock = threading.Lock()

        self.reflection_prompt_template = reflection_prompt_template
        # Track parameters for which we've already logged missing template warnings
        self._missing_template_warnings: set[str] = set()

        if isinstance(reflection_prompt_template, dict):
            for _param_name, template in reflection_prompt_template.items():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)

        if self.skip_perfect_score and self.perfect_score is None:
            raise ValueError(
                "perfect_score must be provided when skip_perfect_score is True. "
                "If you do not have a perfect target score, set skip_perfect_score=False."
            )

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str], float]:
        """Propose new instruction texts for the given components.

        Returns:
            A tuple of (new_texts, prompts, raw_lm_outputs, reflection_cost_usd)
            where the first three are dicts keyed by component name and the last
            is the total USD cost of reflection LM calls made for this proposal.
            When the adapter or a custom proposer handles the call, prompts and
            raw_lm_outputs are empty dicts and reflection_cost is ``0.0``.
        """
        empty: dict[str, str | list[dict[str, Any]]] = {}
        if self.adapter.propose_new_texts is not None:
            return (
                self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update),
                empty,
                {},
                0.0,
            )

        if self.custom_candidate_proposer is not None:
            return (
                self.custom_candidate_proposer(candidate, reflective_dataset, components_to_update),
                empty,
                {},
                0.0,
            )

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")

        new_texts: dict[str, str] = {}
        prompts: dict[str, str | list[dict[str, Any]]] = {}
        raw_lm_outputs: dict[str, str] = {}
        total_cost = 0.0
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]

            # Determine which prompt template to use for this parameter
            prompt_template = None
            if isinstance(self.reflection_prompt_template, dict):
                # Use parameter-specific template if available
                prompt_template = self.reflection_prompt_template.get(name)
                if prompt_template is None and name not in self._missing_template_warnings:
                    self.logger.log(
                        f"No reflection_prompt_template found for parameter '{name}'. Using default template."
                    )
                    self._missing_template_warnings.add(name)
            else:
                # Use the single template for all parameters
                prompt_template = self.reflection_prompt_template

            result, prompt, raw_output, call_cost = InstructionProposalSignature.run_with_metadata(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": prompt_template,
                },
            )
            new_texts[name] = result["new_instruction"]
            prompts[name] = prompt
            raw_lm_outputs[name] = raw_output
            total_cost += call_cost
        return new_texts, prompts, raw_lm_outputs, total_cost

    def prepare_proposal(self, state: GEPAState) -> ProposalContext:
        """Select parent candidate and sample minibatch. Must be called sequentially.

        Performs the state-dependent, non-parallelizable parts of a proposal:
        candidate selection, minibatch sampling, and callback notifications
        that should fire in order.
        """
        i = state.i + 1

        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        curr_prog_score = state.program_full_scores_val_set[curr_prog_id]
        self.logger.log(f"Iteration {i}: Selected program {curr_prog_id} score: {curr_prog_score}")

        notify_callbacks(
            self.callbacks,
            "on_candidate_selected",
            CandidateSelectedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                candidate=curr_prog,
                score=curr_prog_score,
            ),
        )

        self.experiment_tracker.log_metrics(
            {"iteration": i, "selected_program_candidate": curr_prog_id, "total_metric_calls": state.total_num_evals},
            step=i,
        )

        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        minibatch = self.trainset.fetch(subsample_ids)

        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=i,
                minibatch_ids=subsample_ids,
                trainset_size=len(self.trainset),
            ),
        )

        curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
        is_seed_candidate = curr_prog_id == 0

        return ProposalContext(
            iteration=i,
            curr_prog_id=curr_prog_id,
            curr_prog=curr_prog,
            curr_prog_score=curr_prog_score,
            subsample_ids=subsample_ids,
            minibatch=minibatch,
            parent_ids=curr_parent_ids,
            is_seed_candidate=is_seed_candidate,
        )

    def execute_proposal(self, ctx: ProposalContext, state: GEPAState) -> ProposalOutput:
        """Run the evaluation + proposal pipeline. Safe for parallel execution.

        The only state mutation is the module_selector (e.g. RoundRobin counter),
        which is protected by a lock. All other state updates are deferred to
        :meth:`apply_proposal_output`.
        """
        i = ctx.iteration
        trace_data: dict[str, Any] = {
            "selected_program_candidate": ctx.curr_prog_id,
            "subsample_ids": ctx.subsample_ids,
        }
        total_evals = 0
        cache_entry = None

        # 1) Evaluate current program with traces
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=ctx.curr_prog_id,
                batch_size=len(ctx.minibatch),
                capture_traces=True,
                parent_ids=ctx.parent_ids,
                inputs=ctx.minibatch,
                is_seed_candidate=ctx.is_seed_candidate,
            ),
        )
        eval_curr = self.adapter.evaluate(ctx.minibatch, ctx.curr_prog, capture_traces=True)
        total_evals += eval_curr.num_metric_calls if eval_curr.num_metric_calls is not None else len(ctx.subsample_ids)
        trace_data["subsample_scores"] = eval_curr.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=ctx.curr_prog_id,
                scores=eval_curr.scores,
                has_trajectories=bool(eval_curr.trajectories),
                parent_ids=ctx.parent_ids,
                outputs=eval_curr.outputs,
                trajectories=eval_curr.trajectories,
                objective_scores=eval_curr.objective_scores,
                is_seed_candidate=ctx.is_seed_candidate,
            ),
        )

        # Prepare cache entry for parent evaluation
        objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
        cache_entry = (ctx.curr_prog, ctx.subsample_ids, eval_curr.outputs, eval_curr.scores, objective_scores_list)

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=ctx.curr_prog_id,
                    reason="no_trajectories",
                    scores=eval_curr.scores,
                    is_seed_candidate=ctx.is_seed_candidate,
                ),
            )
            return ProposalOutput(
                proposal=None, total_evals=total_evals, trace_data=trace_data, cache_entry=cache_entry
            )

        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
        ):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=ctx.curr_prog_id,
                    reason="all_scores_perfect",
                    scores=eval_curr.scores,
                    is_seed_candidate=ctx.is_seed_candidate,
                ),
            )
            return ProposalOutput(
                proposal=None, total_evals=total_evals, trace_data=trace_data, cache_entry=cache_entry
            )

        self.experiment_tracker.log_metrics(
            {"subsample_score": sum(eval_curr.scores), "total_metric_calls": total_evals}, step=i
        )

        # 2) Decide which components to update (lock protects RoundRobin state mutation)
        with self._lock:
            predictor_names_to_update = self.module_selector(
                state, eval_curr.trajectories, eval_curr.scores, ctx.curr_prog_id, ctx.curr_prog
            )

        # 3) Build reflective dataset and propose new content
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(ctx.curr_prog, eval_curr, predictor_names_to_update)

            reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                k: [dict(item) for item in v] for k, v in reflective_dataset.items()
            }

            notify_callbacks(
                self.callbacks,
                "on_reflective_dataset_built",
                ReflectiveDatasetBuiltEvent(
                    iteration=i,
                    candidate_idx=ctx.curr_prog_id,
                    components=predictor_names_to_update,
                    dataset=reflective_dataset_concrete,
                ),
            )

            notify_callbacks(
                self.callbacks,
                "on_proposal_start",
                ProposalStartEvent(
                    iteration=i,
                    parent_candidate=ctx.curr_prog,
                    components=predictor_names_to_update,
                    reflective_dataset=reflective_dataset_concrete,
                ),
            )

            new_texts, prompts, raw_lm_outputs, reflection_cost = self.propose_new_texts(
                ctx.curr_prog, reflective_dataset, predictor_names_to_update
            )

            notify_callbacks(
                self.callbacks,
                "on_proposal_end",
                ProposalEndEvent(
                    iteration=i,
                    new_instructions=new_texts,
                    prompts=prompts,
                    raw_lm_outputs=raw_lm_outputs,
                ),
            )

            _lm_metadata: dict[str, Any] = {}
            for comp in new_texts:
                _lm_metadata[f"prompt:{comp}"] = prompts.get(comp, "")
                _lm_metadata[f"raw_lm_output:{comp}"] = raw_lm_outputs.get(comp, "")

            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback

            self.logger.log(traceback.format_exc())
            return ProposalOutput(
                proposal=None, total_evals=total_evals, trace_data=trace_data, cache_entry=cache_entry
            )

        # 4) Create candidate, evaluate on same minibatch
        new_candidate = ctx.curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=None,
                batch_size=len(ctx.minibatch),
                capture_traces=True,
                parent_ids=[ctx.curr_prog_id],
                inputs=ctx.minibatch,
                is_seed_candidate=False,
            ),
        )

        eval_after = self.adapter.evaluate(ctx.minibatch, new_candidate, capture_traces=True)
        new_scores = eval_after.scores
        new_outputs = eval_after.outputs
        total_evals += eval_after.num_metric_calls if eval_after.num_metric_calls is not None else len(ctx.subsample_ids)

        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=None,
                scores=new_scores,
                has_trajectories=bool(eval_after.trajectories),
                parent_ids=[ctx.curr_prog_id],
                outputs=new_outputs,
                trajectories=eval_after.trajectories,
                objective_scores=eval_after.objective_scores,
                is_seed_candidate=False,
            ),
        )

        trace_data["new_subsample_scores"] = new_scores
        new_sum = sum(new_scores)
        self.experiment_tracker.log_metrics(
            {"new_subsample_score": new_sum, "total_metric_calls": total_evals}, step=i
        )

        proposal = CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[ctx.curr_prog_id],
            subsample_indices=ctx.subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=new_scores,
            eval_before=SubsampleEvaluation(
                scores=eval_curr.scores,
                outputs=eval_curr.outputs,
                objective_scores=list(eval_curr.objective_scores) if eval_curr.objective_scores else None,
                trajectories=eval_curr.trajectories,
            ),
            eval_after=SubsampleEvaluation(
                scores=new_scores,
                outputs=new_outputs,
                objective_scores=list(eval_after.objective_scores) if eval_after.objective_scores else None,
                trajectories=eval_after.trajectories,
            ),
            tag="reflective_mutation",
            metadata=_lm_metadata,
            reflection_cost=reflection_cost,
        )
        return ProposalOutput(proposal=proposal, total_evals=total_evals, trace_data=trace_data, cache_entry=cache_entry)

    def apply_proposal_output(self, output: ProposalOutput, state: GEPAState) -> None:
        """Apply deferred state updates from a proposal. Must be called sequentially."""
        state.increment_evals(output.total_evals)
        if output.cache_entry is not None and state.evaluation_cache is not None:
            candidate, ids, outputs, scores, obj_scores = output.cache_entry
            state.evaluation_cache.put_batch(candidate, ids, outputs, scores, obj_scores)

    def propose_output(self, state: GEPAState) -> ProposalOutput:
        """Run a single reflective mutation iteration, returning a :class:`ProposalOutput`.

        The caller is responsible for passing the output to
        :meth:`apply_proposal_output`.
        """
        ctx = self.prepare_proposal(state)
        state.full_program_trace[-1].update({
            "selected_program_candidate": ctx.curr_prog_id,
            "subsample_ids": ctx.subsample_ids,
        })
        return self.execute_proposal(ctx, state)

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Run a single reflective mutation iteration.

        Convenience method equivalent to :meth:`propose_output` followed by
        :meth:`apply_proposal_output`.
        """
        output = self.propose_output(state)
        self.apply_proposal_output(output, state)
        state.full_program_trace[-1].update(output.trace_data)
        return output.proposal
