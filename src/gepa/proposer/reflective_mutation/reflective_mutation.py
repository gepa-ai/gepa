# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import threading
import traceback as _traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from gepa.proposer.base import CandidateProposal, Proposer, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.strategies.acceptance import AcceptanceCriterion
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.proposal_sampling import ProposalTask, SamplingStrategy
from gepa.strategies.proposal_selection import SelectionStrategy


@dataclass
class ProposalContext:
    """Pre-sampled context for a single proposal worker.

    Created by :meth:`ReflectiveMutationProposer._context_from_task` (sequential),
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


class ReflectiveMutationProposer(Proposer[DataId]):
    """Implements the reflective mutation flow.

    Inherits from :class:`~gepa.proposer.base.Proposer`.  The engine calls
    ``propose(state)`` which drives the full sampling → mutation → selection
    pipeline; subclass responsibility is only ``_mutate()``.

    Parallel execution is controlled by ``num_parallel_proposals`` (default 1).
    Pass ``sampling_strategy=IndependentSampling(n)`` or another strategy to
    control how tasks are generated; the proposer runs them all in parallel.
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
        num_parallel_proposals: int = 1,
        acceptance_criterion: AcceptanceCriterion | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        selection_strategy: SelectionStrategy | None = None,
    ):
        resolved_trainset = ensure_loader(trainset)
        super().__init__(
            trainset=resolved_trainset,
            candidate_selector=candidate_selector,
            batch_sampler=batch_sampler,
            acceptance_criterion=acceptance_criterion,
            sampling_strategy=sampling_strategy,
            selection_strategy=selection_strategy,
        )
        self.logger = logger
        self.adapter = adapter
        self.module_selector = module_selector
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm
        self.custom_candidate_proposer = custom_candidate_proposer
        self.callbacks = callbacks
        self.num_parallel_proposals = num_parallel_proposals
        self._lock = threading.Lock()

        self.reflection_prompt_template = reflection_prompt_template
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
    ) -> tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str]]:
        """Propose new instruction texts for the given components.

        Returns:
            A tuple of (new_texts, prompts, raw_lm_outputs) where each is a
            dict keyed by component name.  When the adapter or a custom proposer
            handles the call, prompts and raw_lm_outputs are empty dicts.
        """
        empty: dict[str, str | list[dict[str, Any]]] = {}
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update), empty, {}

        if self.custom_candidate_proposer is not None:
            return self.custom_candidate_proposer(candidate, reflective_dataset, components_to_update), empty, {}

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")

        new_texts: dict[str, str] = {}
        prompts: dict[str, str | list[dict[str, Any]]] = {}
        raw_lm_outputs: dict[str, str] = {}
        for name in components_to_update:
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]

            prompt_template = None
            if isinstance(self.reflection_prompt_template, dict):
                prompt_template = self.reflection_prompt_template.get(name)
                if prompt_template is None and name not in self._missing_template_warnings:
                    self.logger.log(
                        f"No reflection_prompt_template found for parameter '{name}'. Using default template."
                    )
                    self._missing_template_warnings.add(name)
            else:
                prompt_template = self.reflection_prompt_template

            result, prompt, raw_output = InstructionProposalSignature.run_with_metadata(
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
        return new_texts, prompts, raw_lm_outputs

    def _context_from_task(self, task: ProposalTask, state: GEPAState) -> ProposalContext:
        """Build a ProposalContext from a ProposalTask, firing sequential callbacks."""
        i = state.i + 1
        curr_prog_id = task.parent_idx
        curr_prog = task.parent_candidate
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

        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=i,
                minibatch_ids=task.minibatch_ids,
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
            subsample_ids=task.minibatch_ids,
            minibatch=task.minibatch,
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

            new_texts, prompts, raw_lm_outputs = self.propose_new_texts(
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
        )
        return ProposalOutput(proposal=proposal, total_evals=total_evals, trace_data=trace_data, cache_entry=cache_entry)

    def apply_proposal_output(self, output: ProposalOutput, state: GEPAState) -> None:
        """Apply deferred state updates from a proposal. Must be called sequentially."""
        state.increment_evals(output.total_evals)
        if output.cache_entry is not None and state.evaluation_cache is not None:
            candidate, ids, outputs, scores, obj_scores = output.cache_entry
            state.evaluation_cache.put_batch(candidate, ids, outputs, scores, obj_scores)

    def _mutate(
        self, tasks: list[ProposalTask], state: GEPAState[Any, DataId]
    ) -> list[CandidateProposal[DataId]]:
        """Run reflective mutation for each task, in parallel when len(tasks) > 1.

        All tasks belong to the same engine iteration (state.i is not advanced here).
        Fires sequential callbacks per task before parallel execution.
        """
        # Build contexts sequentially (fires callbacks that must run in order)
        contexts: list[ProposalContext] = [self._context_from_task(task, state) for task in tasks]

        # Update trace entry for this iteration (first task's data)
        if contexts:
            state.full_program_trace[-1]["selected_program_candidate"] = contexts[0].curr_prog_id
            state.full_program_trace[-1]["subsample_ids"] = contexts[0].subsample_ids

        # Execute proposals — parallel when multiple tasks
        raw_outputs: list[ProposalOutput | None] = [None] * len(contexts)
        if len(contexts) == 1:
            raw_outputs[0] = self.execute_proposal(contexts[0], state)
        else:
            with ThreadPoolExecutor(max_workers=len(contexts)) as executor:
                future_to_idx = {
                    executor.submit(self.execute_proposal, ctx, state): idx
                    for idx, ctx in enumerate(contexts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        raw_outputs[idx] = future.result()
                    except Exception as e:
                        self.logger.log(
                            f"Iteration {contexts[idx].iteration}: Parallel proposal failed: {e}\n"
                            + _traceback.format_exc()
                        )

        # Apply deferred state updates and collect proposals (no Nones)
        proposals: list[CandidateProposal[DataId]] = []
        for ctx, output in zip(contexts, raw_outputs, strict=False):
            if output is None:
                continue
            self.apply_proposal_output(output, state)
            state.full_program_trace[-1].update(output.trace_data)
            if output.proposal is not None:
                output.proposal.metadata["iteration"] = ctx.iteration
                proposals.append(output.proposal)

        return proposals
