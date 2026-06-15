# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import (
    DataInst,
    EvaluationBatch,
    GEPAAdapter,
    ProposalFn,
    RolloutOutput,
    Trajectory,
    default_batch_evaluate,
)
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState, _candidate_hash
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM
from gepa.strategies.acceptance import AcceptanceCriterion, StrictImprovementAcceptance
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.proposal_sampling import ProposalTask, SamplingStrategy, SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements, SelectionStrategy


class ReflectiveMutationProposer:
    """Implements the reflective mutation flow.

    Each iteration, the proposer:

    1. Samples one or more (parent, minibatch) tasks via ``sampling_strategy``
    2. Batch-evaluates all parents (deduplicated)
    3. For each task: builds a reflective dataset and proposes new texts
    4. Batch-evaluates all children
    5. Filters proposals via ``selection_strategy`` + ``acceptance_criterion``

    With the default ``SingleMutationSampling``, this produces exactly one
    task per iteration — matching GEPA's original sequential behavior.
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
        acceptance_criterion: AcceptanceCriterion | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        selection_strategy: SelectionStrategy | None = None,
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
        self.acceptance_criterion: AcceptanceCriterion = acceptance_criterion or StrictImprovementAcceptance()
        self.sampling_strategy: SamplingStrategy = sampling_strategy or SingleMutationSampling()
        self.selection_strategy: SelectionStrategy = selection_strategy or AllImprovements()

        self.reflection_prompt_template = reflection_prompt_template

        if isinstance(reflection_prompt_template, dict):
            for _param_name, template in reflection_prompt_template.items():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)

        # Reflection LM (#329 Phase 1); None when an adapter/custom proposer owns reflection.
        self._reflection_lm: StatelessReflectionLM | None = (
            StatelessReflectionLM(reflection_lm, reflection_prompt_template, logger)
            if reflection_lm is not None
            else None
        )

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
            dict keyed by component name.
        """
        empty: dict[str, str | list[dict[str, Any]]] = {}
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update), empty, {}

        if self.custom_candidate_proposer is not None:
            return self.custom_candidate_proposer(candidate, reflective_dataset, components_to_update), empty, {}

        if self._reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")

        # Delegate to the ReflectionLM (#329 Phase 1). The stateless default
        # returns itself as next_lm, so we ignore it; a later phase pools it.
        proposal, _next_lm = self._reflection_lm.reflect(candidate, reflective_dataset, components_to_update)
        return proposal.new_texts, proposal.prompts, proposal.raw_lm_outputs

    # ------------------------------------------------------------------
    # Batch evaluate helper
    # ------------------------------------------------------------------

    def _batch_evaluate(self, items: list[tuple[dict[str, str], list]]) -> list[EvaluationBatch]:
        """Evaluate (candidate, batch) pairs via the adapter's batch_evaluate or fallback."""
        batch_fn = getattr(self.adapter, "batch_evaluate", None)
        if batch_fn is not None:
            return batch_fn(items)
        return default_batch_evaluate(self.adapter, items)

    # ------------------------------------------------------------------
    # Main proposal method
    # ------------------------------------------------------------------

    def propose(self, state: GEPAState) -> list[CandidateProposal]:
        """Run the reflective mutation pipeline and return accepted proposals.

        With the default ``SingleMutationSampling`` + ``AllImprovements``,
        this returns 0 or 1 proposals — identical to the original sequential
        behavior.
        """
        i = state.i + 1

        # Stage 1: Sample (parent, minibatch) tasks
        tasks = self.sampling_strategy.sample_tasks(state, self.candidate_selector, self.batch_sampler, self.trainset)
        if not tasks:
            return []

        # Fire callbacks for each sampled task
        for task in tasks:
            notify_callbacks(
                self.callbacks,
                "on_candidate_selected",
                CandidateSelectedEvent(
                    iteration=i,
                    candidate_idx=task.parent_idx,
                    candidate=task.parent_candidate,
                    score=state.program_full_scores_val_set[task.parent_idx],
                ),
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

        # Stage 2: Batch evaluate parents (deduplicated)
        unique_keys: dict[tuple[str, tuple], tuple[dict[str, str], list[Any]]] = {}
        task_to_key: list[tuple[str, tuple]] = []
        for task in tasks:
            key = (_candidate_hash(task.parent_candidate), tuple(task.minibatch_ids))
            unique_keys.setdefault(key, (task.parent_candidate, task.minibatch))
            task_to_key.append(key)

        key_list = list(unique_keys.keys())
        items = [unique_keys[k] for k in key_list]

        # Fire evaluation start callbacks for each task
        for task in tasks:
            notify_callbacks(
                self.callbacks,
                "on_evaluation_start",
                EvaluationStartEvent(
                    iteration=i,
                    candidate_idx=task.parent_idx,
                    batch_size=len(task.minibatch),
                    capture_traces=True,
                    parent_ids=[p for p in state.parent_program_for_candidate[task.parent_idx] if p is not None],
                    inputs=task.minibatch,
                    is_seed_candidate=task.parent_idx == 0,
                ),
            )

        parent_evals = self._batch_evaluate(items)
        key_to_eval: dict[tuple[str, tuple], EvaluationBatch] = dict(zip(key_list, parent_evals, strict=True))

        # Fire evaluation end callbacks for each task
        for task, key in zip(tasks, task_to_key, strict=True):
            eval_curr = key_to_eval[key]
            notify_callbacks(
                self.callbacks,
                "on_evaluation_end",
                EvaluationEndEvent(
                    iteration=i,
                    candidate_idx=task.parent_idx,
                    scores=eval_curr.scores,
                    has_trajectories=bool(eval_curr.trajectories),
                    parent_ids=[p for p in state.parent_program_for_candidate[task.parent_idx] if p is not None],
                    outputs=eval_curr.outputs,
                    trajectories=eval_curr.trajectories,
                    objective_scores=eval_curr.objective_scores,
                    is_seed_candidate=task.parent_idx == 0,
                ),
            )

        total_parent_evals = sum(
            e.num_metric_calls if e.num_metric_calls is not None else len(items[idx][1])
            for idx, e in enumerate(parent_evals)
        )
        state.increment_evals(total_parent_evals)

        # Update evaluation cache for parents
        if state.evaluation_cache is not None:
            for task, key in zip(tasks, task_to_key, strict=True):
                eval_curr = key_to_eval[key]
                objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
                state.evaluation_cache.put_batch(
                    task.parent_candidate,
                    task.minibatch_ids,
                    eval_curr.outputs,
                    eval_curr.scores,
                    objective_scores_list,
                )

        # Log first task's selection (for trace compatibility)
        first_task = tasks[0]
        state.full_program_trace[-1]["selected_program_candidate"] = first_task.parent_idx
        state.full_program_trace[-1]["subsample_ids"] = first_task.minibatch_ids
        self.logger.log(
            f"Iteration {i}: Selected program {first_task.parent_idx} "
            f"score: {state.program_full_scores_val_set[first_task.parent_idx]}"
        )

        self.experiment_tracker.log_metrics(
            {
                "iteration": i,
                "selected_program_candidate": first_task.parent_idx,
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        # Stage 3: Reflect + propose (per task)
        children: list[tuple[ProposalTask, dict[str, str], EvaluationBatch, dict[str, Any]] | None] = []
        for task, key in zip(tasks, task_to_key, strict=True):
            eval_curr = key_to_eval[key]

            if not eval_curr.trajectories:
                self.logger.log(f"Iteration {i}: No trajectories for parent {task.parent_idx}. Skipping.")
                children.append(None)
                continue

            if (
                self.skip_perfect_score
                and self.perfect_score is not None
                and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
            ):
                self.logger.log(f"Iteration {i}: All subsample scores perfect for parent {task.parent_idx}. Skipping.")
                children.append(None)
                continue

            predictor_names = self.module_selector(
                state, eval_curr.trajectories, eval_curr.scores, task.parent_idx, task.parent_candidate
            )

            try:
                reflective_dataset = self.adapter.make_reflective_dataset(
                    task.parent_candidate, eval_curr, predictor_names
                )
                reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                    k: [dict(item) for item in v] for k, v in reflective_dataset.items()
                }
                notify_callbacks(
                    self.callbacks,
                    "on_reflective_dataset_built",
                    ReflectiveDatasetBuiltEvent(
                        iteration=i,
                        candidate_idx=task.parent_idx,
                        components=predictor_names,
                        dataset=reflective_dataset_concrete,
                    ),
                )
                notify_callbacks(
                    self.callbacks,
                    "on_proposal_start",
                    ProposalStartEvent(
                        iteration=i,
                        parent_candidate=task.parent_candidate,
                        components=predictor_names,
                        reflective_dataset=reflective_dataset_concrete,
                    ),
                )

                new_texts, prompts, raw_outputs = self.propose_new_texts(
                    task.parent_candidate, reflective_dataset, predictor_names
                )

                notify_callbacks(
                    self.callbacks,
                    "on_proposal_end",
                    ProposalEndEvent(
                        iteration=i,
                        new_instructions=new_texts,
                        prompts=prompts,
                        raw_lm_outputs=raw_outputs,
                    ),
                )

                _lm_metadata: dict[str, Any] = {}
                for comp in new_texts:
                    _lm_metadata[f"prompt:{comp}"] = prompts.get(comp, "")
                    _lm_metadata[f"raw_lm_output:{comp}"] = raw_outputs.get(comp, "")

                for pname, text in new_texts.items():
                    self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")

                new_candidate = task.parent_candidate.copy()
                for name, text in new_texts.items():
                    assert name in new_candidate, f"{name} missing in candidate"
                    new_candidate[name] = text

                children.append((task, new_candidate, eval_curr, _lm_metadata))
            except Exception as e:
                self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
                self.logger.log(traceback.format_exc())
                children.append(None)

        # Stage 4: Batch evaluate children
        valid_children = [(idx, c) for idx, c in enumerate(children) if c is not None]
        if not valid_children:
            return []

        child_items = [(c[1], c[0].minibatch) for _, c in valid_children]
        child_evals = self._batch_evaluate(child_items)

        total_child_evals = sum(
            e.num_metric_calls if e.num_metric_calls is not None else len(child_items[idx][1])
            for idx, e in enumerate(child_evals)
        )
        state.increment_evals(total_child_evals)

        # Update evaluation cache for children
        if state.evaluation_cache is not None:
            for (_, (task, new_candidate, _, _)), child_eval in zip(valid_children, child_evals, strict=True):
                new_obj_scores = list(child_eval.objective_scores) if child_eval.objective_scores else None
                state.evaluation_cache.put_batch(
                    new_candidate, task.minibatch_ids, child_eval.outputs, child_eval.scores, new_obj_scores
                )

        # Log subsample scores for first task (trace compatibility)
        if valid_children:
            first_child_idx = valid_children[0][0]
            first_child = children[first_child_idx]
            if first_child is not None:
                state.full_program_trace[-1]["subsample_scores"] = key_to_eval[task_to_key[first_child_idx]].scores
                state.full_program_trace[-1]["new_subsample_scores"] = child_evals[0].scores

                subsample_before = sum(key_to_eval[task_to_key[first_child_idx]].scores)
                subsample_after = sum(child_evals[0].scores)
                self.experiment_tracker.log_metrics(
                    {
                        "subsample/before": subsample_before,
                        "subsample/after": subsample_after,
                        "total_metric_calls": state.total_num_evals,
                    },
                    step=i,
                )

        # Stage 5: Build proposals and filter
        proposals: list[CandidateProposal] = []
        for (_, (task, new_candidate, eval_curr, _lm_metadata)), child_eval in zip(
            valid_children, child_evals, strict=True
        ):
            proposal = CandidateProposal(
                candidate=new_candidate,
                parent_program_ids=[task.parent_idx],
                subsample_indices=task.minibatch_ids,
                subsample_scores_before=eval_curr.scores,
                subsample_scores_after=child_eval.scores,
                eval_before=SubsampleEvaluation(
                    scores=eval_curr.scores,
                    outputs=eval_curr.outputs,
                    objective_scores=list(eval_curr.objective_scores) if eval_curr.objective_scores else None,
                    trajectories=eval_curr.trajectories,
                ),
                eval_after=SubsampleEvaluation(
                    scores=child_eval.scores,
                    outputs=child_eval.outputs,
                    objective_scores=list(child_eval.objective_scores) if child_eval.objective_scores else None,
                    trajectories=child_eval.trajectories,
                ),
                tag="reflective_mutation",
                metadata=_lm_metadata,
            )
            proposals.append(proposal)

        return self.selection_strategy.select(proposals, state, self.acceptance_criterion)
