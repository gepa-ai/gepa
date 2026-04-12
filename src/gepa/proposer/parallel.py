# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Batch-based parallel proposals for GEPA.

No ThreadPoolExecutor — all parallelism delegated to adapter.batch_evaluate()
and LM.batch_complete().
"""

from dataclasses import dataclass
from typing import Any, Protocol

from gepa.core.adapter import EvaluationBatch, default_batch_evaluate
from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.acceptance import AcceptanceCriterion
from gepa.strategies.batch_sampler import BatchSampler


@dataclass
class ProposalTask:
    """One (parent, minibatch) pair to propose from."""

    parent_idx: int
    parent_candidate: dict[str, str]
    minibatch_ids: list
    minibatch: list


@dataclass
class ParallelConfig:
    """Configuration for batch-based parallel proposals."""

    sampling_strategy: "ParallelSamplingStrategy"
    selection_strategy: "ProposalSelectionStrategy"


class ParallelSamplingStrategy(Protocol):
    """Protocol for generating N (parent, minibatch) tasks."""

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]: ...


class ProposalSelectionStrategy(Protocol):
    """Protocol for filtering proposals after evaluation."""

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]: ...


# --- Built-in sampling strategies ---


class SingleMutationSampling:
    """1 parent, 1 mutation — default GEPA behavior."""

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        parent_idx = candidate_selector.select_candidate_idx(state)
        mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
        return [ProposalTask(parent_idx, state.program_candidates[parent_idx], mb_ids, trainset.fetch(mb_ids))]


class SameParentSampling:
    """Best-of-N on the same parent with different minibatches."""

    def __init__(self, n: int):
        self.n = n

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        parent_idx = candidate_selector.select_candidate_idx(state)
        parent = state.program_candidates[parent_idx]
        tasks = []
        for _ in range(self.n):
            mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
            tasks.append(ProposalTask(parent_idx, parent, mb_ids, trainset.fetch(mb_ids)))
        return tasks


class IndependentSampling:
    """N different parents, each with their own minibatch."""

    def __init__(self, n: int):
        self.n = n

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        tasks = []
        for _ in range(self.n):
            parent_idx = candidate_selector.select_candidate_idx(state)
            mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
            tasks.append(ProposalTask(parent_idx, state.program_candidates[parent_idx], mb_ids, trainset.fetch(mb_ids)))
        return tasks


class PxNSampling:
    """P parents x N mutations each = P*N total tasks."""

    def __init__(self, p: int, n: int):
        self.p = p
        self.n = n

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader,
    ) -> list[ProposalTask]:
        tasks = []
        for _ in range(self.p):
            parent_idx = candidate_selector.select_candidate_idx(state)
            parent = state.program_candidates[parent_idx]
            for _ in range(self.n):
                mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
                tasks.append(ProposalTask(parent_idx, parent, mb_ids, trainset.fetch(mb_ids)))
        return tasks


# --- Built-in selection strategies ---


class AllImprovements:
    """Accept all proposals that pass the acceptance criterion."""

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]:
        return [p for p in proposals if acceptance_criterion.should_accept(p, state)]


class BestImprovement:
    """Accept only the single best-improving proposal."""

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]:
        best: CandidateProposal | None = None
        best_imp = float("-inf")
        for p in proposals:
            if not acceptance_criterion.should_accept(p, state):
                continue
            imp = sum(p.subsample_scores_after or []) - sum(p.subsample_scores_before or [])
            if imp > best_imp:
                best, best_imp = p, imp
        return [best] if best else []


class TopKImprovements:
    """Accept top-K proposals by improvement margin."""

    def __init__(self, k: int):
        self.k = k

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]:
        passing = [
            (sum(p.subsample_scores_after or []) - sum(p.subsample_scores_before or []), p)
            for p in proposals
            if acceptance_criterion.should_accept(p, state)
        ]
        passing.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in passing[: self.k]]


# --- Core batch pipeline ---


def propose_batch(
    proposer: Any,
    adapter: Any,
    state: GEPAState,
    sampling_strategy: ParallelSamplingStrategy,
    selection_strategy: ProposalSelectionStrategy,
    acceptance_criterion: AcceptanceCriterion,
    logger: Any,
    experiment_tracker: Any,
    callbacks: Any = None,
) -> list[CandidateProposal]:
    """4-stage batch pipeline. No ThreadPoolExecutor.

    Stage 1: Sample N tasks [sequential]
    Stage 2: Batch evaluate all parents with deduplication [adapter.batch_evaluate]
    Stage 3: Build reflective datasets + propose [sequential per task]
    Stage 4: Batch evaluate all children [adapter.batch_evaluate]
    Stage 5: Filter via selection_strategy
    """
    from gepa.core.state import _candidate_hash

    # Stage 1: Sample
    tasks = sampling_strategy.sample_tasks(
        state, proposer.candidate_selector, proposer.batch_sampler, proposer.trainset
    )
    if not tasks:
        return []

    # Stage 2: Batch evaluate parents (deduplicated)
    unique_keys: dict[tuple[str, tuple], tuple[dict[str, str], list]] = {}
    task_to_key: list[tuple[str, tuple]] = []
    for task in tasks:
        key = (_candidate_hash(task.parent_candidate), tuple(task.minibatch_ids))
        unique_keys.setdefault(key, (task.parent_candidate, task.minibatch))
        task_to_key.append(key)

    key_list = list(unique_keys.keys())
    items = [unique_keys[k] for k in key_list]

    batch_evaluate_fn = getattr(adapter, "batch_evaluate", None)
    if batch_evaluate_fn is not None:
        parent_evals: list[EvaluationBatch] = batch_evaluate_fn(items, capture_traces=True)
    else:
        parent_evals = default_batch_evaluate(adapter, items, capture_traces=True)

    key_to_eval: dict[tuple[str, tuple], EvaluationBatch] = dict(zip(key_list, parent_evals, strict=True))

    total_parent_evals = sum(
        e.num_metric_calls if e.num_metric_calls is not None else len(items[i][1]) for i, e in enumerate(parent_evals)
    )
    state.increment_evals(total_parent_evals)

    # Stage 3: Reflect + propose (sequential per task)
    children: list[tuple[ProposalTask, dict[str, str], EvaluationBatch, dict[str, Any]] | None] = []
    for task, key in zip(tasks, task_to_key, strict=True):
        eval_curr = key_to_eval[key]

        if not eval_curr.trajectories:
            children.append(None)
            continue

        if proposer.skip_perfect_score and proposer.perfect_score is not None and all(
            s is not None and s >= proposer.perfect_score for s in eval_curr.scores
        ):
            children.append(None)
            continue

        # Select components
        predictor_names = proposer.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, task.parent_idx, task.parent_candidate
        )

        # Build reflective dataset and propose
        try:
            reflective_dataset = adapter.make_reflective_dataset(task.parent_candidate, eval_curr, predictor_names)
            new_texts, prompts, raw_outputs = proposer.propose_new_texts(
                task.parent_candidate, reflective_dataset, predictor_names
            )

            _lm_metadata: dict[str, Any] = {}
            for comp in new_texts:
                _lm_metadata[f"prompt:{comp}"] = prompts.get(comp, "")
                _lm_metadata[f"raw_lm_output:{comp}"] = raw_outputs.get(comp, "")

            new_candidate = task.parent_candidate.copy()
            for name, text in new_texts.items():
                new_candidate[name] = text

            children.append((task, new_candidate, eval_curr, _lm_metadata))
        except Exception as e:
            logger.log(f"Parallel proposal failed for parent {task.parent_idx}: {e}")
            children.append(None)

    # Stage 4: Batch evaluate children
    valid_children = [(i, c) for i, c in enumerate(children) if c is not None]
    if not valid_children:
        return []

    child_items = [(c[1], c[0].minibatch) for _, c in valid_children]  # (new_candidate, minibatch)
    if batch_evaluate_fn is not None:
        child_evals: list[EvaluationBatch] = batch_evaluate_fn(child_items, capture_traces=True)
    else:
        child_evals = default_batch_evaluate(adapter, child_items, capture_traces=True)

    total_child_evals = sum(
        e.num_metric_calls if e.num_metric_calls is not None else len(child_items[i][1])
        for i, e in enumerate(child_evals)
    )
    state.increment_evals(total_child_evals)

    # Stage 5: Build proposals and filter
    proposals: list[CandidateProposal] = []
    for (_, (task, new_candidate, eval_curr, _lm_metadata)), child_eval in zip(valid_children, child_evals, strict=True):
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
            tag="parallel_mutation",
            metadata=_lm_metadata,
        )
        proposals.append(proposal)

    return selection_strategy.select(proposals, state, acceptance_criterion)
