# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol

from gepa.core.adapter import DataInst, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import BatchSampler

logger = logging.getLogger(__name__)


@dataclass
class MutationTask(Generic[DataId]):
    """Data specification for one parallel mutation worker."""

    parent_idx: int  # Index into state.program_candidates
    minibatch_ids: list[DataId]  # Training examples to evaluate on
    worker_id: int  # Ordinal for logging/tracing


class MinibatchPolicy(str, Enum):
    """How minibatches are assigned across tasks."""

    SHARED_PER_PARENT = "shared_per_parent"
    INDEPENDENT_PER_TASK = "independent_per_task"
    GLOBAL_SHARED = "global_shared"


class ParallelSamplingStrategy(Protocol[DataId]):
    """Protocol for generating mutation tasks."""

    def generate_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader[DataId, Any],
    ) -> list[MutationTask[DataId]]: ...


class ProposalSelectionStrategy(Protocol):
    """Protocol for selecting which proposals to keep."""

    def select(
        self,
        proposals: list[CandidateProposal],
    ) -> list[CandidateProposal]: ...


# --- Built-in Sampling Strategies ---


class PxNSampling:
    """Full generality: P parents x N mutations per parent."""

    def __init__(
        self,
        p: int,
        n: int,
        minibatch_policy: MinibatchPolicy | str = MinibatchPolicy.SHARED_PER_PARENT,
    ):
        self.p = p
        self.n = n
        self.minibatch_policy = MinibatchPolicy(minibatch_policy)

    def generate_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        trainset: DataLoader[DataId, Any],
    ) -> list[MutationTask[DataId]]:
        tasks: list[MutationTask] = []
        worker_id = 0

        # Generate a single global minibatch if policy is global_shared
        global_minibatch: list[DataId] | None = None
        if self.minibatch_policy == MinibatchPolicy.GLOBAL_SHARED:
            global_minibatch = batch_sampler.next_minibatch_ids(trainset, state)

        for _ in range(self.p):
            parent_idx = candidate_selector.select_candidate_idx(state)

            # Generate per-parent minibatch if shared_per_parent
            parent_minibatch: list[DataId] | None = None
            if self.minibatch_policy == MinibatchPolicy.SHARED_PER_PARENT:
                parent_minibatch = batch_sampler.next_minibatch_ids(trainset, state)

            for _ in range(self.n):
                if self.minibatch_policy == MinibatchPolicy.GLOBAL_SHARED:
                    assert global_minibatch is not None
                    mb = list(global_minibatch)
                elif self.minibatch_policy == MinibatchPolicy.SHARED_PER_PARENT:
                    assert parent_minibatch is not None
                    mb = list(parent_minibatch)
                else:
                    # independent_per_task
                    mb = batch_sampler.next_minibatch_ids(trainset, state)

                tasks.append(MutationTask(parent_idx=parent_idx, minibatch_ids=mb, worker_id=worker_id))
                worker_id += 1

        return tasks


class SingleMutationSampling(PxNSampling):
    """Current behavior: 1 parent, 1 mutation, 1 minibatch."""

    def __init__(self):
        super().__init__(p=1, n=1)


class SameParentSampling(PxNSampling):
    """Best-of-N on same parent with shared minibatch. LLM randomness provides variation."""

    def __init__(self, n: int):
        super().__init__(p=1, n=n, minibatch_policy=MinibatchPolicy.SHARED_PER_PARENT)


class IndependentSampling(PxNSampling):
    """Explore from N different frontier positions with independent minibatches."""

    def __init__(self, n: int):
        super().__init__(p=n, n=1, minibatch_policy=MinibatchPolicy.INDEPENDENT_PER_TASK)


# --- Built-in Selection Strategies ---


class AllImprovements:
    """Accept all proposals where sum(after) > sum(before)."""

    def select(self, proposals: list[CandidateProposal]) -> list[CandidateProposal]:
        return [
            p
            for p in proposals
            if p.subsample_scores_after is not None
            and p.subsample_scores_before is not None
            and sum(p.subsample_scores_after) > sum(p.subsample_scores_before)
        ]


class BestImprovement:
    """Accept only the proposal with the largest improvement margin."""

    def select(self, proposals: list[CandidateProposal]) -> list[CandidateProposal]:
        best: CandidateProposal | None = None
        best_improvement = 0.0
        for p in proposals:
            if p.subsample_scores_after is None or p.subsample_scores_before is None:
                continue
            improvement = sum(p.subsample_scores_after) - sum(p.subsample_scores_before)
            if improvement > best_improvement:
                best_improvement = improvement
                best = p
        return [best] if best is not None else []


class BestAbsolute:
    """Accept only the proposal with the highest absolute sum(after). Only accepts improvements."""

    def select(self, proposals: list[CandidateProposal]) -> list[CandidateProposal]:
        best: CandidateProposal | None = None
        best_score = float("-inf")
        for p in proposals:
            if p.subsample_scores_after is None or p.subsample_scores_before is None:
                continue
            if sum(p.subsample_scores_after) <= sum(p.subsample_scores_before):
                continue
            score = sum(p.subsample_scores_after)
            if score > best_score:
                best_score = score
                best = p
        return [best] if best is not None else []


class TopKImprovements:
    """Accept top K proposals by improvement margin."""

    def __init__(self, k: int):
        self.k = k

    def select(self, proposals: list[CandidateProposal]) -> list[CandidateProposal]:
        improvements: list[tuple[float, CandidateProposal]] = []
        for p in proposals:
            if p.subsample_scores_after is None or p.subsample_scores_before is None:
                continue
            improvement = sum(p.subsample_scores_after) - sum(p.subsample_scores_before)
            if improvement > 0:
                improvements.append((improvement, p))
        improvements.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in improvements[: self.k]]


# --- Orchestrator ---


@dataclass
class _TaskContext(Generic[DataId]):
    """Internal: holds precomputed context for a mutation task."""

    task: MutationTask[DataId]
    eval_batch: Any  # EvaluationBatch
    components: list[str]
    reflective_ds: Any  # Mapping[str, Sequence[Mapping[str, Any]]]


class ParallelMutationOrchestrator(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Coordinates parallel mutation attempts per iteration.

    Runs P*N mutation workers in parallel using ThreadPoolExecutor.
    Each worker calls the LLM to propose a mutation and evaluates the child candidate.

    Thread safety: adapter.evaluate() must be thread-safe for parallel mode.
    For non-thread-safe adapters, set max_workers=1 (sequential fallback).
    """

    def __init__(
        self,
        proposer: ReflectiveMutationProposer,
        sampling_strategy: ParallelSamplingStrategy,
        selection_strategy: ProposalSelectionStrategy,
        max_workers: int | None = None,
    ):
        self.proposer = proposer
        self.sampling_strategy = sampling_strategy
        self.selection_strategy = selection_strategy
        self.max_workers = max_workers

    def propose_batch(self, state: GEPAState) -> list[CandidateProposal]:
        """Generate parallel mutation proposals and filter them.

        Steps:
        1. Generate tasks via sampling_strategy
        2. Evaluate parents + build reflective datasets (sequential)
        3. LLM proposals + child evaluations (parallel via ThreadPoolExecutor)
        4. Aggregate eval counts and build CandidateProposal objects
        5. Filter via selection_strategy
        """
        # 1. Generate tasks
        tasks = self.sampling_strategy.generate_tasks(
            state,
            self.proposer.candidate_selector,
            self.proposer.batch_sampler,
            self.proposer.trainset,
        )
        if not tasks:
            return []

        # 2. Evaluate parents + build reflective datasets (SEQUENTIAL)
        task_contexts: list[_TaskContext] = []
        parent_eval_cache: dict[tuple, tuple] = {}  # (parent_idx, mb_key) -> (eval_batch, components, reflective_ds)
        total_parent_evals = 0

        for task in tasks:
            cache_key = (task.parent_idx, tuple(task.minibatch_ids))
            if cache_key not in parent_eval_cache:
                parent = state.program_candidates[task.parent_idx]
                minibatch = self.proposer.trainset.fetch(task.minibatch_ids)
                eval_batch = self.proposer.adapter.evaluate(minibatch, parent, capture_traces=True)
                total_parent_evals += len(task.minibatch_ids)

                assert eval_batch.trajectories is not None, "capture_traces=True must return trajectories"
                components = self.proposer.module_selector(
                    state, eval_batch.trajectories, eval_batch.scores, task.parent_idx, parent
                )
                reflective_ds = self.proposer.adapter.make_reflective_dataset(parent, eval_batch, components)
                parent_eval_cache[cache_key] = (eval_batch, components, reflective_ds)

            eval_batch, components, reflective_ds = parent_eval_cache[cache_key]
            task_contexts.append(
                _TaskContext(task=task, eval_batch=eval_batch, components=components, reflective_ds=reflective_ds)
            )

        state.increment_evals(total_parent_evals)

        # 3. Parallel: LLM proposals + child evaluations (THREAD-SAFE)
        def _worker(ctx: _TaskContext) -> tuple[MutationTask, Any, dict[str, str], Any]:
            """Runs in a thread: LLM call + child evaluation. No state mutation."""
            parent = state.program_candidates[ctx.task.parent_idx]

            # LLM proposal (the main bottleneck)
            new_texts = self.proposer.propose_new_texts(
                parent,
                ctx.reflective_ds,
                ctx.components,
            )

            # Build child candidate
            new_candidate = parent.copy()
            for name, text in new_texts.items():
                new_candidate[name] = text

            # Evaluate child (can also be expensive — hence parallel)
            minibatch = self.proposer.trainset.fetch(ctx.task.minibatch_ids)
            child_eval = self.proposer.adapter.evaluate(minibatch, new_candidate, capture_traces=False)

            return ctx.task, ctx.eval_batch, new_candidate, child_eval

        results: list[tuple[MutationTask, Any, dict[str, str], Any]] = []
        effective_workers = self.max_workers or len(task_contexts)

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_worker, ctx): ctx.task for ctx in task_contexts}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    task = futures[future]
                    logger.exception("Worker %d failed for parent %d", task.worker_id, task.parent_idx)

        # 4. Build proposals + aggregate eval counts (SEQUENTIAL — fast bookkeeping)
        proposals: list[CandidateProposal] = []
        child_evals = 0
        for task, eval_batch, new_candidate, child_eval in results:
            child_evals += len(task.minibatch_ids)

            proposals.append(
                CandidateProposal(
                    candidate=new_candidate,
                    parent_program_ids=[task.parent_idx],
                    subsample_indices=task.minibatch_ids,
                    subsample_scores_before=eval_batch.scores,
                    subsample_scores_after=child_eval.scores,
                    tag="parallel_mutation",
                    metadata={"worker_id": task.worker_id},
                )
            )

        state.increment_evals(child_evals)

        # 5. Filter
        return self.selection_strategy.select(proposals)
