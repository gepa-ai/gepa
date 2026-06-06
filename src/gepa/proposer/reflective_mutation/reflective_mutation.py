# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
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
from gepa.proposer.reflective_mutation.reflection_lm import ComBEEReflectionLM, StatelessReflectionLM
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature


@dataclass
class ProposalContext:
    """Pre-sampled context for a single proposal task.

    Created by :meth:`ReflectiveMutationProposer.prepare_proposal` (sequential),
    then consumed (in a batch) by :meth:`ReflectiveMutationProposer.execute_batch`.
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


@dataclass
class _PreReflect:
    """A task that has passed parent evaluation and is ready for reflection.

    Produced by :meth:`ReflectiveMutationProposer._execute_eval_and_prepare`
    and consumed (after the batched reflection step) by
    :meth:`ReflectiveMutationProposer._execute_after_reflection`. It carries the
    parent evaluation and reflective dataset so the reflection call for many
    tasks can be batched together between the two halves.
    """

    eval_curr: Any
    components: list[str]
    reflective_dataset: Any
    total_evals: int
    cache_entry: tuple | None
    trace_data: dict[str, Any]


class ReflectiveMutationProposer(ProposeNewCandidate[DataId]):
    """Implements the reflective mutation flow.

    Proposals are produced a batch at a time: :meth:`prepare_proposal` selects
    the (parent, minibatch) tasks sequentially, :meth:`execute_batch` runs the
    evaluate-reflect-evaluate pipeline over the whole batch (issuing the
    reflection LM calls together as one batched/concurrent request rather than
    via engine threads), and the engine applies acceptance sequentially. A batch
    of size one reproduces the original single-proposal behavior.
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
        use_combee: bool = False,
        combee_duplication_factor: int = 2,
        combee_aggregation_prompt: str | None = None,
        combee_rng: random.Random | None = None,
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

        self.reflection_prompt_template = reflection_prompt_template

        if isinstance(reflection_prompt_template, dict):
            for _param_name, template in reflection_prompt_template.items():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)

        # The reflection LM (#329).  When a custom adapter proposer or candidate
        # proposer overrides reflection, this stays None.  ComBEE is just a
        # different ReflectionLM implementation, swapped in here.
        self._reflection_lm: StatelessReflectionLM | None
        if reflection_lm is None:
            self._reflection_lm = None
        elif use_combee:
            self._reflection_lm = ComBEEReflectionLM(
                reflection_lm,
                reflection_prompt_template,
                aggregation_prompt_template=combee_aggregation_prompt,
                duplication_factor=combee_duplication_factor,
                rng=combee_rng,
                logger=logger,
            )
        else:
            self._reflection_lm = StatelessReflectionLM(reflection_lm, reflection_prompt_template, logger)

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

        if self._reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")

        # Delegate the stateless LM call(s) to the ReflectionLM (#329 Phase 1).
        # ``_next_lm`` is the (possibly extended) LM to use next; the stateless
        # default returns itself, so we ignore it here.  The reflection-LM pool
        # introduced in a later phase will route it back into the pool.
        proposal, _next_lm = self._reflection_lm.reflect(candidate, reflective_dataset, components_to_update)
        return proposal.new_texts, proposal.prompts, proposal.raw_lm_outputs

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

    def _execute_eval_and_prepare(self, ctx: ProposalContext, state: GEPAState) -> "ProposalOutput | _PreReflect":
        """Evaluate the parent and build the reflective dataset for one task.

        First half of the proposal pipeline. Returns a :class:`_PreReflect`
        ready for the (possibly batched) reflection call, or a skip
        :class:`ProposalOutput` (proposal=None) when the task should not proceed
        (no trajectories, all-perfect scores, or a reflective-dataset error).
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

        # 2) Decide which components to update. The engine is single-threaded
        # (proposals are vectorized, not threaded), so no lock is needed around
        # the module selector's internal state (e.g. RoundRobin counter).
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, ctx.curr_prog_id, ctx.curr_prog
        )

        # 3) Build reflective dataset and propose new content
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
                ctx.curr_prog, eval_curr, predictor_names_to_update
            )

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
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback

            self.logger.log(traceback.format_exc())
            return ProposalOutput(
                proposal=None, total_evals=total_evals, trace_data=trace_data, cache_entry=cache_entry
            )

        return _PreReflect(
            eval_curr=eval_curr,
            components=predictor_names_to_update,
            reflective_dataset=reflective_dataset,
            total_evals=total_evals,
            cache_entry=cache_entry,
            trace_data=trace_data,
        )

    def _execute_after_reflection(
        self,
        ctx: ProposalContext,
        pre: "_PreReflect",
        new_texts: dict[str, str],
        prompts: dict[str, str | list[dict[str, Any]]],
        raw_lm_outputs: dict[str, str],
        state: GEPAState,
    ) -> ProposalOutput:
        """Second half of the pipeline: build the child candidate and evaluate it."""
        i = ctx.iteration
        eval_curr = pre.eval_curr
        total_evals = pre.total_evals
        trace_data = pre.trace_data
        cache_entry = pre.cache_entry

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
        total_evals += (
            eval_after.num_metric_calls if eval_after.num_metric_calls is not None else len(ctx.subsample_ids)
        )

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
        self.experiment_tracker.log_metrics({"new_subsample_score": new_sum, "total_metric_calls": total_evals}, step=i)

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
        return ProposalOutput(
            proposal=proposal, total_evals=total_evals, trace_data=trace_data, cache_entry=cache_entry
        )

    def execute_proposal(self, ctx: ProposalContext, state: GEPAState) -> ProposalOutput:
        """Run the full evaluation + proposal pipeline for a single task.

        Equivalent to ``execute_batch([ctx], state)[0]``; retained for callers
        that drive one proposal at a time.
        """
        pre = self._execute_eval_and_prepare(ctx, state)
        if isinstance(pre, ProposalOutput):
            return pre
        try:
            new_texts, prompts, raw_lm_outputs = self.propose_new_texts(
                ctx.curr_prog, pre.reflective_dataset, pre.components
            )
        except Exception as e:
            self.logger.log(f"Iteration {ctx.iteration}: Exception during reflection/proposal: {e}")
            import traceback

            self.logger.log(traceback.format_exc())
            return ProposalOutput(
                proposal=None, total_evals=pre.total_evals, trace_data=pre.trace_data, cache_entry=pre.cache_entry
            )
        return self._execute_after_reflection(ctx, pre, new_texts, prompts, raw_lm_outputs, state)

    def _propose_texts_batch(
        self, jobs: list[tuple[dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]]]
    ) -> list[tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str]]]:
        """Propose new texts for many tasks, batching the reflection LM calls.

        Only the built-in stateless reflection path can be batched. When an
        adapter proposer or a custom proposer owns the call, fall back to one
        invocation per task (their batching, if any, is their concern).
        """
        if (
            self.adapter.propose_new_texts is not None
            or self.custom_candidate_proposer is not None
            or self._reflection_lm is None
        ):
            return [self.propose_new_texts(cand, refds, comps) for cand, refds, comps in jobs]

        results = self._reflection_lm.reflect_many(jobs)
        return [(proposal.new_texts, proposal.prompts, proposal.raw_lm_outputs) for proposal, _next_lm in results]

    def _propose_texts_batch_safe(
        self, jobs: list[tuple[dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]]]
    ) -> list[tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str]] | None]:
        """Like :meth:`_propose_texts_batch` but isolates per-task failures.

        Returns ``None`` in the slot of any task whose reflection raised, so one
        bad task does not sink the whole batch.
        """
        if not jobs:
            return []
        try:
            return list(self._propose_texts_batch(jobs))
        except Exception as e:
            self.logger.log(f"Batched reflection failed ({e}); retrying per task.")
            import traceback

            self.logger.log(traceback.format_exc())
            out: list[tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str]] | None] = []
            for cand, refds, comps in jobs:
                try:
                    out.append(self.propose_new_texts(cand, refds, comps))
                except Exception as e2:
                    self.logger.log(f"Per-task reflection failed: {e2}")
                    out.append(None)
            return out

    def execute_batch(self, contexts: list[ProposalContext], state: GEPAState) -> list[ProposalOutput]:
        """Run the proposal pipeline for a batch of tasks (vectorized).

        Evaluates all parents, builds reflective datasets, issues a single
        batched reflection across the surviving tasks, then evaluates all
        children — returning one :class:`ProposalOutput` per input context, in
        order. The engine applies acceptance sequentially afterwards, so
        ``GEPAState`` is never mutated concurrently. ``len(contexts) == 1``
        reproduces :meth:`execute_proposal` exactly.
        """
        outputs: list[ProposalOutput | None] = [None] * len(contexts)

        # Phase 1: evaluate parents + build reflective datasets per task.
        pres: list[_PreReflect | None] = []
        for idx, ctx in enumerate(contexts):
            result = self._execute_eval_and_prepare(ctx, state)
            if isinstance(result, ProposalOutput):
                outputs[idx] = result  # skipped (no trajectories / perfect / error)
                pres.append(None)
            else:
                pres.append(result)

        # Phase 2: one batched reflection across all tasks ready to reflect.
        ready_idxs = [idx for idx, pre in enumerate(pres) if pre is not None]
        jobs: list[tuple[dict[str, str], Mapping[str, Sequence[Mapping[str, Any]]], list[str]]] = []
        for idx in ready_idxs:
            pre = pres[idx]
            assert pre is not None
            jobs.append((contexts[idx].curr_prog, pre.reflective_dataset, pre.components))
        batch_texts = self._propose_texts_batch_safe(jobs)

        # Phase 3: build + evaluate children per task.
        for job_pos, idx in enumerate(ready_idxs):
            pre = pres[idx]
            assert pre is not None
            texts = batch_texts[job_pos]
            if texts is None:
                outputs[idx] = ProposalOutput(
                    proposal=None, total_evals=pre.total_evals, trace_data=pre.trace_data, cache_entry=pre.cache_entry
                )
                continue
            new_texts, prompts, raw_lm_outputs = texts
            outputs[idx] = self._execute_after_reflection(contexts[idx], pre, new_texts, prompts, raw_lm_outputs, state)

        # Every context slot is filled (skip or full output); order is preserved.
        return [out for out in outputs if out is not None]

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
        state.full_program_trace[-1].update(
            {
                "selected_program_candidate": ctx.curr_prog_id,
                "subsample_ids": ctx.subsample_ids,
            }
        )
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
