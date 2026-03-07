"""AgenticProposer — ProposeNewCandidate implementation for CC agentic mode."""

from __future__ import annotations

import random
import re
from typing import Any

from gepa.adapters.optimize_anything_adapter.agent_mode.global_note import GlobalNote, NoteUpdater
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.optimize_anything import _STR_CANDIDATE_KEY
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import CandidateSelector, LanguageModel
from gepa.strategies.batch_sampler import BatchSampler


def _extract_last_fenced_block(lm_out: str) -> str:
    """Extract content from the last ``` fenced block in LM output.

    Only considers ``` fences that appear at the start of a line (after
    optional whitespace), ignoring inline backtick references like
    "output within the ``` blocks".
    """
    positions: list[int] = []
    for m in re.finditer(r"(?:^|\n)[ \t]*(```)", lm_out):
        positions.append(m.start(1))

    if len(positions) >= 2:
        start = positions[-2] + 3  # after the opening ```
        end = positions[-1]  # the closing ```
        content = lm_out[start:end]
        match = re.match(r"^\S*\n", content)
        if match:
            content = content[match.end():]
        return content.strip()

    # Fallback: single or no fences
    stripped = lm_out.strip()
    if stripped.startswith("```"):
        match = re.match(r"^```\S*\n?", lm_out)
        if match:
            return lm_out[match.end():].strip()
    elif stripped.endswith("```"):
        return stripped[:-3].strip()
    return stripped


class AgenticProposer:
    """
    ProposeNewCandidate implementation for CC agentic mode.

    Differences from ReflectiveMutationProposer:
    - No skill layer: CC reads note + parent directly → new candidate string
    - No make_reflective_dataset(): eval outputs are the ASI
    - Task-aligned selection for multi-task search mode
    - GlobalNote updated after each proposal via Anthropic API
    - CC subprocess uses RLM to read eval files selectively (no inline dumps)
    """

    def __init__(
        self,
        trainset: DataLoader,
        adapter: Any,  # OptimizeAnythingAdapter, typed as Any to avoid circular import
        cc_lm: LanguageModel,
        global_note: GlobalNote,
        note_updater: NoteUpdater,
        objective: str | None,
        background: str | None,
        run_dir: str,
        task_aligned: bool,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler,
        num_reference_tasks: int = 2,
        perfect_score: float | None = None,
        skip_perfect_score: bool = True,
        callbacks: list[GEPACallback] | None = None,
        logger: Any = None,
        experiment_tracker: Any = None,
        rng: random.Random | None = None,
    ) -> None:
        self.trainset = trainset  # GEPAEngine reads self.trainset
        self.adapter = adapter
        self.cc_lm = cc_lm
        self.global_note = global_note
        self.note_updater = note_updater
        self.objective = objective
        self.background = background
        self.run_dir = run_dir
        self.task_aligned = task_aligned
        self.candidate_selector = candidate_selector
        self.batch_sampler = batch_sampler
        self.num_reference_tasks = num_reference_tasks
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.callbacks = callbacks
        self.logger = logger
        self.experiment_tracker = experiment_tracker
        self.rng = rng or random.Random()

    def propose(self, state: GEPAState) -> CandidateProposal | None:  # type: ignore[type-arg]
        i = state.i + 1

        # STEP 1 — Select parent + minibatch
        if self.task_aligned:
            curr_prog_id, subsample_ids, reference_task_ids = self._task_aligned_select(state)
        else:
            curr_prog_id = self.candidate_selector.select_candidate_idx(state)
            subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
            reference_task_ids: list[Any] = []

        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        minibatch = self.trainset.fetch(subsample_ids)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
        is_seed_candidate = curr_prog_id == 0

        notify_callbacks(
            self.callbacks,
            "on_candidate_selected",
            CandidateSelectedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                candidate=curr_prog,
                score=state.program_full_scores_val_set[curr_prog_id],
            ),
        )
        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=i,
                minibatch_ids=subsample_ids,
                trainset_size=len(self.trainset),
            ),
        )

        # STEP 2 — Evaluate parent to get scores + outputs (ASI)
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                batch_size=len(minibatch),
                capture_traces=False,
                parent_ids=curr_parent_ids,
                inputs=minibatch,
                is_seed_candidate=is_seed_candidate,
            ),
        )
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=False)
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                candidate=curr_prog,
                example_ids=list(subsample_ids),
                scores=eval_curr.scores,
                has_trajectories=False,
                parent_ids=curr_parent_ids,
                outputs=eval_curr.outputs,
                trajectories=None,
                objective_scores=eval_curr.objective_scores,
                is_seed_candidate=is_seed_candidate,
            ),
        )

        # Early exit if perfect score (only when skip_perfect_score is enabled)
        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
        ):
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="all_scores_perfect",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        # STEP 3 — Build mutation prompt and call CC
        if len(curr_prog) != 1:
            raise ValueError(
                f"CC agentic mode requires a single-key candidate dict, got keys: {list(curr_prog.keys())}"
            )
        candidate_key = _STR_CANDIDATE_KEY if _STR_CANDIDATE_KEY in curr_prog else next(iter(curr_prog))
        parent_text = curr_prog[candidate_key]

        note = self.global_note.read()
        prompt = build_mutation_prompt(
            objective=self.objective,
            background=self.background,
            note=note,
            parent_candidate=parent_text,
            parent_scores=eval_curr.scores,
            parent_idx=curr_prog_id,
            run_dir=self.run_dir,
            target_task_id=subsample_ids[0] if self.task_aligned else None,
            reference_task_ids=reference_task_ids,
        )
        notify_callbacks(
            self.callbacks,
            "on_proposal_start",
            ProposalStartEvent(
                iteration=i,
                parent_candidate=curr_prog,
                components=[candidate_key],
                reflective_dataset={},
            ),
        )
        raw_reply = self.cc_lm(prompt)
        new_text = _extract_last_fenced_block(raw_reply)
        notify_callbacks(
            self.callbacks,
            "on_proposal_end",
            ProposalEndEvent(
                iteration=i,
                new_instructions={candidate_key: new_text},
            ),
        )
        new_candidate = curr_prog.copy()
        new_candidate[candidate_key] = new_text

        # STEP 4 — Evaluate child on same minibatch (with caching)
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=None,
                batch_size=len(minibatch),
                capture_traces=False,
                parent_ids=[curr_prog_id],
                inputs=minibatch,
                is_seed_candidate=False,
            ),
        )

        def evaluator_fn(b: Any, c: Any) -> tuple[list[Any], list[float], list[Any] | None]:
            r = self.adapter.evaluate(b, c, capture_traces=False)
            return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None

        outputs_by_id, scores_by_id, objective_by_id, actual_evals = state.cached_evaluate_full(
            new_candidate, subsample_ids, self.trainset.fetch, evaluator_fn
        )
        new_scores = [scores_by_id[eid] for eid in subsample_ids]
        child_outputs = [outputs_by_id[eid] for eid in subsample_ids]
        state.increment_evals(actual_evals)

        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=None,
                candidate=new_candidate,
                example_ids=list(subsample_ids),
                scores=new_scores,
                has_trajectories=False,
                parent_ids=[curr_prog_id],
                outputs=child_outputs,
                trajectories=None,
                objective_scores=[objective_by_id[eid] for eid in subsample_ids] if objective_by_id else None,
                is_seed_candidate=False,
            ),
        )

        # STEP 5 — Update note (always — even on regression)
        state.full_program_trace[-1]["new_subsample_scores"] = new_scores
        try:
            self.note_updater.update(
                iteration=i,
                parent_candidate=parent_text,
                child_candidate=new_text,
                parent_scores=eval_curr.scores,
                child_scores=new_scores,
                parent_outputs=eval_curr.outputs,
                child_outputs=child_outputs,
                objective=self.objective,
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("NoteUpdater failed (iteration %d): %s", i, e)

        # STEP 6 — Return proposal (engine handles accept/reject)
        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=new_scores,
            tag="agentic_mutation",
        )

    def _task_aligned_select(
        self, state: GEPAState  # type: ignore[type-arg]
    ) -> tuple[int, list[Any], list[Any]]:
        """Sample task T, find Pareto-best parent for T, sample reference tasks.

        Used in multi-task search mode only.
        """
        pareto_by_task = state.program_at_pareto_front_valset  # dict[DataId, set[int]]
        task_id = self.rng.choice(list(pareto_by_task.keys()))
        curr_prog_id = self.rng.choice(list(pareto_by_task[task_id]))
        other_tasks = [t for t in self.trainset.all_ids() if t != task_id]
        n_ref = min(self.num_reference_tasks, len(other_tasks))
        reference_task_ids = self.rng.sample(other_tasks, n_ref)
        return curr_prog_id, [task_id], reference_task_ids


def build_mutation_prompt(
    objective: str | None,
    background: str | None,
    note: str,
    parent_candidate: str,
    parent_scores: list[float],
    parent_idx: int,
    run_dir: str,
    target_task_id: DataId | None,
    reference_task_ids: list[DataId],
) -> str:
    """Build the mutation prompt (stdin) for the CC subprocess.

    Does NOT dump eval history inline — instead gives CC the directory paths
    so it can use RLM Haiku sub-agents to read what it needs.
    The system prompt (_CC_AGENTIC_MUTATION_SYSTEM_PROMPT) teaches CC how to use RLM.
    """
    parent_avg = sum(parent_scores) / len(parent_scores) if parent_scores else 0.0
    parent_dir = f"{run_dir}/evals/c{parent_idx:05d}"
    is_multitask = target_task_id is not None

    sections = []

    if objective:
        sections.append(f"## Objective\n{objective}")
    if background:
        sections.append(f"## Background\n{background}")

    note_text = note.strip() if note.strip() else "(No entries yet — this is the first mutation.)"
    sections.append(f"## Reflection notes (accumulated insights)\n{note_text}")

    eval_section = f"## Eval history\nParent directory: `{parent_dir}/tasks/`\n"
    eval_section += "Files: `task_{{id}}_eval_{{num}}.json` — use RLM to read relevant files.\n"
    if is_multitask:
        eval_section += f"\nTarget task ID: `{target_task_id}`\n"
        if reference_task_ids:
            ref_ids_str = ", ".join(f"`{t}`" for t in reference_task_ids)
            eval_section += (
                f"Reference task IDs (for cross-task inspiration, not eval): {ref_ids_str}\n"
                "Browse their eval files for patterns that might apply to the target task.\n"
            )
    sections.append(eval_section)

    sections.append(f"## Parent candidate (score: {parent_avg:.4f})\n```\n{parent_candidate}\n```")

    if is_multitask:
        instruction = (
            f"Improve the candidate for task `{target_task_id}`. "
            "Read the parent's eval file for that task to understand what went wrong. "
            "Optionally read reference task eval files for cross-task inspiration. "
            "Output your improved candidate in a fenced code block."
        )
    else:
        instruction = (
            "Read the parent's eval history to understand what went wrong. "
            "Output an improved candidate in a fenced code block."
        )
    sections.append(f"## Your task\n{instruction}")

    return "\n\n".join(sections)
