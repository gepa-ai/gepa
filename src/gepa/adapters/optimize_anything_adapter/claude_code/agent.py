"""Claude Code agentic orchestration helpers for optimize_anything."""

from __future__ import annotations

import json
import os
import random
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.adapters.optimize_anything_adapter.claude_code.prompts import build_note_update_prompt
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
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import CandidateSelector, LanguageModel
from gepa.strategies.batch_sampler import BatchSampler

if TYPE_CHECKING:
    from gepa.core.adapter import GEPAAdapter


_STR_CANDIDATE_KEY = "current_candidate"


def _extract_last_fenced_block(lm_out: str) -> str:
    """Extract content from the last ``` fenced block in LM output."""
    positions: list[int] = []
    for match in re.finditer(r"(?:^|\n)[ \t]*(```)", lm_out):
        positions.append(match.start(1))

    if len(positions) >= 2:
        start = positions[-2] + 3
        end = positions[-1]
        content = lm_out[start:end]
        match = re.match(r"^\S*\n", content)
        if match:
            content = content[match.end():]
        return content.strip()

    stripped = lm_out.strip()
    if stripped.startswith("```"):
        match = re.match(r"^```\S*\n?", lm_out)
        if match:
            return lm_out[match.end():].strip()
    elif stripped.endswith("```"):
        return stripped[:-3].strip()
    return stripped


class GlobalNote:
    """Append-only, thread-safe reflection journal stored at run_dir/note.md."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> str:
        return self.path.read_text() if self.path.exists() else ""

    def append(self, entry: str) -> None:
        with self._lock:
            with self.path.open("a") as file:
                file.write("\n" + entry.strip() + "\n")


class NoteUpdater:
    """Uses Claude Code to generate and append one reflection entry per iteration."""

    def __init__(self, global_note: GlobalNote, cc_lm: LanguageModel) -> None:
        self.global_note = global_note
        self.cc_lm = cc_lm

    def update(
        self,
        iteration: int,
        parent_candidate: str,
        child_candidate: str,
        example_ids: list[Any],
        parent_scores: list[float],
        child_scores: list[float],
        parent_outputs: list[Any],
        child_outputs: list[Any],
        parent_objective_scores: list[dict[str, float]] | None = None,
        child_objective_scores: list[dict[str, float]] | None = None,
        objective: str | None = None,
    ) -> None:
        current_note = self.global_note.read()
        prompt = build_note_update_prompt(
            current_note=current_note,
            iteration=iteration,
            parent_candidate=parent_candidate,
            child_candidate=child_candidate,
            example_ids=example_ids,
            parent_scores=parent_scores,
            child_scores=child_scores,
            parent_outputs=parent_outputs,
            child_outputs=child_outputs,
            parent_objective_scores=parent_objective_scores,
            child_objective_scores=child_objective_scores,
            objective=objective,
        )
        entry = self.cc_lm(prompt)
        self.global_note.append(entry)


class EvalRecorderCallback:
    """GEPACallback that writes training evaluation results to ``{run_dir}/evals/``."""

    def __init__(self, run_dir: str) -> None:
        self.evals_dir = os.path.join(run_dir, "evals")
        os.makedirs(self.evals_dir, exist_ok=True)
        self._proposed_counter = 0
        self._eval_counter = 0

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        candidate: dict[str, str] = event["candidate"]
        example_ids: list[Any] = event["example_ids"]
        scores: list[float] = event["scores"]
        outputs: list[Any] = event["outputs"]
        iteration: int = event["iteration"]
        candidate_idx = event["candidate_idx"]
        parent_ids = list(event["parent_ids"])

        eval_id = self._eval_counter
        self._eval_counter += 1

        if candidate_idx is not None:
            cand_dir_name = f"c{candidate_idx:05d}"
        else:
            cand_dir_name = f"proposed_{self._proposed_counter:05d}"
            self._proposed_counter += 1

        cand_dir = os.path.join(self.evals_dir, cand_dir_name)
        tasks_dir = os.path.join(cand_dir, "tasks")
        os.makedirs(tasks_dir, exist_ok=True)

        skill_path = os.path.join(cand_dir, "skill.md")
        if not os.path.exists(skill_path):
            _write_skill_md(skill_path, candidate)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        meta: dict[str, Any] = {
            "candidate_idx": candidate_idx,
            "parents": parent_ids,
            "iteration": iteration,
            "average_score": avg_score,
        }
        _write_json(os.path.join(cand_dir, "meta.json"), meta)

        for eid, score, output in zip(example_ids, scores, outputs, strict=False):
            task_data: dict[str, Any] = {
                "eval_id": eval_id,
                "candidate_idx": candidate_idx,
                "task_id": str(eid),
                "score": score,
                "iteration": iteration,
            }
            side_info = _extract_side_info(output)
            if side_info is not None:
                task_data["side_info"] = _make_json_serializable(side_info)
            task_id_str = f"{eid:05d}" if isinstance(eid, int) else str(eid)
            _write_json(os.path.join(tasks_dir, f"task_{task_id_str}_eval_{eval_id:05d}.json"), task_data)


class AgenticProposer:
    """ProposeNewCandidate implementation for Claude Code agentic mode."""

    def __init__(
        self,
        trainset: DataLoader,
        adapter: GEPAAdapter[Any, Any, Any] | Any,
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
        self.trainset = trainset
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

        if self.task_aligned:
            curr_prog_id, subsample_ids, reference_task_ids = self._task_aligned_select(state)
        else:
            curr_prog_id = self.candidate_selector.select_candidate_idx(state)
            subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
            reference_task_ids = []

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
            MinibatchSampledEvent(iteration=i, minibatch_ids=subsample_ids, trainset_size=len(self.trainset)),
        )

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

        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(score is not None and score >= self.perfect_score for score in eval_curr.scores)
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

        if len(curr_prog) != 1:
            raise ValueError(f"CC agentic mode requires a single-key candidate dict, got keys: {list(curr_prog.keys())}")
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
            ProposalStartEvent(iteration=i, parent_candidate=curr_prog, components=[candidate_key], reflective_dataset={}),
        )
        raw_reply = self.cc_lm(prompt)
        new_text = _extract_last_fenced_block(raw_reply)
        notify_callbacks(
            self.callbacks,
            "on_proposal_end",
            ProposalEndEvent(iteration=i, new_instructions={candidate_key: new_text}),
        )
        new_candidate = curr_prog.copy()
        new_candidate[candidate_key] = new_text

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

        def evaluator_fn(batch: Any, candidate: Any) -> tuple[list[Any], list[float], list[Any] | None]:
            result = self.adapter.evaluate(batch, candidate, capture_traces=False)
            return result.outputs, result.scores, list(result.objective_scores) if result.objective_scores else None

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

        state.full_program_trace[-1]["new_subsample_scores"] = new_scores
        try:
            self.note_updater.update(
                iteration=i,
                parent_candidate=parent_text,
                child_candidate=new_text,
                example_ids=list(subsample_ids),
                parent_scores=eval_curr.scores,
                child_scores=new_scores,
                parent_outputs=eval_curr.outputs,
                child_outputs=child_outputs,
                parent_objective_scores=list(eval_curr.objective_scores) if eval_curr.objective_scores else None,
                child_objective_scores=[objective_by_id[eid] for eid in subsample_ids] if objective_by_id else None,
                objective=self.objective,
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("NoteUpdater failed (iteration %d): %s", i, exc)

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=new_scores,
            tag="agentic_mutation",
        )

    def _task_aligned_select(self, state: GEPAState) -> tuple[int, list[Any], list[Any]]:  # type: ignore[type-arg]
        """Sample task T, find Pareto-best parent for T, sample reference tasks."""
        pareto_by_task = state.program_at_pareto_front_valset
        task_id = self.rng.choice(list(pareto_by_task.keys()))
        curr_prog_id = self.rng.choice(list(pareto_by_task[task_id]))
        other_tasks = [task for task in self.trainset.all_ids() if task != task_id]
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
    """Build the stdin prompt for the Claude Code mutation subprocess."""
    parent_avg = sum(parent_scores) / len(parent_scores) if parent_scores else 0.0
    is_multitask = target_task_id is not None

    sections = []
    if objective:
        sections.append(f"## Objective\n{objective}")
    if background:
        sections.append(f"## Background\n{background}")

    note_text = note.strip() if note.strip() else "(No entries yet — this is the first mutation.)"
    sections.append(f"## Reflection notes (accumulated insights)\n{note_text}")

    evals_dir = str(Path(run_dir) / "evals")
    eval_section = f"## Eval history\nEvals directory: `{evals_dir}/`\n"
    eval_section += (
        "Structure: `c{{idx}}/` for accepted candidates, `proposed_{{idx}}/` for rejected ones.\n"
        "Each contains `skill.md` (the candidate code), `meta.json` (score + parents), "
        "and `tasks/task_{{id}}_eval_{{num}}.json` (per-task results).\n"
        f"Current parent: `c{parent_idx:05d}/` (score: {parent_avg:.4f})\n\n"
        "You can freely explore any candidate directory to understand the full optimization history. "
        "Use RLM sub-agents to read files cheaply and in parallel."
    )
    if is_multitask:
        eval_section += f"\n\nTarget task ID: `{target_task_id}`\n"
        if reference_task_ids:
            ref_ids_str = ", ".join(f"`{task}`" for task in reference_task_ids)
            eval_section += (
                f"Reference task IDs (for cross-task inspiration, not eval): {ref_ids_str}\n"
                "Browse their eval files for patterns that might apply to the target task.\n"
            )
    sections.append(eval_section)
    sections.append(f"## Parent candidate (score: {parent_avg:.4f})\n```\n{parent_candidate}\n```")

    if is_multitask:
        instruction = (
            f"Improve the candidate for task `{target_task_id}`. "
            "Read relevant eval files (parent, rejected candidates, reference tasks) to understand what worked and what failed. "
            "Output your improved candidate in a fenced code block."
        )
    else:
        instruction = (
            "Explore the eval history as you see fit to understand the optimization landscape. "
            "Output an improved candidate in a fenced code block."
        )
    sections.append(f"## Your task\n{instruction}")

    return "\n\n".join(sections)


def _write_skill_md(path: str, candidate: dict[str, str]) -> None:
    lines: list[str] = []
    for key, value in candidate.items():
        lines.append(f"## {key}\n\n{value}\n")
    with open(path, "w") as file:
        file.write("\n".join(lines))


def _extract_side_info(output: Any) -> dict[str, Any] | None:
    if isinstance(output, tuple) and len(output) >= 3:
        side_info = output[2]
        if isinstance(side_info, dict):
            return side_info
    return None


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w") as file:
        json.dump(data, file, indent=2, default=str)


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, str | int | float | bool | type(None)):
        return obj
    return str(obj)
