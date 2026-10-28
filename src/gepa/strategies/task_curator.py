# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Efficient task curation: scheduled revisits and one retained success per task."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Generic

from gepa.core.adapter import DataInst, EvaluationBatch
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.batch_sampler import BatchSamplerError, ReflectiveDatasetBuilder

_TASK_CURATOR_VIEW_KEY = "_task_curator_view"
_TASK_CURATOR_EVIDENCE_KEY = "task_curator_specific_info"
_EVIDENCE_EVENTS = frozenset({"evaluation_result", "mutation_comparison"})
_CURATOR_DECISION_EVENTS = frozenset({"curator_decision", "curator_anchor_decision"})
_TASK_STATUSES = frozenset(
    {
        "mastered",
        "normal_failure",
        "prompt_sensitive",
        "impossible",
        "no_optimization_signal",
        "flaky_evaluation",
        "infrastructure_error",
        "uncertain",
    }
)
_TASK_STATUS_GUIDANCE = (
    "Return task_status for both run and skip, based on the latest available executions: "
    "mastered (stable clean success), normal_failure (ordinary failure, potentially improvable), "
    "prompt_sensitive (prompt changes demonstrably affect behavior), impossible (missing essential capability), "
    "no_optimization_signal (relevant changes repeatedly yield no progress or new information), "
    "flaky_evaluation (inconsistent scores for similar behavior or explicit evaluator errors), "
    "infrastructure_error (execution infrastructure failure), or uncertain (unseen or insufficient evidence). "
    "An ordinary failure does not imply flaky evaluation or impossibility. Historical task_status values are "
    "curator judgments, not ground truth; reassess them against the latest evidence."
)


def _is_empty_reflection_signal(exc: Exception) -> bool:
    """Recognize DSPy's legacy no-material signal without importing DSPy.

    These builders raise plain Exception instead of returning an empty mapping.
    Match the *origin* as well as the exact type/message: a feedback callback or
    custom builder raising the same text is still an error. Never infer missing
    reflection material from an arbitrary adapter's trajectory schema.
    """
    if type(exc) is not Exception or str(exc) not in {
        "No valid predictions found for any module.",
        "No valid predictions found for program.",
    }:
        return False
    origin = exc.__traceback__
    if origin is None:
        return False
    while origin.tb_next is not None:
        origin = origin.tb_next
    frame = origin.tb_frame
    return frame.f_code.co_name == "make_reflective_dataset" and frame.f_globals.get("__name__") in {
        "gepa.adapters.dspy_adapter.dspy_adapter",
        "gepa.adapters.dspy_full_program_adapter.full_program_adapter",
        "dspy.teleprompt.gepa.gepa_utils",
    }


def _parse_task_status(payload: dict[str, Any]) -> str | None:
    # Old logs/responses have no classification; do not infer one from their score.
    if "task_status" not in payload:
        return None
    status = payload["task_status"]
    if not isinstance(status, str) or status not in _TASK_STATUSES:
        raise ValueError("invalid task_status")
    return status


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _task_view(task: Any) -> Any:
    if isinstance(task, dict) and _TASK_CURATOR_VIEW_KEY in task:
        return _json_safe(task[_TASK_CURATOR_VIEW_KEY])
    return _json_safe(task)


def _evaluator_feedback(trajectory: Any) -> Any:
    if not isinstance(trajectory, dict):
        return _json_safe(trajectory)

    curator_evidence = trajectory.get(_TASK_CURATOR_EVIDENCE_KEY)
    if curator_evidence is not None:
        return _json_safe(curator_evidence)

    # Optional legacy evidence schema; no benchmark implementation is needed.
    presentation_fields = ("Task", "Score", "JudgeFeedback", "Error", "EvalFailures", "ReferenceEvaluation")
    if not any(field in trajectory for field in presentation_fields):
        return _json_safe(trajectory)

    summary = {field: _json_safe(trajectory.get(field)) for field in presentation_fields}
    packet = trajectory.get("evaluation_evidence_specific_info")
    if isinstance(packet, dict):
        summary["Evaluation"] = {
            field: _json_safe(packet.get(field))
            for field in (
                "evaluation_status",
                "evaluation_types",
                "evaluation_feedback",
                "error",
                "trajectory_complete",
                "execution_stage",
            )
        }
    return summary


def _historical_success_state(trajectory: Any) -> Any:
    if isinstance(trajectory, dict) and trajectory.get(_TASK_CURATOR_EVIDENCE_KEY) is not None:
        return _json_safe(trajectory[_TASK_CURATOR_EVIDENCE_KEY])

    summary = _evaluator_feedback(trajectory)
    if not isinstance(summary, dict):
        return summary
    # Only project the legacy presentation format. Other adapters can supply
    # arbitrary evidence without having it replaced by empty legacy fields.
    presentation_fields = ("Task", "Score", "JudgeFeedback", "Error", "EvalFailures", "ReferenceEvaluation")
    if not isinstance(trajectory, dict) or not any(field in trajectory for field in presentation_fields):
        return summary

    evaluation = summary.get("Evaluation")
    evaluation_feedback = evaluation.get("evaluation_feedback") if isinstance(evaluation, dict) else None
    evaluator_feedback = summary.get("JudgeFeedback") or evaluation_feedback

    state = {
        "Task": summary.get("Task"),
        "Score": summary.get("Score"),
        "EvaluatorFeedback": _json_safe(evaluator_feedback),
        "ReferenceEvaluation": _json_safe(summary.get("ReferenceEvaluation")),
    }
    return state


def _history_row_without_output(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"task_id", "output"}}


def _is_clean_success(score: float, trajectory: Any) -> bool:
    if score != 1.0:
        return False
    if not isinstance(trajectory, dict):
        return True
    if trajectory.get("Error") or trajectory.get("error") or trajectory.get("EvalFailures"):
        return False
    packet = trajectory.get("evaluation_evidence_specific_info")
    if not isinstance(packet, dict):
        return True
    status = str(packet.get("evaluation_status", "")).strip().lower()
    if status and status != "success":
        return False
    if packet.get("trajectory_complete") is False:
        return False
    return not packet.get("error")


def _success_state_id(task_id: Any, iteration: int, source: str) -> str:
    encoded_task_id = json.dumps(_json_safe(task_id), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"task:{encoded_task_id}/iteration:{iteration}/{source}"


class TaskCuratorLMError(BatchSamplerError):
    """An LM callable can raise this for a fatal error that must stop optimization."""


class TaskCuratorContractError(BatchSamplerError):
    """Curator evidence or checkpoint state is inconsistent with the sampled tasks."""


@dataclass(frozen=True)
class _HistoryNode:
    """An append-only record; descendants share the prefix, never a mutable list."""

    previous: _HistoryNode | None
    record: dict[str, Any]


@dataclass
class _BranchState:
    history_heads: dict[Any, _HistoryNode | None] = field(default_factory=dict)
    success_states: dict[Any, dict[str, dict[str, Any]]] = field(default_factory=dict)
    retained_success: dict[Any, str] = field(default_factory=dict)
    next_run_epoch: dict[Any, int] = field(default_factory=dict)
    skip_types: dict[Any, str | None] = field(default_factory=dict)
    epoch: int = -1
    epoch_order: list[Any] = field(default_factory=list)
    position: int = 0
    selected_in_epoch: int = 0

    def fork(self) -> _BranchState:
        # Copy small indexes/cursors only. Evidence and history nodes are never
        # mutated after insertion, so no trajectory-sized copy is needed.
        return replace(
            self,
            history_heads=dict(self.history_heads),
            success_states={tid: dict(states) for tid, states in self.success_states.items()},
            retained_success=dict(self.retained_success),
            next_run_epoch=dict(self.next_run_epoch),
            skip_types=dict(self.skip_types),
            epoch_order=list(self.epoch_order),
        )


@dataclass(frozen=True)
class _CuratorDecision:
    should_run: bool
    reason: str = ""
    failure: dict[str, str] | None = None
    skip_epochs: int = 0
    cooldown: bool = False
    forced: bool = False
    skip_type: str | None = None
    task_status: str | None = None
    success_pool_update: dict[str, Any] | None = None


class TaskCuratorSampler(Generic[DataId, DataInst]):
    """Select tasks using an LLM and retain one success for failed-task reflection.

    Pass an instance as ``batch_sampler`` to ``gepa.optimize`` or to
    ``ReflectionConfig``. The optimizer supplies parent evaluations and final
    mutation outcomes, and adds retained successes to the reflective dataset.
    No adapter wrapper or benchmark-specific trajectory format is required.
    With anchors enabled, the curator calls the original reflective-dataset
    builder once per evaluated task and injects that task's retained success
    before merging the records. Adapters need no task-ID fields, but their
    builder must support singleton batches. Filtering, reordering, and multiple
    records per task are safe. Tasks with no reflective records are skipped,
    including DSPy's legacy no-material exceptions; other errors still raise.
    Their evaluation scores and history are preserved. No reflection is requested
    when the entire minibatch lacks material. Set ``use_reflection_anchors=False`` for builders
    that need whole-batch context; scheduling and history still work normally.

    ``history_window`` defaults to 3 execution units, with a before/after pair
    counting once. Older units keep outcomes and curator assessments. ``None`` keeps all
    units in detail. Full evidence and decisions remain in the GEPA trace for
    audit and resume; the window limits prompt detail, not stored evidence.

    Default extraction honors ``_task_curator_view`` on tasks and
    ``task_curator_specific_info`` on trajectories. Without a dedicated packet,
    legacy Task/Score/JudgeFeedback records use selected evaluation fields for
    history and a smaller success reference; other formats pass through. Raw
    rollout artifacts outside an explicit packet are not included in that view.
    These legacy field names are optional compatibility conventions; arbitrary
    task/trajectory formats work without benchmark packages or services.

    ``task_view``, ``evidence_extractor``, and ``success_extractor`` can override
    those views. A custom ``evidence_extractor`` also supplies success references
    unless ``success_extractor`` is set separately. Their outputs are converted
    to JSON-safe values. Evidence must include what the curator needs to judge
    correctness. The default ``is_success`` requires score == 1 and excludes
    explicit evaluation errors, non-success evaluation status, and incomplete
    evaluation packets. ``is_failure`` defaults to score <= 0. Both predicates
    receive the score and the original trajectory; override them for other metrics.
    Success snapshots and pool eligibility share this single success predicate.
    Missing trajectories cannot supply an anchor.

    A skip of N includes the current epoch. The task runs automatically at
    epoch + N without another LLM call. Small epoch-tail batches are allowed.
    Invalid LLM decisions conservatively run the task and preserve its anchor;
    set ``raise_on_error`` to stop optimization on any LLM/response error instead.
    An LM callable can raise ``TaskCuratorLMError`` for a fatal error (for example,
    exhausted quota); it always stops optimization, even with GEPA's
    ``raise_on_exception=False``.

    Each candidate owns a curriculum snapshot: task histories, retained anchors,
    cooldowns and an epoch cursor. Accepted children share immutable history
    prefixes with their parent and then advance independently. Returning to an
    older candidate restores its own curriculum, never a descendant's outcomes.
    A rejected proposal remains diagnostic evidence in its parent's history.
    ``history`` exposes a detached audit view of the active candidate.

    This sampler supports greedy and sequential Pareto search, with single-parent
    mutations and a stable task-ID universe. Multi-proposal sampling and
    multi-parent merge are rejected explicitly. Resume uses saved curator trace
    entries automatically. Create a new sampler per optimization run. Full-curator and heuristic anchor modes
    are intentionally not part of this implementation.
    """

    requires_sequential_sampling = True
    requires_single_parent = True

    def __init__(
        self,
        *,
        minibatch_size: int,
        lm: LanguageModel,
        logger: LoggerProtocol | None = None,
        max_workers: int = 16,
        seed: int = 0,
        history_window: int | None = 3,
        use_reflection_anchors: bool = True,
        task_view: Callable[[DataInst], Any] | None = None,
        evidence_extractor: Callable[[Any], Any] | None = None,
        success_extractor: Callable[[Any], Any] | None = None,
        is_success: Callable[[float, Any], bool] | None = None,
        is_failure: Callable[[float, Any], bool] | None = None,
        raise_on_error: bool = False,
    ) -> None:
        for name, value in (("minibatch_size", minibatch_size), ("max_workers", max_workers)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if history_window is not None and (
            isinstance(history_window, bool) or not isinstance(history_window, int) or history_window < 1
        ):
            raise ValueError("history_window must be a positive integer or None")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(use_reflection_anchors, bool) or not isinstance(raise_on_error, bool):
            raise ValueError("use_reflection_anchors and raise_on_error must be booleans")
        self.minibatch_size = minibatch_size
        self.lm = lm
        self.logger = logger
        self.max_workers = max_workers
        self.seed = seed
        self.history_window = history_window
        self.use_reflection_anchors = use_reflection_anchors
        self.task_view = task_view if task_view is not None else _task_view
        self.evidence_extractor = evidence_extractor if evidence_extractor is not None else _evaluator_feedback
        self.success_extractor = (
            success_extractor
            if success_extractor is not None
            else evidence_extractor
            if evidence_extractor is not None
            else _historical_success_state
        )
        self.is_success = is_success if is_success is not None else _is_clean_success
        self.is_failure = is_failure or (lambda score, trajectory: score <= 0)
        self.raise_on_error = raise_on_error
        self._branches: dict[int, _BranchState] = {0: _BranchState()}
        self._active_candidate_idx = 0
        self._iteration_parents: dict[int, int] = {}
        self._known_ids: tuple[DataId, ...] | None = None
        self._expected_batches: dict[int, tuple[DataId, ...]] = {}
        self._recorded_iterations: set[int] = set()
        self._restored = False

    @property
    def _branch(self) -> _BranchState:
        return self._branches[self._active_candidate_idx]

    def _task_history(self, task_id: DataId) -> list[dict[str, Any]]:
        records = []
        node = self._branch.history_heads[task_id]
        while node is not None:
            records.append(node.record)
            node = node.previous
        records.reverse()
        return records

    @property
    def history(self) -> dict[DataId, list[dict[str, Any]]]:
        """Detached audit view for the active candidate; changing it cannot alter a shared prefix."""
        return {tid: deepcopy(self._task_history(tid)) for tid in self._branch.history_heads}

    def _append_history(self, task_id: DataId, row: dict[str, Any]) -> None:
        self._branch.history_heads[task_id] = _HistoryNode(self._branch.history_heads[task_id], deepcopy(row))

    def _check_iteration_parent(self, iteration: int) -> int:
        parent = self._iteration_parents.get(iteration)
        if parent is None or parent != self._active_candidate_idx:
            raise TaskCuratorContractError("Curator evidence does not belong to the active sampled parent")
        return parent

    def _fork_candidate(self, candidate_idx: int) -> None:
        if isinstance(candidate_idx, bool) or not isinstance(candidate_idx, int) or candidate_idx < 1:
            raise TaskCuratorContractError("An accepted mutation needs an explicit new candidate ID")
        if candidate_idx in self._branches:
            raise TaskCuratorContractError("Curator candidate already exists")
        self._branches[candidate_idx] = self._branch.fork()
        self._active_candidate_idx = candidate_idx

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    def _settings(self) -> dict[str, Any]:
        return {
            "version": 8,
            "history_window": self.history_window,
            "minibatch_size": self.minibatch_size,
            "seed": self.seed,
            "use_reflection_anchors": self.use_reflection_anchors,
            "uses_default_success": self.is_success is _is_clean_success,
        }

    @staticmethod
    def _trace_entry(state: GEPAState, iteration: int) -> dict:
        trace_index = iteration - 1
        matches = [entry for entry in state.full_program_trace if entry.get("i") == trace_index]
        if len(matches) != 1:
            raise TaskCuratorContractError(
                f"Task curator iteration {iteration} requires exactly one trace entry with i={trace_index}; "
                f"found {len(matches)}."
            )
        return matches[0]

    def _set_or_validate_universe(self, loader: DataLoader[DataId, DataInst]) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise TaskCuratorContractError("Task curator cannot sample from an empty loader.")
        if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(loader):
            raise TaskCuratorContractError("Task curator requires a stable loader with unique IDs.")
        universe = tuple(all_ids)
        if self._known_ids is None:
            self._known_ids = universe
            self._branch.history_heads = dict.fromkeys(all_ids)
            self._branch.success_states = {task_id: {} for task_id in all_ids}
        elif universe != self._known_ids:
            raise TaskCuratorContractError("The task curator loader universe changed during the run.")
        return all_ids

    def _start_epoch(self, all_ids: Sequence[DataId]) -> None:
        # If every task is cooling down, jump to the first due rollout epoch.
        next_due = min(self._branch.next_run_epoch.get(task_id, self._branch.epoch + 1) for task_id in all_ids)
        self._branch.epoch = max(self._branch.epoch + 1, next_due)
        self._branch.epoch_order = list(all_ids)
        random.Random(self.seed + self._branch.epoch).shuffle(self._branch.epoch_order)
        self._branch.position = 0
        self._branch.selected_in_epoch = 0

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        text = fenced.group(1) if fenced else text
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            text = match.group(0) if match else text
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload

    def _extract_decision(self, payload: dict[str, Any]) -> _CuratorDecision:
        action = payload.get("decision")
        if not isinstance(action, str) or action.strip().lower() not in {"run", "skip"}:
            raise ValueError("expected decision='run'|'skip'")
        should_run = action.strip().lower() == "run"
        duration = payload.get("skip_epochs", 0 if should_run else 1)
        if isinstance(duration, bool) or not isinstance(duration, int):
            raise ValueError("skip_epochs must be an integer")
        if (should_run and duration != 0) or (not should_run and duration < 1):
            raise ValueError("skip_epochs must be 0 for run and positive for skip")
        category = payload.get("skip_type")
        if should_run and category is not None:
            raise ValueError("skip_type must be null for run")
        if not should_run and ("skip_epochs" in payload or "skip_type" in payload):
            if category not in ("mastered", "low_signal"):
                raise ValueError("skip_type must be mastered or low_signal for skip")
        return _CuratorDecision(
            should_run, skip_epochs=duration, skip_type=category, task_status=_parse_task_status(payload)
        )

    def _execution_units(self, task_id: DataId) -> list[dict[str, Any]]:
        """Deduplicate evaluations covered by a before/after comparison."""
        units: dict[tuple[Any, Any], dict[str, Any]] = {}
        for index, row in enumerate(self._task_history(task_id)):
            if row.get("event") not in _EVIDENCE_EVENTS:
                continue
            iteration = row.get("iteration")
            key = (row.get("epoch"), iteration if iteration is not None else ("row", index))
            previous = units.get(key)
            if previous is None or row.get("event") == "mutation_comparison":
                units[key] = row
        return list(units.values())

    def _history_for_prompt(self, task_id: DataId) -> list[dict[str, Any]]:
        rows = self._execution_units(task_id)
        return rows if self.history_window is None else rows[-self.history_window :]

    @staticmethod
    def _compact_outcome(row: dict[str, Any]) -> str:
        feedback = row.get("evaluator_feedback")
        feedback = feedback if isinstance(feedback, dict) else {}
        packet = feedback.get("Evaluation") or feedback.get("evaluation_evidence_specific_info") or {}
        packet = packet if isinstance(packet, dict) else {}
        status = str(packet.get("evaluation_status") or feedback.get("evaluation_status") or "").lower()
        error = feedback.get("Error") or feedback.get("error") or packet.get("error")
        if feedback.get("infrastructure_error") or status in {"timeout", "infrastructure_error"}:
            return "infrastructure_error"
        if error and any(
            word in str(error).lower() for word in ("timeout", "timed out", "connection", "reset", "network")
        ):
            return "infrastructure_error"
        # Actor failures are task failures, not evidence of a broken evaluator.
        if feedback.get("actor_error"):
            return "fail"
        # Preserve independently reported evaluation errors separately.
        if error or feedback.get("EvalFailures") or status in {"error", "invalid"}:
            return "error"
        if packet.get("trajectory_complete") is False:
            return "incomplete"
        score = row.get("score")
        if not isinstance(score, Real) or not math.isfinite(float(score)):
            return "unknown"
        if score == 1:
            return "success"
        if score == 0:
            return "fail"
        return f"score={score:g}"

    def _execution_assessments(self, task_id: DataId) -> dict[tuple[Any, Any], dict[str, Any]]:
        assessments = {}
        last_execution = None
        for row in self._task_history(task_id):
            if row.get("event") in _EVIDENCE_EVENTS:
                last_execution = (row.get("epoch"), row.get("iteration"))
            elif row.get("event") in _CURATOR_DECISION_EVENTS and row.get("task_status") is not None:
                if last_execution is not None:
                    assessments[last_execution] = {
                        "task_status": row["task_status"],
                        "status_epoch_id": row.get("epoch"),
                    }
        return assessments

    def _older_history_for_prompt(self, task_id: DataId) -> list[dict[str, Any]]:
        if self.history_window is None:
            return []
        summaries = []
        assessments = self._execution_assessments(task_id)
        for row in self._execution_units(task_id)[: -self.history_window]:
            summary = {"epoch_id": row.get("epoch"), "task_status": None}
            summary.update(assessments.get((row.get("epoch"), row.get("iteration")), {}))
            if row.get("event") == "mutation_comparison":
                summary.update(
                    before=self._compact_outcome(row.get("before", {})),
                    after=self._compact_outcome(row.get("after", {})),
                    mutation_accepted=row.get("mutation_accepted"),
                    before_candidate_idx=row.get("before", {}).get("candidate_idx"),
                    after_candidate_idx=row.get("after", {}).get("candidate_idx"),
                )
            else:
                summary["outcome"] = self._compact_outcome(row)
                summary["candidate_idx"] = row.get("candidate_idx")
            summaries.append(summary)
        return summaries

    def _history_section(self, task_id: DataId, history: list[dict[str, Any]]) -> str:
        older = self._older_history_for_prompt(task_id)
        compact = "\n".join(json.dumps(row, ensure_ascii=False) for row in older) if older else "[]"
        return (
            f"{self._history_label()}:\n{json.dumps(history, ensure_ascii=False, indent=2)}"
            f"\n\nOlder executions (epoch, outcomes, and curator-assigned task_status):\n{compact}"
        )

    def _prompt_view(self, task_id: DataId) -> list[dict[str, Any]]:
        # Retain the experimental alias for the current anchor only. Internal
        # task/iteration IDs and unselected successes are not exposed to the LM.
        all_ids = list(self._branch.success_states[task_id])
        retained_id = self._branch.retained_success.get(task_id) if self.use_reflection_anchors else None
        aliases = {retained_id: f"success-{all_ids.index(retained_id) + 1}"} if retained_id in all_ids else {}

        def clean_record(row: dict[str, Any]) -> dict[str, Any]:
            result = {}
            for key, item in row.items():
                if key in {"iteration", "iteration_id", "task_id", "reason", "anchor_reason"}:
                    continue
                if key == "success_state_id":
                    if item in aliases:
                        result[key] = aliases[item]
                    continue
                if row.get("event") == "mutation_comparison" and key in {"before", "after"}:
                    result[key] = clean_record(item)
                else:
                    # Only clean curator-owned records. Evaluator payloads may
                    # legitimately use the same keys for substantive evidence.
                    result["epoch_id" if key == "epoch" else key] = _json_safe(item)
            return result

        return [clean_record(row) for row in self._history_for_prompt(task_id)]

    def _history_label(self) -> str:
        window = "all" if self.history_window is None else str(self.history_window)
        return (
            f"Current candidate: {self._active_candidate_idx}. "
            f"Current epoch: {self._branch.epoch}. History: latest {window} execution units "
            "(before/after pair = 1), older executions retain epoch/outcomes/task_status, up to 1 success anchors; "
            "skip decisions omitted.\nHistory"
        )

    def _latest_pool_candidates(self, task_id: DataId) -> dict[str, str]:
        for row in reversed(self._task_history(task_id)):
            if row.get("event") not in _EVIDENCE_EVENTS:
                continue
            sides = (
                {side: row.get(side, {}) for side in ("before", "after")}
                if row["event"] == "mutation_comparison"
                else {"before": row}
            )
            return {
                side: evidence["success_state_id"]
                for side, evidence in sides.items()
                if evidence.get("success_state_id") in self._branch.success_states[task_id]
            }
        return {}

    def _decide(self, task_id: DataId, task: DataInst) -> tuple[DataId, _CuratorDecision]:
        prompt_history = self._prompt_view(task_id)
        return_instruction = (
            '\nReturn only JSON: {"task_status":"...","decision":"run|skip",'
            '"skip_epochs":0,"skip_type":null|"mastered"|"low_signal"}.\n\n'
        )
        prompt = (
            "You are a task curator allocating evaluation budget during text optimization. "
            "Judge this task independently: no other task is visible. Decide whether another rollout is currently useful "
            "for learning a better prompt or obtaining information about the task.\n\n"
            "Use these definitions:\n"
            "- Mastered: multiple stable, clean successes across separate rollouts. The agent must complete the task "
            "correctly, not merely benefit from evaluator luck, leaked state, or an accidentally correct final page. "
            "There must be no recent regression. One success is not sufficient evidence of mastery.\n"
            "- Prompt-sensitive: before/after evaluation pairs show that prompt changes can improve the behavior, "
            "including partial behavioral correction even when the binary score does not change. Run these tasks.\n"
            "- Operationally impossible: the available tools or UI genuinely lack the required capability, supported by "
            "the action schema, trajectory-derived diagnostics, reference evaluation, or evaluator evidence. "
            "A single unsupported actor claim is insufficient; repeated independent attempts consistently identifying "
            "the same missing essential capability are meaningful evidence.\n"
            "- No current optimization signal: the same failure mode persists across multiple materially relevant prompt "
            "mutations, with no behavioral progress or new diagnostic information. Do not infer this merely because several "
            "mutations failed when they did not address the observed failure.\n"
            "- Flaky evaluation: materially similar behavior and ending states repeatedly receive inconsistent outcomes "
            "without a credible relationship to prompt changes, or there is explicit evidence of an incorrect label, "
            "stale or contradictory target, or evaluator expectation inconsistent with the user request. Do not infer "
            "this from one disagreement, inconsistent actor behavior, or an unusually strict or difficult evaluator.\n"
            "- Infrastructure failure: timeout, reset, network errors, or local-state/file collisions caused by missing "
            "resets provide no evidence about prompt quality. Do not interpret them as agent failures.\n"
            "- Uncertain: run rather than skip. A rollout has information value: use it to test uncertain mastery, "
            "suspected flakiness, possible regressions, or ambiguous failure mechanisms.\n\n"
            "Run unseen or underexplored tasks and tasks with credible prompt-sensitive failures. Skip only when the "
            "history provides strong evidence that another rollout currently has little optimization or diagnostic value. "
            "A mutation_comparison records paired before/after evaluations even when the proposed mutation was rejected. "
            "Use an improved after result as evidence that the task may be prompt-sensitive, but when mutation_accepted is "
            "false, do not assume that behavior belongs to the current prompt or that the task is now mastered. "
            "For run, return skip_epochs=0 and skip_type=null. For skip, choose a positive integer skip_epochs=N "
            "and skip_type=mastered for stable mastery or low_signal for other reasons. N includes the current "
            "epoch: skipping at epoch E schedules a forced rollout at epoch E+N. No curator call is made during "
            "the wait or at expiry; the task runs automatically at expiry. Choose when another rollout will be "
            "useful, rather than repeatedly postponing that decision. A skip is not a permanent retirement."
            f"\n{_TASK_STATUS_GUIDANCE}"
            f"{return_instruction}"
            "Evaluate the value of another rollout under the current prompt, not only whether previous rollouts "
            "added new information. A persistent, concrete agent mistake can remain useful optimization "
            "evidence even when repeated. Repeated failure alone does not establish that targeted fixes have "
            "been tried or that the current prompt cannot improve. If a task was deferred while optimization "
            "continued on other tasks, allow a fresh rollout to test whether its behavior changed; do not treat "
            "an unchanged historical record during the skipped interval as new evidence against retrying. "
            "This "
            "does not override stable clean mastery or well-supported missing capability, evaluator defects, or "
            "infrastructure failure.\n\n"
            f"Task ID:\n{json.dumps(_json_safe(task_id), ensure_ascii=False)}\n\n"
            f"Task:\n{json.dumps(_json_safe(self.task_view(task)), ensure_ascii=False, indent=2)}\n\n"
            f"{self._history_section(task_id, prompt_history)}"
        )
        pool_candidates = self._latest_pool_candidates(task_id) if self.use_reflection_anchors else {}
        if self.use_reflection_anchors:
            retained_id = self._branch.retained_success.get(task_id)
            retained = self._branch.success_states[task_id].get(retained_id) if retained_id is not None else None
            prompt += "\n\nRetained success (automatically supplied to reflection on failure):\n" + json.dumps(
                retained["success_state"] if retained else None, ensure_ascii=False, indent=2
            )
            if pool_candidates:
                prompt += (
                    "\n\nThe latest execution unit contains successful rollouts. Choose a reference success for "
                    "future reflection from the retained success and the eligible successful rollouts in History. "
                    "Consider genuine task completion, reproducibility, and clear, efficient execution. "
                    "If a retained success exists, replace it only if another success provides a better reference. "
                    'Include "selected_success": "retained", "before", or "after" in your JSON response. '
                    '"retained" leaves the pool unchanged, including when it is empty. '
                    '"before" or "after" stores that successful trajectory. Only choose a side listed below; '
                    "failed sides are not eligible. A success remains eligible even if its mutation was rejected."
                    "\n\nEligible successes (references to the latest History entry; use its evidence above):\n"
                    + json.dumps(
                        {
                            side: (
                                f"History[-1].{side}.evaluator_feedback"
                                if prompt_history[-1].get("event") == "mutation_comparison"
                                else "History[-1].evaluator_feedback"
                            )
                            for side in pool_candidates
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        try:
            payload = self._parse_response(self.lm(prompt))
            decision = self._extract_decision(payload)
            selected = "retained"
            if pool_candidates:
                choice = payload.get("selected_success")
                if not isinstance(choice, str) or (choice != "retained" and choice not in pool_candidates):
                    raise ValueError("Missing, invalid, or unavailable selected_success")
                selected = choice
        except BatchSamplerError:
            raise
        except Exception as exc:
            if self.raise_on_error:
                raise TaskCuratorLMError(f"Task curator failed for task {task_id!r}: {exc}") from exc
            failure = {"error_type": type(exc).__name__, "error": str(exc)}
            self._log(
                f"Task curator decision failed for task {task_id!r}; running conservatively and preserving its retained success: "
                f"{failure['error_type']}: {failure['error']}"
            )
            return task_id, _CuratorDecision(
                should_run=True,
                reason="The curator decision failed, so the task is being run conservatively to preserve training coverage.",
                failure=failure,
            )

        if pool_candidates:
            old_id = self._branch.retained_success.get(task_id)
            if selected != "retained":
                self._branch.retained_success[task_id] = pool_candidates[selected]
            decision = replace(
                decision,
                success_pool_update={
                    "previous": old_id,
                    "candidates": pool_candidates,
                    "selected_success": selected,
                    "retained": self._branch.retained_success.get(task_id),
                    "selection_error": None,
                },
            )
        return task_id, decision

    def _decide_if_due(self, task_id: DataId, task: DataInst) -> tuple[DataId, _CuratorDecision]:
        due = self._branch.next_run_epoch.get(task_id)
        if due is None:
            return self._decide(task_id, task)
        category = self._branch.skip_types.get(task_id)
        if self._branch.epoch < due:
            return task_id, _CuratorDecision(
                False, skip_epochs=due - self._branch.epoch, cooldown=True, skip_type=category
            )
        return task_id, _CuratorDecision(True, forced=True, skip_type=category)

    def next_minibatch_ids(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
    ) -> list[DataId]:
        raise TaskCuratorContractError("Task curator requires candidate-aware sampling; supply the selected parent ID")

    def next_minibatch_ids_for_candidate(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
        *,
        candidate_idx: int,
    ) -> list[DataId]:
        if not self._restored:
            prior = [row for row in state.full_program_trace if row.get("i", -1) < state.i]
            self.load_from_trace(prior, loader, require_history=bool(prior))
        self.assert_no_unrecorded_through(state.i)
        all_ids = self._set_or_validate_universe(loader)
        iteration = state.i + 1
        if iteration in self._expected_batches or iteration in self._recorded_iterations:
            raise TaskCuratorContractError(f"Task curator iteration {iteration} was sampled more than once.")
        if isinstance(candidate_idx, bool) or not isinstance(candidate_idx, int) or candidate_idx not in self._branches:
            raise TaskCuratorContractError(f"No curator history for candidate {candidate_idx!r}")
        self._active_candidate_idx = candidate_idx
        self._iteration_parents[iteration] = candidate_idx
        selected: list[DataId] = []
        selected_set: set[DataId] = set()
        decisions: list[dict[str, Any]] = []
        considered_in_epoch = 0
        while len(selected) < self.minibatch_size:
            if not self._branch.epoch_order or self._branch.position >= len(self._branch.epoch_order):
                # Return a nonempty epoch tail without filling from the next epoch.
                if selected:
                    break
                if self._branch.epoch_order and self._branch.selected_in_epoch == 0:
                    self._log(
                        f"Task curator skipped every task in epoch {self._branch.epoch}; advancing to the next epoch."
                    )
                self._start_epoch(all_ids)

            while len(selected) < self.minibatch_size and self._branch.position < len(self._branch.epoch_order):
                needed = self.minibatch_size - len(selected)
                wave_ids: list[DataId] = []
                position_after_wave = self._branch.position
                while position_after_wave < len(self._branch.epoch_order) and len(wave_ids) < needed:
                    task_id = self._branch.epoch_order[position_after_wave]
                    position_after_wave += 1
                    if task_id not in selected_set:
                        wave_ids.append(task_id)
                if not wave_ids:
                    self._branch.position = position_after_wave
                    continue

                tasks = loader.fetch(wave_ids)
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(wave_ids))) as executor:
                    results = list(
                        executor.map(
                            lambda pair: self._decide_if_due(pair[0], pair[1]), zip(wave_ids, tasks, strict=True)
                        )
                    )
                self._branch.position = position_after_wave
                considered_in_epoch += len(wave_ids)

                for task_id, decision in results:
                    if decision.should_run:
                        self._branch.next_run_epoch.pop(task_id, None)
                        self._branch.skip_types.pop(task_id, None)
                    elif not decision.cooldown:
                        self._branch.next_run_epoch[task_id] = self._branch.epoch + decision.skip_epochs
                        self._branch.skip_types[task_id] = decision.skip_type
                    row = {
                        "event": (
                            "curator_cooldown"
                            if decision.cooldown
                            else "curator_forced_run"
                            if decision.forced
                            else "curator_decision"
                        ),
                        "epoch": self._branch.epoch,
                        "iteration": iteration,
                        "decision": "run" if decision.should_run else "skip",
                        "reason": decision.reason,
                        "skip_epochs": decision.skip_epochs,
                        "skip_type": decision.skip_type,
                        "task_status": decision.task_status,
                        "next_run_epoch": self._branch.next_run_epoch.get(task_id),
                    }
                    row["success_pool_update"] = decision.success_pool_update
                    if decision.failure is not None:
                        row["curator_failure"] = decision.failure
                    self._append_history(task_id, row)
                    decisions.append({"task_id": task_id, **row})
                    if decision.should_run:
                        selected.append(task_id)
                        selected_set.add(task_id)
                        self._branch.selected_in_epoch += 1

        self._expected_batches[iteration] = tuple(selected)
        trace_entry = self._trace_entry(state, iteration)
        trace_entry.update(
            {
                "task_curator_settings": self._settings(),
                "task_curator_parent_idx": candidate_idx,
                "task_curator_enabled": True,
                "task_curator_controls_sampling": True,
                "task_curator_epoch": self._branch.epoch,
                "task_curator_epoch_order": list(self._branch.epoch_order),
                "task_curator_position_after_sampling": self._branch.position,
                "task_curator_selected_in_epoch_after_sampling": self._branch.selected_in_epoch,
                "task_curator_considered_this_iteration": considered_in_epoch,
                "task_curator_calls_this_iteration": sum(row["event"] == "curator_decision" for row in decisions),
                "task_curator_decisions": decisions,
                "task_curator_sampled_ids": list(selected),
            }
        )
        return selected

    def _record_success_state(
        self,
        *,
        task_id: DataId,
        iteration: int,
        source: str,
        score: float,
        trajectory: Any,
        state: GEPAState,
        mutation_accepted: bool | None = None,
    ) -> str | None:
        if trajectory is None or not self.use_reflection_anchors or not self.is_success(score, trajectory):
            return None
        success_state_id = _success_state_id(task_id, iteration, source)
        snapshot = {
            "success_state_id": success_state_id,
            "task_id": task_id,
            "epoch": self._branch.epoch,
            "iteration": iteration,
            "source": source,
            "candidate_idx": self._active_candidate_idx if source == "before" or mutation_accepted else None,
            "score": float(score),
            "success_state": _json_safe(self.success_extractor(trajectory)),
        }
        if mutation_accepted is not None:
            snapshot["mutation_accepted"] = bool(mutation_accepted)
        existing = self._branch.success_states[task_id].get(success_state_id)
        if existing is not None and existing != snapshot:
            raise TaskCuratorContractError(f"Conflicting task-curator success state {success_state_id!r}.")
        if existing is None:
            self._branch.success_states[task_id][success_state_id] = snapshot
            trace_entry = self._trace_entry(state, iteration)
            trace_entry.setdefault("task_curator_success_states", []).append(snapshot)
        return success_state_id

    def record_first_pass(
        self,
        *,
        iteration: int,
        task_ids: Sequence[DataId],
        scores: Sequence[float],
        outputs: Sequence[Any],
        trajectories: Sequence[Any],
        state: GEPAState,
    ) -> None:
        parent_idx = self._check_iteration_parent(iteration)
        if iteration in self._recorded_iterations:
            raise TaskCuratorContractError(f"Task curator iteration {iteration} was recorded more than once.")
        expected = self._expected_batches.get(iteration)
        if expected is None or tuple(task_ids) != expected:
            raise TaskCuratorContractError(
                f"Task curator task association mismatch at iteration {iteration}: "
                f"expected {list(expected or ())!r}, received {list(task_ids)!r}."
            )
        if not (len(task_ids) == len(scores) == len(outputs) == len(trajectories)):
            raise TaskCuratorContractError(f"Task curator first-pass fields are misaligned at iteration {iteration}.")

        results = []
        for task_id, score, _output, trajectory in zip(task_ids, scores, outputs, trajectories, strict=True):
            if not isinstance(score, Real) or not math.isfinite(float(score)):
                raise TaskCuratorContractError(f"Task curator received invalid score for task {task_id!r}: {score!r}.")
            row = {
                "event": "evaluation_result",
                "candidate_idx": parent_idx,
                "epoch": self._branch.epoch,
                "iteration": iteration,
                "score": float(score),
                "evaluator_feedback": _json_safe(self.evidence_extractor(trajectory)),
            }
            success_state_id = self._record_success_state(
                task_id=task_id,
                iteration=iteration,
                source="before",
                score=float(score),
                trajectory=trajectory,
                state=state,
            )
            if success_state_id is not None:
                row["success_state_id"] = success_state_id
            self._append_history(task_id, row)
            results.append({"task_id": task_id, **row})

        self._recorded_iterations.add(iteration)
        del self._expected_batches[iteration]
        self._trace_entry(state, iteration)["task_curator_results"] = results

    def record_mutation_comparison(
        self,
        *,
        iteration: int,
        task_ids: Sequence[DataId],
        before_scores: Sequence[float],
        before_trajectories: Sequence[Any],
        after_scores: Sequence[float],
        after_trajectories: Sequence[Any],
        mutation_accepted: bool,
        state: GEPAState,
        candidate_idx: int | None = None,
    ) -> None:
        """Commit all child evidence together; failed extraction must not create a partial branch."""
        parent_idx = self._check_iteration_parent(iteration)
        parent = self._branch
        existing = set(self._branches)
        entry = self._trace_entry(state, iteration)
        previous_entry = dict(entry)
        if "task_curator_success_states" in entry:
            entry["task_curator_success_states"] = list(entry["task_curator_success_states"])
        self._branches[parent_idx] = parent.fork()
        try:
            self._record_mutation_comparison(
                iteration=iteration,
                task_ids=task_ids,
                before_scores=before_scores,
                before_trajectories=before_trajectories,
                after_scores=after_scores,
                after_trajectories=after_trajectories,
                mutation_accepted=mutation_accepted,
                state=state,
                candidate_idx=candidate_idx,
            )
        except Exception as exc:
            for idx in set(self._branches) - existing:
                del self._branches[idx]
            self._branches[parent_idx] = parent
            self._active_candidate_idx = parent_idx
            entry.clear()
            entry.update(previous_entry)
            if isinstance(exc, TaskCuratorContractError):
                raise
            raise TaskCuratorContractError("Failed to record curator mutation evidence") from exc
        if mutation_accepted:
            self._branches[parent_idx] = parent

    def _record_mutation_comparison(
        self,
        *,
        iteration: int,
        task_ids: Sequence[DataId],
        before_scores: Sequence[float],
        before_trajectories: Sequence[Any],
        after_scores: Sequence[float],
        after_trajectories: Sequence[Any],
        mutation_accepted: bool,
        state: GEPAState,
        candidate_idx: int | None,
    ) -> None:
        parent_idx = self._check_iteration_parent(iteration)
        trace_entry = self._trace_entry(state, iteration)
        if "task_curator_mutation_results" in trace_entry:
            raise TaskCuratorContractError("Mutation comparison was recorded more than once")
        if iteration not in self._recorded_iterations:
            raise TaskCuratorContractError("Mutation comparison requires a recorded parent evaluation")
        sampled_ids = trace_entry.get("task_curator_sampled_ids")
        if list(task_ids) != sampled_ids:
            raise TaskCuratorContractError(
                f"Task curator mutation comparison mismatch at iteration {iteration}: "
                f"expected {sampled_ids!r}, received {list(task_ids)!r}."
            )
        if not (
            len(task_ids)
            == len(before_scores)
            == len(before_trajectories)
            == len(after_scores)
            == len(after_trajectories)
        ):
            raise TaskCuratorContractError(
                f"Task curator mutation comparison fields are misaligned at iteration {iteration}."
            )

        if any(
            not isinstance(score, Real) or not math.isfinite(float(score)) for score in (*before_scores, *after_scores)
        ):
            raise TaskCuratorContractError("Task curator received invalid mutation-comparison scores")
        if mutation_accepted:
            if candidate_idx is None:
                raise TaskCuratorContractError("An accepted mutation needs an explicit new candidate ID")
            self._fork_candidate(candidate_idx)
        elif candidate_idx is not None:
            raise TaskCuratorContractError("A rejected mutation cannot have an accepted candidate ID")
        trace_entry["task_curator_child_idx"] = candidate_idx

        before_sum = sum(float(score) for score in before_scores)
        after_sum = sum(float(score) for score in after_scores)
        results = []
        for task_id, before_score, before_trajectory, after_score, after_trajectory in zip(
            task_ids,
            before_scores,
            before_trajectories,
            after_scores,
            after_trajectories,
            strict=True,
        ):
            row = {
                "event": "mutation_comparison",
                "epoch": self._branch.epoch,
                "iteration": iteration,
                "mutation_accepted": bool(mutation_accepted),
                "batch_score_before": before_sum,
                "batch_score_after": after_sum,
                "before": {
                    "candidate_idx": parent_idx,
                    "score": float(before_score),
                    "evaluator_feedback": _json_safe(self.evidence_extractor(before_trajectory)),
                },
                "after": {
                    "candidate_idx": candidate_idx,
                    "score": float(after_score),
                    "evaluator_feedback": _json_safe(self.evidence_extractor(after_trajectory)),
                },
            }
            before_id = _success_state_id(task_id, iteration, "before")
            if before_id in self._branch.success_states[task_id]:
                row["before"]["success_state_id"] = before_id
            after_id = self._record_success_state(
                task_id=task_id,
                iteration=iteration,
                source="after",
                score=float(after_score),
                trajectory=after_trajectory,
                state=state,
                mutation_accepted=mutation_accepted,
            )
            if after_id is not None:
                row["after"]["success_state_id"] = after_id
            self._append_history(task_id, row)
            results.append({"task_id": task_id, **row})

        trace_entry["task_curator_mutation_results"] = results

    def assert_no_unrecorded_through(self, iteration: int) -> None:
        missing = sorted(candidate for candidate in self._expected_batches if candidate <= iteration)
        if missing:
            raise TaskCuratorContractError(
                "Task curator sampled minibatches without first-pass results for iteration(s): "
                + ", ".join(str(candidate) for candidate in missing)
            )

    def observe_evaluation(
        self,
        *,
        iteration: int,
        task_ids: Sequence[DataId],
        evaluation: EvaluationBatch,
        state: GEPAState,
    ) -> None:
        """Record parent evidence even if reflection is skipped or fails."""
        self.record_first_pass(
            iteration=iteration,
            task_ids=task_ids,
            scores=evaluation.scores,
            outputs=evaluation.outputs,
            trajectories=evaluation.trajectories or [None] * len(task_ids),
            state=state,
        )

    def observe_proposal(
        self,
        *,
        iteration: int,
        proposal: CandidateProposal,
        accepted: bool,
        state: GEPAState,
        candidate_idx: int | None = None,
    ) -> None:
        """Record paired evidence with the engine's final selection outcome."""
        before, after = proposal.eval_before, proposal.eval_after
        if before is None or after is None or proposal.subsample_indices is None:
            raise TaskCuratorContractError("Curator needs before/after evaluations and task IDs")
        parent_idx = self._check_iteration_parent(iteration)
        if proposal.parent_program_ids != [parent_idx]:
            raise TaskCuratorContractError("Proposal parents do not match the curator's sampled parent")
        ids = proposal.subsample_indices
        self.record_mutation_comparison(
            iteration=iteration,
            task_ids=ids,
            before_scores=before.scores,
            before_trajectories=before.trajectories or [None] * len(ids),
            after_scores=after.scores,
            after_trajectories=after.trajectories or [None] * len(ids),
            mutation_accepted=accepted,
            state=state,
            candidate_idx=candidate_idx,
        )

    def build_reflective_dataset(
        self,
        *,
        candidate: dict[str, str],
        task_ids: Sequence[DataId],
        evaluation: EvaluationBatch,
        components_to_update: list[str],
        make_reflective_dataset: ReflectiveDatasetBuilder,
        state: GEPAState,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build and enrich each task's records before merging the minibatch.

        The adapter sees one already-evaluated task at a time. Every record it
        returns belongs to that task, even when it filters, reorders, or expands
        records. Identity stays here, outside adapter payloads. This does not
        re-evaluate tasks or call the reflection LM.

        Anchor-enabled reflection requires a builder that supports singleton
        batches and returns only evidence for the supplied task. Empty results
        and known DSPy no-material signals skip that task's reflection only.
        Other exceptions propagate as contract errors with task context.
        Batch-dependent builders should use
        use_reflection_anchors=False to preserve whole-batch assembly. Splitting
        can also change RNG/callback order in stateful builders.
        """
        self._check_iteration_parent(state.i + 1)
        if not self.use_reflection_anchors:
            return make_reflective_dataset(candidate, evaluation, components_to_update)
        for name in ("scores", "outputs", "trajectories", "objective_scores"):
            values = getattr(evaluation, name)
            if values is not None and len(values) != len(task_ids):
                raise TaskCuratorContractError(f"Reflection task IDs and {name} do not align")
        if len(set(task_ids)) != len(task_ids):
            raise TaskCuratorContractError("Reflection task IDs must be unique")

        enriched: dict[str, list[dict[str, Any]]] = {}
        injected_task_ids = set()
        skipped_task_ids = []
        for index, tid in enumerate(task_ids):
            single = replace(
                evaluation,
                outputs=[evaluation.outputs[index]],
                scores=[evaluation.scores[index]],
                trajectories=None if evaluation.trajectories is None else [evaluation.trajectories[index]],
                objective_scores=None if evaluation.objective_scores is None else [evaluation.objective_scores[index]],
                # This is a batch aggregate; no per-task allocation is known.
                num_metric_calls=None,
            )
            try:
                dataset = make_reflective_dataset(candidate, single, components_to_update)
            except Exception as exc:
                if not _is_empty_reflection_signal(exc):
                    raise TaskCuratorContractError(
                        f"Cannot build single-task reflection for task {tid!r}: {exc}. "
                        "Curator anchors require a builder that supports singleton batches; "
                        "use use_reflection_anchors=False for whole-batch assembly."
                    ) from exc
                dataset = {}

            if not any(len(records) for records in dataset.values()):
                skipped_task_ids.append(tid)
                self._log(f"Task {tid!r}: no reflection material; skipping reflection, retaining evaluation history.")
                continue

            sid = self._branch.retained_success.get(tid)
            trajectory = single.trajectories[0] if single.trajectories is not None else None
            anchor = None
            if sid is not None and self.is_failure(single.scores[0], trajectory):
                anchor = {
                    "instruction": (
                        "This is a previously observed successful evaluator state for the same task. Use it as a "
                        "contrastive reference to identify how the current behavior or outcome differs from the past "
                        "success and what needs to be recovered. The past success is evidence, not proof of correct "
                        "behavior, and may reflect stochasticity or evaluator leniency. Treat it critically: draw only "
                        "conclusions supported by the contrast and prefer generalizable guidance over task-specific or "
                        "overfitted rules."
                    ),
                    "successful_state": _json_safe(self._branch.success_states[tid][sid]["success_state"]),
                }
            for component, examples in dataset.items():
                if not examples:
                    continue
                records = enriched.setdefault(component, [])
                for example in examples:
                    # A builder may reuse its buffers across calls/components.
                    # Snapshot before the next task can overwrite that evidence.
                    record = deepcopy(dict(example))
                    if anchor is not None:
                        record["HistoricalSuccess"] = _json_safe(anchor)
                        injected_task_ids.add(tid)
                    records.append(record)

        entry = self._trace_entry(state, state.i + 1)
        entry["task_curator_reflection_skips"] = [
            {"task_id": tid, "reason": "no_reflection_material"} for tid in skipped_task_ids
        ]
        entry["task_curator_reflection_anchors"] = [
            {"task_id": tid, "success_state_id": self._branch.retained_success[tid], "supporting_success_ids": []}
            for tid in task_ids
            if tid in injected_task_ids
        ]
        return enriched

    def load_from_trace(
        self,
        trace: Sequence[dict],
        loader: DataLoader[DataId, DataInst],
        *,
        require_history: bool,
    ) -> None:
        """Replay candidate-scoped events; never reinterpret a legacy global history as branches."""
        self._set_or_validate_universe(loader)
        entries = [entry for entry in trace if entry.get("task_curator_enabled")]
        if not entries and require_history:
            raise TaskCuratorContractError("Cannot resume curator from a run without curator history")
        self._branches = {
            0: _BranchState(
                history_heads=dict.fromkeys(self._known_ids or ()),
                success_states={tid: {} for tid in self._known_ids or ()},
            )
        }
        self._active_candidate_idx = 0
        self._iteration_parents.clear()
        self._recorded_iterations.clear()
        self._expected_batches.clear()
        self._restored = False
        for entry in entries:
            if entry.get("task_curator_settings") != self._settings():
                raise TaskCuratorContractError("Curator settings changed on resume (including history format)")
            iteration = entry["i"] + 1
            if iteration in self._recorded_iterations:
                raise TaskCuratorContractError("Duplicate curator iteration in checkpoint")
            parent = entry.get("task_curator_parent_idx")
            if isinstance(parent, bool) or not isinstance(parent, int) or parent not in self._branches:
                raise TaskCuratorContractError("Checkpoint references unknown curator parent")
            if entry.get("selected_program_candidate", parent) != parent:
                raise TaskCuratorContractError("Curator parent differs from selected candidate")
            self._active_candidate_idx = parent
            self._iteration_parents[iteration] = parent
            for row in entry["task_curator_decisions"]:
                tid = row["task_id"]
                if tid not in self._branch.history_heads:
                    raise TaskCuratorContractError("Checkpoint references unknown task")
                update = row.get("success_pool_update")
                if update:
                    old = self._branch.retained_success.get(tid)
                    choices = update["candidates"]
                    choice = update["selected_success"]
                    if (
                        update["previous"] != old
                        or any(sid not in self._branch.success_states[tid] for sid in choices.values())
                        or (choice != "retained" and choice not in choices)
                    ):
                        raise TaskCuratorContractError("Invalid retained success predecessor or candidates")
                    expected = old if choice == "retained" else choices[choice]
                    if update["retained"] != expected:
                        raise TaskCuratorContractError("Invalid retained success result")
                    if expected is not None:
                        self._branch.retained_success[tid] = expected
                if row["decision"] == "skip":
                    due = row["epoch"] + row["skip_epochs"]
                    if row["skip_epochs"] < 1 or row["next_run_epoch"] != due:
                        raise TaskCuratorContractError("Invalid cooldown in checkpoint")
                    self._branch.next_run_epoch[tid] = due
                    self._branch.skip_types[tid] = row["skip_type"]
                else:
                    self._branch.next_run_epoch.pop(tid, None)
                    self._branch.skip_types.pop(tid, None)
                self._append_history(tid, _history_row_without_output(row))

            results = entry.get("task_curator_results")
            ids = entry["task_curator_sampled_ids"]
            if not isinstance(results, list) or [r["task_id"] for r in results] != ids:
                raise TaskCuratorContractError("Checkpoint is missing aligned parent evaluation results")
            if any(r.get("candidate_idx") != parent for r in results):
                raise TaskCuratorContractError("Parent evaluation belongs to a different candidate")
            self._restore_successes(entry, source="before", candidate_idx=parent)
            for row in results:
                self._append_history(row["task_id"], _history_row_without_output(row))

            # Restore sampling progress on the parent before taking the child's
            # snapshot. Returning to this parent resumes its own cursor/clock.
            self._branch.epoch = entry["task_curator_epoch"]
            self._branch.epoch_order = list(entry["task_curator_epoch_order"])
            self._branch.position = entry["task_curator_position_after_sampling"]
            self._branch.selected_in_epoch = entry["task_curator_selected_in_epoch_after_sampling"]
            if (
                len(self._branch.epoch_order) != len(self._known_ids or ())
                or set(self._branch.epoch_order) != set(self._known_ids or ())
                or not 0 <= self._branch.position <= len(self._branch.epoch_order)
            ):
                raise TaskCuratorContractError("Invalid saved epoch position")

            comparisons = entry.get("task_curator_mutation_results", [])
            child = entry.get("task_curator_child_idx")
            if entry.get("new_program_idx") is not None and child is None:
                raise TaskCuratorContractError("Accepted candidate is missing its curator branch evidence")
            if comparisons:
                if [r["task_id"] for r in comparisons] != ids or "task_curator_child_idx" not in entry:
                    raise TaskCuratorContractError("Checkpoint is missing aligned mutation results")
                accepted = child is not None
                if any(
                    r["mutation_accepted"] is not accepted
                    or r["before"].get("candidate_idx") != parent
                    or r["after"].get("candidate_idx") != child
                    for r in comparisons
                ):
                    raise TaskCuratorContractError("Mutation ownership or acceptance differs from checkpoint lineage")
                if accepted:
                    if entry.get("new_program_idx", child) != child:
                        raise TaskCuratorContractError("Curator child differs from accepted candidate")
                    self._fork_candidate(child)
                self._restore_successes(entry, source="after", candidate_idx=child)
                for row in comparisons:
                    self._append_history(row["task_id"], _history_row_without_output(row))
            elif child is not None or any(s["source"] == "after" for s in entry.get("task_curator_success_states", [])):
                raise TaskCuratorContractError("Checkpoint has a child without mutation evidence")
            self._recorded_iterations.add(iteration)
        self._restored = True

    def _restore_successes(self, entry: dict, *, source: str, candidate_idx: int | None) -> None:
        for snapshot in entry.get("task_curator_success_states", []):
            if snapshot["source"] not in {"before", "after"}:
                raise TaskCuratorContractError("Invalid success source in checkpoint")
            if snapshot["source"] != source:
                continue
            tid, sid = snapshot["task_id"], snapshot["success_state_id"]
            if (
                tid not in self._branch.success_states
                or sid in self._branch.success_states[tid]
                or snapshot.get("candidate_idx") != candidate_idx
            ):
                raise TaskCuratorContractError("Invalid success snapshot or candidate in checkpoint")
            self._branch.success_states[tid][sid] = deepcopy(snapshot)
