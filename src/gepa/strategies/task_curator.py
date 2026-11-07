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
from enum import Enum
from numbers import Real
from typing import Any, Generic

from gepa.core.adapter import DataInst, EvaluationBatch
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import LanguageModel

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


def _default_evidence(trajectory: Any) -> Any:
    """Honor an explicit curator view; otherwise preserve the supplied trajectory."""
    if isinstance(trajectory, dict) and trajectory.get(_TASK_CURATOR_EVIDENCE_KEY) is not None:
        return trajectory[_TASK_CURATOR_EVIDENCE_KEY]
    return trajectory


class _EvidenceDefault(Enum):
    SOURCE = "source"


@dataclass(frozen=True)
class CuratorEvidence:
    """User-controlled views of one evaluation, returned by ``evidence_fn``.

    ``history`` is shown for recent executions; ``summary`` supplements the
    score/outcome metadata after an execution leaves ``history_window``.
    ``success`` is shown when selecting an anchor and later during reflection.
    Omitted history and success independently use the original source evidence
    (the explicit ``task_curator_specific_info`` packet, or the trajectory).
    Explicit ``history=None`` shows metadata only; ``success=None`` declines to
    provide an anchor. Values must be JSON-serializable, not opaque objects.
    The curator snapshots them once and restores those snapshots on resume.
    """

    history: Any = _EvidenceDefault.SOURCE
    summary: Any = None
    success: Any = _EvidenceDefault.SOURCE


def _history_row_without_output(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"task_id", "output"}}


def _default_success(score: float, trajectory: Any) -> bool:
    return score == 1.0


def _reported_crash(trajectory: Any) -> bool:
    """Only a producer's explicit per-rollout flag establishes a recovered crash."""
    packet = _default_evidence(trajectory)
    return any(
        isinstance(record, dict)
        and (record.get("infrastructure_error") is True or record.get("_gepa_transient_failure") is True)
        for record in (trajectory, packet)
    )


def _success_state_id(task_id: Any, iteration: int, source: str) -> str:
    encoded_task_id = json.dumps(_json_safe(task_id), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"task:{encoded_task_id}/iteration:{iteration}/{source}"


class TaskCuratorError(ValueError):
    """Fatal curator failure; continuing could corrupt curriculum state."""


class TaskCuratorLMError(TaskCuratorError):
    """An LM callable can raise this for a fatal error that must stop optimization."""


class TaskCuratorContractError(TaskCuratorError):
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


@dataclass
class _ProposalContext:
    iteration: int
    proposal_id: str
    parent_idx: int
    entry: dict[str, Any]
    anchors: dict[Any, str] = field(default_factory=dict)
    evaluation_id: str | None = None


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
    units in detail. Extracted evidence and decisions remain in the GEPA trace for
    audit and resume; the window limits prompt detail, not stored evidence.

    Default extraction honors ``_task_curator_view`` on tasks and
    ``task_curator_specific_info`` on trajectories. Without a dedicated packet,
    the supplied trajectory is retained without filtering its field names.
    Default serialization represents opaque objects with str(); use ``evidence_fn``
    for a meaningful projection of custom objects or images.
    Raw rollout artifacts outside an explicit packet are not included in that view.

    ``task_view`` controls task descriptions. Optional ``evidence_fn(candidate,
    evaluation)`` returns :class:`CuratorEvidence`: separate recent-history,
    older-summary, and success views. It receives an ``EvaluationBatch`` for one
    task, including original score, output, trajectory and objective scores;
    ``num_metric_calls`` is None because the batch cost cannot be apportioned.
    The candidate is the one that produced this evaluation, even for rejected
    mutations. Both omitted history and omitted success use the source evidence
    independently. A short history does not silently shorten a success reference.
    Callbacks must not mutate their inputs. Views must be JSON-serializable;
    provide a projection for opaque objects or images rather than relying on str().
    Each evaluation's views are recorded once and restored unchanged on resume;
    formatting is not rerun for shared parent evaluations or before/after pairing.
    Changing the callback only affects new records; previously omitted evidence
    cannot be recovered from the curator trace. Metadata such as score, outcome,
    task identity and lineage remains curator-owned, outside user views.

    Legacy ``evidence_extractor`` and ``success_extractor`` remain supported but
    cannot be combined with ``evidence_fn``. They receive the original trajectory;
    a custom history extractor still supplies the success view unless separately
    overridden, and their outputs retain the previous JSON-safe conversion.
    A success extractor returning None declines to provide an anchor.
    Evidence must include what the curator needs to judge
    correctness. An explicitly reported recovered evaluator crash takes precedence.
    By default, score == 1 is a success and any other finite score is a failure.
    Optional predicates receive the score and original trajectory. If an explicit
    failure predicate also declines a non-success, the outcome is unknown.
    Only successes supply anchors, and only task failures receive them.
    Without an explicit success view, missing trajectories cannot supply an anchor.

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
    A new seed supplied on resume forks the saved seed's curriculum without
    treating its validation results as task-history or success-anchor evidence.
    ``history`` exposes a detached audit view of the active candidate.

    This sampler supports greedy, sequential Pareto, and batched single-parent
    proposals (Best-of-N and PxN), with a stable task-ID universe. Sampling is
    ordered; sibling mutations inherit a frozen common parent prefix. Multi-parent
    merge is rejected explicitly. Resume uses saved curator trace
    entries automatically. Create a new sampler per optimization run. Full-curator and heuristic anchor modes
    are intentionally not part of this implementation.
    """

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
        evidence_fn: Callable[[dict[str, str], EvaluationBatch], CuratorEvidence] | None = None,
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
        if evidence_fn is not None and (evidence_extractor is not None or success_extractor is not None):
            raise ValueError("Use evidence_fn or the legacy extractors, not both")
        self.minibatch_size = minibatch_size
        self.lm = lm
        self.logger = logger
        self.max_workers = max_workers
        self.seed = seed
        self.history_window = history_window
        self.use_reflection_anchors = use_reflection_anchors
        self.task_view = task_view if task_view is not None else _task_view
        self.evidence_fn = evidence_fn
        self._legacy_extractors = evidence_extractor is not None or success_extractor is not None
        self.evidence_extractor = evidence_extractor if evidence_extractor is not None else _default_evidence
        self.success_extractor = success_extractor if success_extractor is not None else self.evidence_extractor
        self.is_success = is_success if is_success is not None else _default_success
        self.is_failure = is_failure
        self._custom_metrics = is_success is not None or is_failure is not None
        self.raise_on_error = raise_on_error
        self._branches: dict[int, _BranchState] = {0: _BranchState()}
        self._active_candidate_idx = 0
        self._contexts: dict[str, _ProposalContext] = {}
        self._active_proposal_id: str | None = None
        self._parent_bases: dict[int, _BranchState] = {}
        self._batch_entry: dict[str, Any] | None = None
        self._known_ids: tuple[DataId, ...] | None = None
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

    @property
    def _context(self) -> _ProposalContext:
        if self._active_proposal_id not in self._contexts:
            raise TaskCuratorContractError("No active curator proposal")
        return self._contexts[self._active_proposal_id]

    def _activate_proposal(self, iteration: int, proposal_id: str | None) -> None:
        if proposal_id is None and len(self._contexts) == 1:
            proposal_id = next(iter(self._contexts))
        context = self._contexts.get(proposal_id) if isinstance(proposal_id, str) else None
        if context is None or context.iteration != iteration:
            raise TaskCuratorContractError("Missing or unknown curator proposal ID")
        self._active_proposal_id = proposal_id
        self._active_candidate_idx = context.parent_idx

    def _begin_proposal(self, state: GEPAState, parent: int, proposal_id: str | None) -> None:
        iteration = state.i + 1
        proposal_id = proposal_id if proposal_id is not None else f"{iteration}-0"
        if not isinstance(proposal_id, str) or not proposal_id:
            raise TaskCuratorContractError("A proposal ID must be a nonempty string")
        top = self._iteration_entry(state, iteration)
        self._batch_entry = top
        if self._contexts and next(iter(self._contexts.values())).iteration != iteration:
            self._contexts.clear()
            self._parent_bases.clear()
        if proposal_id in self._contexts or any("task_curator_results" in c.entry for c in self._contexts.values()):
            raise TaskCuratorContractError("Duplicate proposal ID or sampling after evaluation began")
        if not self._contexts:
            if top.get("task_curator_enabled"):
                raise TaskCuratorContractError("Iteration already contains curator evidence")
            entry = top
        else:
            if len(self._contexts) == 1:
                # Preserve the version-8 trace for N=1. Promote only when a
                # second slot exists, before evaluations or outcomes begin.
                first = next(iter(self._contexts.values()))
                first.entry = {"i": state.i, "proposal_id": first.proposal_id}
                for key in list(top):
                    if key.startswith("task_curator_"):
                        first.entry[key] = top.pop(key)
                first.entry["task_curator_sampled_anchors"] = [
                    {"task_id": tid, "success_state_id": sid} for tid, sid in first.anchors.items()
                ]
                top.update(
                    task_curator_enabled=True,
                    task_curator_settings={**self._settings(), "version": 9},
                    task_curator_proposals=[first.entry],
                    task_curator_outcome_order=[],
                    task_curator_batch_complete=False,
                )
            entry = {"i": state.i, "proposal_id": proposal_id}
            top["task_curator_proposals"].append(entry)
        self._contexts[proposal_id] = _ProposalContext(iteration, proposal_id, parent, entry)
        self._activate_proposal(iteration, proposal_id)

    def _freeze_parents(self) -> None:
        if len(self._contexts) <= 1 or self._parent_bases:
            return
        if any("task_curator_results" not in c.entry for c in self._contexts.values()):
            raise TaskCuratorContractError("All parent evaluations must precede mutation outcomes")
        self._parent_bases = {c.parent_idx: self._branches[c.parent_idx].fork() for c in self._contexts.values()}

    def _evidence_metadata(self, *, comparison: bool = False) -> dict[str, Any]:
        if len(self._contexts) <= 1:
            return {}
        c = self._context
        result = {"evaluation_id": c.evaluation_id}
        if comparison:
            result["proposal_id"] = c.proposal_id
        return result

    def finish_iteration(
        self,
        *,
        iteration: int,
        proposals: Sequence[CandidateProposal] | None,
        state: GEPAState,
        aborted_before_registration: bool = False,
    ) -> None:
        """Fail closed before saving an incomplete batch; ordinary skips keep parent evidence."""
        if not self._contexts or next(iter(self._contexts.values())).iteration != iteration:
            return
        can_recover = (
            aborted_before_registration
            and len(self._contexts) == 1
            and self._context.entry.get("task_curator_child_idx") is None
        )
        if proposals is None:
            if can_recover and "task_curator_results" in self._context.entry:
                # The sequential path retains parent evidence after a recoverable
                # failure, but only before any child has entered the pool.
                return
            raise TaskCuratorContractError(
                "Proposal generation failed; incomplete curator batch cannot be checkpointed"
            )
        returned = set()
        for proposal in proposals:
            pid = proposal.metadata.get("proposal_id")
            if pid is None and len(self._contexts) == 1:
                pid = next(iter(self._contexts))
            if pid not in self._contexts or pid in returned:
                raise TaskCuratorContractError("Returned proposals have unknown or duplicate IDs")
            returned.add(pid)
        for pid, context in self._contexts.items():
            entry = context.entry
            if "task_curator_results" not in entry:
                raise TaskCuratorContractError("Proposal is missing its parent evaluation")
            if ("task_curator_mutation_results" in entry) != (pid in returned):
                if can_recover and pid in returned:
                    # Selection may fail after the proposal was evaluated. Keep
                    # the parent history without inventing a rejection or child.
                    continue
                raise TaskCuratorContractError(
                    "Returned proposal is missing its final outcome, or an unreturned one has it"
                )
        if len(self._contexts) > 1:
            top = self._iteration_entry(state, iteration)
            children = [
                c.entry["task_curator_child_idx"]
                for c in self._contexts.values()
                if c.entry.get("task_curator_child_idx") is not None
            ]
            if sorted(children) != sorted(top.get("new_program_indices", [])):
                raise TaskCuratorContractError("Registered candidates and curator outcomes differ")
            for pid, context in self._contexts.items():
                context.entry["task_curator_outcome"] = (
                    "skipped"
                    if pid not in returned
                    else "accepted"
                    if context.entry["task_curator_child_idx"] is not None
                    else "rejected"
                )
            top["task_curator_batch_complete"] = True

    def _check_iteration_parent(self, iteration: int) -> int:
        context = self._context
        if context.iteration != iteration or context.parent_idx != self._active_candidate_idx:
            raise TaskCuratorContractError("Curator evidence does not belong to the active sampled parent")
        return context.parent_idx

    def _fork_candidate(self, candidate_idx: int) -> None:
        if isinstance(candidate_idx, bool) or not isinstance(candidate_idx, int) or candidate_idx < 1:
            raise TaskCuratorContractError("An accepted mutation needs an explicit new candidate ID")
        if candidate_idx in self._branches:
            raise TaskCuratorContractError("Curator candidate already exists")
        self._branches[candidate_idx] = self._branch.fork()
        self._active_candidate_idx = candidate_idx

    def _restore_if_needed(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> None:
        if not self._restored:
            prior = [row for row in state.full_program_trace if row.get("i", -1) < state.i]
            self.load_from_trace(prior, loader, require_history=bool(prior))

    def _record_candidate_registration(
        self,
        *,
        iteration: int,
        parent_idx: int,
        candidate_idx: int,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
    ) -> None:
        """Inherit a registered candidate's parent without inventing training evidence."""
        if iteration != state.i + 1:
            raise TaskCuratorContractError("Candidate registration belongs to a different iteration")
        self._restore_if_needed(loader, state)
        self.assert_no_unrecorded_through(state.i)
        self._set_or_validate_universe(loader)
        entry = self._iteration_entry(state, iteration)
        if (
            iteration in self._recorded_iterations
            or entry.get("task_curator_enabled")
            or entry.get("new_program_idx") != candidate_idx
            or entry.get("proposal_accepted", True) is not True
        ):
            raise TaskCuratorContractError("Candidate requires its own accepted registration")
        if isinstance(parent_idx, bool) or not isinstance(parent_idx, int) or parent_idx not in self._branches:
            raise TaskCuratorContractError("Candidate registration references unknown curator parent")
        previous = self._active_candidate_idx
        self._active_candidate_idx = parent_idx
        try:
            self._fork_candidate(candidate_idx)
        except TaskCuratorContractError:
            self._active_candidate_idx = previous
            raise
        entry.update(
            {
                # Keep the version-8 wire format so existing checkpoints replay
                # unchanged. Registration now arrives from the common pool-add
                # path, before the engine's outer acceptance notification.
                "proposal_accepted": True,
                "task_curator_enabled": True,
                "task_curator_settings": self._settings(),
                "task_curator_resumed_seed": True,
                "task_curator_parent_idx": parent_idx,
                "task_curator_child_idx": candidate_idx,
            }
        )
        self._recorded_iterations.add(iteration)

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
            "uses_default_success": self.is_success is _default_success,
        }

    def _trace_entry(self, state: GEPAState, iteration: int) -> dict:
        if self._active_proposal_id in self._contexts and self._context.iteration == iteration:
            return self._context.entry
        return self._iteration_entry(state, iteration)

    @staticmethod
    def _iteration_entry(state: GEPAState, iteration: int) -> dict:
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

    @staticmethod
    def _execution_key(row: dict[str, Any]) -> tuple:
        if "proposal_id" in row:
            return (row.get("iteration"), "proposal", row["proposal_id"])
        if "evaluation_id" in row:
            return (row.get("iteration"), "evaluation", row["evaluation_id"])
        return (row.get("epoch"), row.get("iteration"))

    def _execution_units(self, task_id: DataId) -> list[dict[str, Any]]:
        """A comparison is one unit; reused parent evaluations are not extra attempts."""
        history = self._task_history(task_id)
        compared = {
            (r["iteration"], r["evaluation_id"])
            for r in history
            if r.get("event") == "mutation_comparison" and "evaluation_id" in r
        }
        units: dict[tuple, dict[str, Any]] = {}
        for index, row in enumerate(history):
            if row.get("event") not in _EVIDENCE_EVENTS:
                continue
            if row.get("event") == "evaluation_result" and (row.get("iteration"), row.get("evaluation_id")) in compared:
                continue
            key = self._execution_key(row) if row.get("iteration") is not None else ("row", index)
            previous = units.get(key)
            if previous is None or row.get("event") == "mutation_comparison":
                units[key] = row
        return list(units.values())

    def _history_for_prompt(self, task_id: DataId) -> list[dict[str, Any]]:
        rows = self._execution_units(task_id)
        return rows if self.history_window is None else rows[-self.history_window :]

    def _rollout_outcome(self, score: float, trajectory: Any) -> str:
        if _reported_crash(trajectory):
            return "infrastructure_error"
        if self.is_success(score, trajectory):
            return "success"
        if self.is_failure is None or self.is_failure(score, trajectory):
            return "fail"
        return "unknown"

    def _compact_outcome(self, row: dict[str, Any]) -> str:
        """Read saved outcomes, including checkpoints that inferred labels from text."""
        saved = row.get("outcome")
        if saved in ("success", "fail", "unknown"):
            return saved
        # New crash rows keep provenance even when a formatter omits the marker.
        # Older infrastructure labels alone could have come from a timeout message.
        if row.get("recovered_crash") is True or _reported_crash(row.get("evaluator_feedback")):
            return "infrastructure_error"
        score = row.get("score")
        if self._custom_metrics or not isinstance(score, Real) or not math.isfinite(float(score)):
            return "unknown"
        return "success" if score == 1 else "fail"

    def _execution_assessments(self, task_id: DataId) -> dict[tuple[Any, Any], dict[str, Any]]:
        assessments = {}
        last_execution = None
        for row in self._task_history(task_id):
            if row.get("event") in _EVIDENCE_EVENTS:
                last_execution = self._execution_key(row)
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
            summary.update(assessments.get(self._execution_key(row), {}))
            if row.get("event") == "mutation_comparison":
                summary.update(
                    before=self._compact_outcome(row.get("before", {})),
                    after=self._compact_outcome(row.get("after", {})),
                    mutation_accepted=row.get("mutation_accepted"),
                    before_candidate_idx=row.get("before", {}).get("candidate_idx"),
                    after_candidate_idx=row.get("after", {}).get("candidate_idx"),
                    before_score=row.get("before", {}).get("score"),
                    after_score=row.get("after", {}).get("score"),
                )
                for side in ("before", "after"):
                    evidence = row.get(side, {}).get("evidence_summary")
                    if evidence is not None:
                        summary[f"{side}_evidence"] = evidence
            else:
                summary["outcome"] = self._compact_outcome(row)
                summary["candidate_idx"] = row.get("candidate_idx")
                summary["score"] = row.get("score")
                if row.get("evidence_summary") is not None:
                    summary["evidence"] = row["evidence_summary"]
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
                if key in {
                    "iteration",
                    "iteration_id",
                    "task_id",
                    "reason",
                    "anchor_reason",
                    "proposal_id",
                    "evaluation_id",
                    "evidence_summary",
                    "recovered_crash",
                }:
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
                eligible = {}
                for side, sid in pool_candidates.items():
                    latest = prompt_history[-1]
                    paired = latest.get("event") == "mutation_comparison"
                    history_evidence = (latest[side] if paired else latest).get("evaluator_feedback")
                    success_evidence = self._branch.success_states[task_id][sid]["success_state"]
                    eligible[side] = (
                        (f"History[-1].{side}.evaluator_feedback" if paired else "History[-1].evaluator_feedback")
                        if json.dumps(success_evidence, sort_keys=True) == json.dumps(history_evidence, sort_keys=True)
                        else {"successful_state": success_evidence}
                    )
                prompt += (
                    "\n\nChoose a success to show the reflector as an example of how to complete this task. A passing score"
                    " makes a rollout eligible for consideration. If an execution only explains why the requested work"
                    " could not be completed, discard it as a reference even if the judge calls that response correct. If"
                    ' every new candidate falls into this category, return "selected_success": "retained", including when'
                    " the pool is empty. When the task asks a question, checking the relevant evidence and giving a"
                    " supported answer can itself complete the task. Accept any execution that genuinely accomplishes the"
                    " task, including by a different route. Among qualifying executions, prefer a clear, efficient,"
                    " reproducible example. Replace an existing reference only with a better one. Include"
                    ' "selected_success": "retained", "before", or "after" in your JSON response. "retained" leaves the'
                    ' pool unchanged, including when it is empty. "before" or "after" stores that successful trajectory.'
                    " Only choose a side listed below, failed sides are not eligible. A success remains eligible even if"
                    " its mutation was rejected."
                    "\n\nEligible successes (the exact success evidence, or a reference when identical to History):\n"
                    + json.dumps(eligible, ensure_ascii=False, indent=2)
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
        except TaskCuratorError:
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
        proposal_id: str | None = None,
    ) -> list[DataId]:
        self._restore_if_needed(loader, state)
        self.assert_no_unrecorded_through(state.i)
        all_ids = self._set_or_validate_universe(loader)
        iteration = state.i + 1
        if isinstance(candidate_idx, bool) or not isinstance(candidate_idx, int) or candidate_idx not in self._branches:
            raise TaskCuratorContractError(f"No curator history for candidate {candidate_idx!r}")
        self._begin_proposal(state, candidate_idx, proposal_id)
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

        self._context.anchors = {
            tid: self._branch.retained_success[tid] for tid in selected if tid in self._branch.retained_success
        }
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
        if len(self._contexts) > 1:
            trace_entry["task_curator_sampled_anchors"] = [
                {"task_id": tid, "success_state_id": sid} for tid, sid in self._context.anchors.items()
            ]
        return selected

    def _prepare_evidence(
        self, candidate: dict[str, str], evaluation: EvaluationBatch
    ) -> list[tuple[dict[str, Any], Any]]:
        """Snapshot views before committing any task's history or success state."""
        count = len(evaluation.scores)
        for name in ("outputs", "trajectories", "objective_scores"):
            values = getattr(evaluation, name)
            if values is not None and len(values) != count:
                raise TaskCuratorContractError(f"Curator evaluation {name} are misaligned")
        prepared = []
        for index, score in enumerate(evaluation.scores):
            if not isinstance(score, Real) or not math.isfinite(float(score)):
                raise TaskCuratorContractError(f"Task curator received invalid score: {score!r}")
            trajectory = evaluation.trajectories[index] if evaluation.trajectories is not None else None
            try:
                outcome = self._rollout_outcome(score, trajectory)
                succeeded = outcome == "success"
                source = _default_evidence(trajectory)
                if self.evidence_fn is not None:
                    single = replace(
                        evaluation,
                        outputs=[evaluation.outputs[index]],
                        scores=[score],
                        trajectories=None if evaluation.trajectories is None else [trajectory],
                        objective_scores=(
                            None if evaluation.objective_scores is None else [evaluation.objective_scores[index]]
                        ),
                        num_metric_calls=None,
                    )
                    views = self.evidence_fn(dict(candidate), single)
                    if not isinstance(views, CuratorEvidence):
                        raise TypeError("evidence_fn must return CuratorEvidence")
                elif self._legacy_extractors:
                    views = CuratorEvidence(
                        history=self.evidence_extractor(trajectory),
                        success=(
                            self.success_extractor(trajectory)
                            if succeeded and self.use_reflection_anchors and trajectory is not None
                            else None
                        ),
                    )
                else:
                    views = CuratorEvidence()

                def snapshot(value: Any, source: Any = source) -> Any:
                    value = source if value is _EvidenceDefault.SOURCE else value
                    if self.evidence_fn is None:
                        return _json_safe(value)
                    # Formatting is explicit: never silently replace a rich
                    # trajectory or image with its Python string representation.
                    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

                row = {
                    "score": float(score),
                    "outcome": outcome,
                    "evaluator_feedback": snapshot(views.history),
                }
                if outcome == "infrastructure_error":
                    row["recovered_crash"] = True
                if views.summary is not None:
                    row["evidence_summary"] = snapshot(views.summary)
                success = snapshot(views.success) if succeeded and self.use_reflection_anchors else None
                prepared.append((row, success))
            except TaskCuratorError:
                raise
            except Exception as exc:
                raise TaskCuratorContractError(
                    f"Cannot prepare curator evidence for evaluation item {index}: {exc}. "
                    "evidence_fn must return CuratorEvidence with JSON-serializable views; "
                    "set history/success explicitly for opaque trajectories."
                ) from exc
        return prepared

    def _record_success_state(
        self,
        *,
        task_id: DataId,
        iteration: int,
        source: str,
        score: float,
        success: Any,
        state: GEPAState,
        mutation_accepted: bool | None = None,
    ) -> str | None:
        if success is None or not self.use_reflection_anchors:
            return None
        success_state_id = _success_state_id(task_id, iteration, source)
        if len(self._contexts) > 1:
            identity = self._context.evaluation_id if source == "before" else self._context.proposal_id
            success_state_id += "/" + str(identity)
        snapshot = {
            "success_state_id": success_state_id,
            "task_id": task_id,
            "epoch": self._context.entry["task_curator_epoch"] if len(self._contexts) > 1 else self._branch.epoch,
            "iteration": iteration,
            "source": source,
            "candidate_idx": self._active_candidate_idx if source == "before" or mutation_accepted else None,
            "score": float(score),
            "success_state": success,
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
        evaluation: EvaluationBatch | None = None,
    ) -> None:
        parent_idx = self._check_iteration_parent(iteration)
        entry = self._trace_entry(state, iteration)
        if "task_curator_results" in entry:
            raise TaskCuratorContractError(
                f"Task curator proposal was recorded more than once at iteration {iteration}."
            )
        expected = entry.get("task_curator_sampled_ids")
        if expected is None or list(task_ids) != expected:
            raise TaskCuratorContractError(
                f"Task curator task association mismatch at iteration {iteration}: "
                f"expected {list(expected or ())!r}, received {list(task_ids)!r}."
            )
        if not (len(task_ids) == len(scores) == len(outputs) == len(trajectories)):
            raise TaskCuratorContractError(f"Task curator first-pass fields are misaligned at iteration {iteration}.")

        context = self._context
        if evaluation is None:
            evaluation = EvaluationBatch(outputs=list(outputs), scores=list(scores), trajectories=list(trajectories))
        if self._parent_bases:
            raise TaskCuratorContractError("Cannot append parent evidence after outcomes began")
        if len(self._contexts) > 1:
            if context.evaluation_id is None:
                context.evaluation_id = context.proposal_id
            entry["task_curator_evaluation_id"] = context.evaluation_id
            for previous in self._contexts.values():
                if (
                    previous is not context
                    and previous.parent_idx == parent_idx
                    and previous.evaluation_id == context.evaluation_id
                    and "task_curator_results" in previous.entry
                ):
                    rows = previous.entry["task_curator_results"]
                    if [r["task_id"] for r in rows] != list(task_ids) or [r["score"] for r in rows] != list(scores):
                        raise TaskCuratorContractError("Reused evaluation ID has different evidence")
                    entry["task_curator_evaluation_ref"] = previous.proposal_id
                    entry["task_curator_results"] = rows
                    return
        prepared = self._prepare_evidence(state.program_candidates[parent_idx], evaluation)
        results = []
        for task_id, (evidence, success) in zip(task_ids, prepared, strict=True):
            row = {
                "event": "evaluation_result",
                **self._evidence_metadata(),
                "candidate_idx": parent_idx,
                "epoch": entry["task_curator_epoch"] if len(self._contexts) > 1 else self._branch.epoch,
                "iteration": iteration,
                **evidence,
            }
            success_state_id = self._record_success_state(
                task_id=task_id,
                iteration=iteration,
                source="before",
                score=evidence["score"],
                success=success,
                state=state,
            )
            if success_state_id is not None:
                row["success_state_id"] = success_state_id
            self._append_history(task_id, row)
            results.append({"task_id": task_id, **row})

        self._recorded_iterations.add(iteration)
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
        after_candidate: dict[str, str] | None = None,
        after_evaluation: EvaluationBatch | None = None,
    ) -> None:
        """Commit all child evidence together; failed extraction must not create a partial branch."""
        parent_idx = self._check_iteration_parent(iteration)
        self._freeze_parents()
        parent = self._branch
        existing = set(self._branches)
        entry = self._trace_entry(state, iteration)
        previous_entry = dict(entry)
        if "task_curator_success_states" in entry:
            entry["task_curator_success_states"] = list(entry["task_curator_success_states"])
        base = self._parent_bases.get(parent_idx, parent) if mutation_accepted else parent
        self._branches[parent_idx] = base.fork()
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
                after_candidate=after_candidate,
                after_evaluation=after_evaluation,
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
        if len(self._contexts) > 1:
            self._iteration_entry(state, iteration)["task_curator_outcome_order"].append(self._context.proposal_id)

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
        after_candidate: dict[str, str] | None,
        after_evaluation: EvaluationBatch | None,
    ) -> None:
        parent_idx = self._check_iteration_parent(iteration)
        trace_entry = self._trace_entry(state, iteration)
        if "task_curator_mutation_results" in trace_entry:
            raise TaskCuratorContractError("Mutation comparison was recorded more than once")
        if "task_curator_results" not in trace_entry:
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
        if self.evidence_fn is not None and (after_candidate is None or after_evaluation is None):
            raise TaskCuratorContractError("evidence_fn requires the actual after candidate and evaluation")
        if after_candidate is None:
            after_candidate = state.program_candidates[candidate_idx if candidate_idx is not None else parent_idx]
        if after_evaluation is None:
            after_evaluation = EvaluationBatch(
                outputs=[None] * len(task_ids), scores=list(after_scores), trajectories=list(after_trajectories)
            )
        prepared = self._prepare_evidence(after_candidate, after_evaluation)
        recorded_before = {r["task_id"]: r for r in trace_entry["task_curator_results"]}
        if any(recorded_before[tid]["score"] != score for tid, score in zip(task_ids, before_scores, strict=True)):
            raise TaskCuratorContractError("Before scores differ from the recorded parent evaluation")
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
        for task_id, (evidence, success) in zip(task_ids, prepared, strict=True):
            before = recorded_before[task_id]
            row = {
                "event": "mutation_comparison",
                **self._evidence_metadata(comparison=True),
                "epoch": trace_entry["task_curator_epoch"] if len(self._contexts) > 1 else self._branch.epoch,
                "iteration": iteration,
                "mutation_accepted": bool(mutation_accepted),
                "batch_score_before": before_sum,
                "batch_score_after": after_sum,
                "before": {
                    "candidate_idx": parent_idx,
                    **{
                        key: before[key]
                        for key in ("score", "outcome", "recovered_crash", "evaluator_feedback", "evidence_summary")
                        if key in before
                    },
                },
                "after": {
                    "candidate_idx": candidate_idx,
                    **evidence,
                },
            }
            before_id = next(
                r.get("success_state_id") for r in trace_entry["task_curator_results"] if r["task_id"] == task_id
            )
            if before_id in self._branch.success_states[task_id]:
                row["before"]["success_state_id"] = before_id
            after_id = self._record_success_state(
                task_id=task_id,
                iteration=iteration,
                source="after",
                score=evidence["score"],
                success=success,
                state=state,
                mutation_accepted=mutation_accepted,
            )
            if after_id is not None:
                row["after"]["success_state_id"] = after_id
            self._append_history(task_id, row)
            results.append({"task_id": task_id, **row})

        trace_entry["task_curator_mutation_results"] = results

    def assert_no_unrecorded_through(self, iteration: int) -> None:
        if (
            len(self._contexts) > 1
            and next(iter(self._contexts.values())).iteration <= iteration
            and (self._batch_entry or {}).get("task_curator_batch_complete") is not True
        ):
            raise TaskCuratorContractError("Previous curator batch has not been finalized")
        missing = [
            c.proposal_id
            for c in self._contexts.values()
            if c.iteration <= iteration and "task_curator_results" not in c.entry
        ]
        if missing:
            raise TaskCuratorContractError(
                "Task curator sampled minibatches without first-pass results: " + ", ".join(missing)
            )

    def observe_evaluation(
        self,
        *,
        iteration: int,
        task_ids: Sequence[DataId],
        evaluation: EvaluationBatch,
        state: GEPAState,
        proposal_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> None:
        """Record parent evidence even if reflection is skipped or fails."""
        self._activate_proposal(iteration, proposal_id)
        self._context.evaluation_id = evaluation_id or self._context.proposal_id
        self.record_first_pass(
            iteration=iteration,
            task_ids=task_ids,
            scores=evaluation.scores,
            outputs=evaluation.outputs,
            trajectories=evaluation.trajectories or [None] * len(task_ids),
            state=state,
            evaluation=evaluation,
        )

    def observe_outcome(
        self,
        *,
        iteration: int,
        parent_ids: Sequence[int],
        candidate_idx: int | None,
        proposal: CandidateProposal | None,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
    ) -> None:
        """Apply a final outcome, keeping child registration and evidence atomic."""
        if len(parent_ids) != 1:
            raise TaskCuratorContractError("Curator outcomes require exactly one parent")
        if proposal is None:
            if candidate_idx is None:
                raise TaskCuratorContractError("An outcome without a proposal must register a child")
            self._record_candidate_registration(
                iteration=iteration,
                parent_idx=parent_ids[0],
                candidate_idx=candidate_idx,
                loader=loader,
                state=state,
            )
            return
        self._activate_proposal(iteration, proposal.metadata.get("proposal_id"))
        before, after = proposal.eval_before, proposal.eval_after
        if before is None or after is None or proposal.subsample_indices is None:
            raise TaskCuratorContractError("Curator needs before/after evaluations and task IDs")
        parent_idx = self._check_iteration_parent(iteration)
        if list(parent_ids) != [parent_idx] or proposal.parent_program_ids != [parent_idx]:
            raise TaskCuratorContractError("Proposal parents do not match the curator's sampled parent")
        ids = proposal.subsample_indices
        self.record_mutation_comparison(
            iteration=iteration,
            task_ids=ids,
            before_scores=before.scores,
            before_trajectories=before.trajectories or [None] * len(ids),
            after_scores=after.scores,
            after_trajectories=after.trajectories or [None] * len(ids),
            mutation_accepted=candidate_idx is not None,
            state=state,
            candidate_idx=candidate_idx,
            after_candidate=proposal.candidate,
            after_evaluation=EvaluationBatch(
                outputs=after.outputs,
                scores=after.scores,
                trajectories=after.trajectories,
                objective_scores=after.objective_scores,
            ),
        )

    def build_reflective_dataset(
        self,
        *,
        candidate: dict[str, str],
        task_ids: Sequence[DataId],
        evaluation: EvaluationBatch,
        components_to_update: list[str],
        make_reflective_dataset: Callable[
            [dict[str, str], EvaluationBatch, list[str]], Mapping[str, Sequence[Mapping[str, Any]]]
        ],
        state: GEPAState,
        proposal_id: str | None = None,
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
        self._activate_proposal(state.i + 1, proposal_id)
        self._check_iteration_parent(state.i + 1)
        if list(task_ids) != self._context.entry.get("task_curator_sampled_ids"):
            raise TaskCuratorContractError("Reflection task IDs do not align with the sampled proposal")
        if self._parent_bases:
            raise TaskCuratorContractError("Reflection must precede mutation outcomes")
        anchors = self._context.anchors
        if not self.use_reflection_anchors:
            return make_reflective_dataset(candidate, evaluation, components_to_update)
        for name in ("scores", "outputs", "trajectories", "objective_scores"):
            values = getattr(evaluation, name)
            if values is not None and len(values) != len(task_ids):
                raise TaskCuratorContractError(f"Reflection task IDs and {name} do not align")
        if len(set(task_ids)) != len(task_ids):
            raise TaskCuratorContractError("Reflection task IDs must be unique")

        outcomes = {row["task_id"]: row["outcome"] for row in self._context.entry["task_curator_results"]}
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

            sid = anchors.get(tid)
            anchor = None
            if sid is not None and outcomes[tid] == "fail":
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
            {"task_id": tid, "success_state_id": anchors[tid], "supporting_success_ids": []}
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
        """Replay v8 single proposals and v9 batches at complete iteration boundaries."""
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
        self._contexts.clear()
        self._active_proposal_id = None
        self._parent_bases.clear()
        self._recorded_iterations.clear()
        self._restored = False
        for entry in entries:
            batch = "task_curator_proposals" in entry
            settings = {**self._settings(), "version": 9} if batch else self._settings()
            if entry.get("task_curator_settings") != settings:
                raise TaskCuratorContractError("Curator settings changed on resume (including history format)")
            iteration = entry["i"] + 1
            if iteration in self._recorded_iterations:
                raise TaskCuratorContractError("Duplicate curator iteration in checkpoint")
            if batch:
                self._restore_batch(entry)
            else:
                parent = self._restore_parent(entry)
                if "task_curator_resumed_seed" in entry:
                    self._restore_candidate_registration(entry)
                else:
                    self._restore_decisions(entry)
                    self._restore_results(entry, parent)
                    self._restore_cursor(entry)
                    self._restore_outcome(entry, parent)
            self._recorded_iterations.add(iteration)
        self._restored = True

    def _restore_parent(self, entry: dict) -> int:
        parent = entry.get("task_curator_parent_idx")
        if isinstance(parent, bool) or not isinstance(parent, int) or parent not in self._branches:
            raise TaskCuratorContractError("Checkpoint references unknown curator parent")
        if entry.get("selected_program_candidate", parent) != parent:
            raise TaskCuratorContractError("Curator parent differs from selected candidate")
        self._active_candidate_idx = parent
        return parent

    def _restore_decisions(self, entry: dict) -> None:
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

    def _restore_results(self, entry: dict, parent: int) -> None:
        results = entry.get("task_curator_results")
        ids = entry["task_curator_sampled_ids"]
        if not isinstance(results, list) or [r["task_id"] for r in results] != ids:
            raise TaskCuratorContractError("Checkpoint is missing aligned parent evaluation results")
        if any(r.get("candidate_idx") != parent for r in results):
            raise TaskCuratorContractError("Parent evaluation belongs to a different candidate")
        self._restore_successes(entry, source="before", candidate_idx=parent)
        for row in results:
            restored = _history_row_without_output(row)
            restored["outcome"] = self._compact_outcome(row)
            self._append_history(row["task_id"], restored)

    def _restore_cursor(self, entry: dict) -> None:
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

    def _restore_outcome(self, entry: dict, parent: int, base: _BranchState | None = None) -> None:
        self._active_candidate_idx = parent
        actual_parent = self._branch
        ids = entry["task_curator_sampled_ids"]
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
                if base is not None:
                    self._branches[parent] = base.fork()
                self._fork_candidate(child)
                self._branches[parent] = actual_parent
            self._restore_successes(entry, source="after", candidate_idx=child)
            for row in comparisons:
                restored = _history_row_without_output(row)
                for side in ("before", "after"):
                    restored[side] = {**row[side], "outcome": self._compact_outcome(row[side])}
                self._append_history(row["task_id"], restored)
        elif child is not None or any(s["source"] == "after" for s in entry.get("task_curator_success_states", [])):
            raise TaskCuratorContractError("Checkpoint has a child without mutation evidence")

    def _restore_batch(self, top: dict) -> None:
        proposals = top["task_curator_proposals"]
        if top.get("task_curator_batch_complete") is not True or not isinstance(proposals, list) or len(proposals) < 2:
            raise TaskCuratorContractError("Incomplete curator batch in checkpoint")
        entries = {}
        parents = {}
        for entry in proposals:
            pid = entry.get("proposal_id")
            if (
                not isinstance(pid, str)
                or not pid
                or pid in entries
                or entry.get("i") != top["i"]
                or entry.get("task_curator_settings") != self._settings()
            ):
                raise TaskCuratorContractError("Invalid proposal identity or settings in checkpoint")
            entries[pid] = entry
            parent = self._restore_parent(entry)
            parents[pid] = parent
            self._restore_decisions(entry)
            self._restore_cursor(entry)
            anchors = [
                {"task_id": tid, "success_state_id": self._branch.retained_success[tid]}
                for tid in entry["task_curator_sampled_ids"]
                if tid in self._branch.retained_success
            ]
            if entry.get("task_curator_sampled_anchors") != anchors:
                raise TaskCuratorContractError("Sampled anchors differ from the proposal's parent history")

        # Replay shared evaluations exactly once on each parent. All siblings
        # see this common evidence, but no sibling mutation results.
        evaluated = {}
        for pid, entry in entries.items():
            parent = parents[pid]
            self._active_candidate_idx = parent
            eid = entry.get("task_curator_evaluation_id")
            if not isinstance(eid, str) or not eid:
                raise TaskCuratorContractError("Missing actual evaluation identity")
            key = (parent, eid)
            ref = entry.get("task_curator_evaluation_ref")
            if ref is not None:
                if (
                    evaluated.get(key) != ref
                    or ref not in entries
                    or entry.get("task_curator_results") != entries[ref].get("task_curator_results")
                    or entry["task_curator_sampled_ids"] != entries[ref]["task_curator_sampled_ids"]
                    or any(s["source"] == "before" for s in entry.get("task_curator_success_states", []))
                ):
                    raise TaskCuratorContractError("Invalid shared parent evaluation reference")
            else:
                if key in evaluated:
                    raise TaskCuratorContractError("Repeated parent evaluation without an explicit reuse reference")
                self._restore_results(entry, parent)
                evaluated[key] = pid
            if any(r.get("evaluation_id") != eid for r in entry["task_curator_results"]):
                raise TaskCuratorContractError("Parent result has an incorrect evaluation identity")

        bases = {parent: self._branches[parent].fork() for parent in parents.values()}
        order = top.get("task_curator_outcome_order", [])
        outcomes = {pid for pid, e in entries.items() if e.get("task_curator_outcome") in {"accepted", "rejected"}}
        if len(order) != len(set(order)) or set(order) != outcomes:
            raise TaskCuratorContractError("Incomplete or duplicate proposal outcomes in checkpoint")
        children = []
        for pid, entry in entries.items():
            status = entry.get("task_curator_outcome")
            child = entry.get("task_curator_child_idx")
            comparisons = entry.get("task_curator_mutation_results", [])
            if status == "skipped":
                if (
                    comparisons
                    or "task_curator_child_idx" in entry
                    or any(s["source"] == "after" for s in entry.get("task_curator_success_states", []))
                ):
                    raise TaskCuratorContractError("Skipped proposal contains mutation evidence")
            elif status in {"accepted", "rejected"}:
                if not comparisons or (child is not None) != (status == "accepted"):
                    raise TaskCuratorContractError("Proposal status differs from its evidence")
                if any(
                    r.get("proposal_id") != pid or r.get("evaluation_id") != entry["task_curator_evaluation_id"]
                    for r in comparisons
                ):
                    raise TaskCuratorContractError("Mutation evidence belongs to another proposal")
                if child is not None:
                    children.append(child)
            else:
                raise TaskCuratorContractError("Unknown proposal outcome")
        if sorted(children) != sorted(top.get("new_program_indices", [])):
            raise TaskCuratorContractError("Registered candidates and curator outcomes differ")
        for pid in order:
            self._restore_outcome(entries[pid], parents[pid], bases[parents[pid]])

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

    def _restore_candidate_registration(self, entry: dict) -> None:
        """Decode version-8 registrations without changing their persisted format."""
        child_idx = entry.get("task_curator_child_idx")
        if (
            entry["task_curator_resumed_seed"] is not True
            or child_idx is None
            or entry.get("new_program_idx") != child_idx
            or entry.get("proposal_accepted") is not True
            or any(
                key in entry
                for key in (
                    "task_curator_decisions",
                    "task_curator_sampled_ids",
                    "task_curator_results",
                    "task_curator_mutation_results",
                    "task_curator_success_states",
                )
            )
        ):
            raise TaskCuratorContractError("Invalid resumed-seed registration in checkpoint")
        self._fork_candidate(child_idx)
