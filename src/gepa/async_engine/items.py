# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The unit of work that flows through the stage buffers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch
from gepa.core.state import ValsetEvaluation
from gepa.proposer.base import CandidateProposal

#: Terminal outcomes of a work item.
OUTCOMES = (
    "accepted",
    "rejected",
    "skipped",
    "no_proposal",
    "duplicate",
    "discarded_stale",
    "validation_rejected",
    "budget",
    "cancelled",
    "error",
)


@dataclass
class WorkItem:
    """One (parent, minibatch) work order and everything the stages add to it.

    Only the head thread mutates a ``WorkItem`` while it sits in a buffer.
    A worker thread owns the items of the task it is running and returns its
    results to the head, which writes them back. Workers never see optimizer
    state.
    """

    order_id: int
    iteration_id: str
    version: int
    parent_idx: int
    parent_candidate: dict[str, str]
    minibatch_ids: list[Any]
    minibatch: list[Any]
    trace: dict[str, Any]
    created_at: float

    # rollout
    parent_eval: EvaluationBatch | None = None
    # head, after rollout
    components: list[str] | None = None
    # propose
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
    new_texts: dict[str, str] | None = None
    lm_metadata: dict[str, Any] = field(default_factory=dict)
    child_candidate: dict[str, str] | None = None
    # patch
    patched: bool = False
    needs_parent_eval: bool = False
    # screen
    child_eval: EvaluationBatch | None = None
    proposal: CandidateProposal | None = None
    # validate
    val_ids: list[Any] = field(default_factory=list)
    val_cached: dict[Any, Any] = field(default_factory=dict)
    val_todo: list[Any] = field(default_factory=list)
    val_prefix_done: bool = False
    val_inflight: list[Any] = field(default_factory=list)
    val_partial: dict[Any, tuple[Any, float, Any]] = field(default_factory=dict)
    valset_evaluation: ValsetEvaluation | None = None
    val_metric_calls: int = 0

    # bookkeeping
    reserved: int = 0
    enqueued_at: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    wait_seconds: dict[str, float] = field(default_factory=dict)
    outcome: str | None = None

    def gap(self, current_version: int) -> int:
        return current_version - self.version
