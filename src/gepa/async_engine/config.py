# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Configuration for the stage-decoupled asynchronous engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StageName = Literal["rollout", "propose", "screen", "validate", "patch"]
StalenessPolicy = Literal["full", "guarded", "reflective"]
DrainMode = Literal["finish", "cancel"]

#: Stages in pipeline order. ``patch`` is only used by the ``reflective`` staleness policy.
PIPELINE_STAGES: tuple[StageName, ...] = ("rollout", "propose", "screen", "validate")
ALL_STAGES: tuple[StageName, ...] = (*PIPELINE_STAGES, "patch")

#: Which shared resource each stage draws from.
STAGE_RESOURCE: dict[str, str] = {
    "rollout": "evaluator",
    "propose": "proposer",
    "screen": "evaluator",
    "validate": "evaluator",
    "patch": "proposer",
}


@dataclass
class StageConfig:
    """Concurrency settings for one stage.

    ``workers`` is the number of tasks the stage may have in flight. It is the
    dial for "scaling a stage". When adaptive control is on, ``workers`` is the
    starting value and the controller moves it within
    ``[min_workers, max_workers]``.

    ``batch_size`` lets one task carry several queued items, so the adapter's
    ``batch_evaluate`` (or a batch-capable reflection LM) can vectorize them.

    ``max_queue`` bounds the stage's input buffer. A full buffer applies
    backpressure: the upstream stage stops dispatching until it drains.
    ``None`` means ``2 * max_workers * batch_size``.
    """

    workers: int = 1
    min_workers: int = 1
    max_workers: int | None = None
    batch_size: int = 1
    max_queue: int | None = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_workers is None:
            self.max_workers = self.workers
        if self.min_workers < 1 or self.min_workers > self.workers or self.workers > self.max_workers:
            raise ValueError("require 1 <= min_workers <= workers <= max_workers")
        if self.max_queue is None:
            self.max_queue = 2 * self.max_workers * self.batch_size
        if self.max_queue < 1:
            raise ValueError("max_queue must be >= 1")


@dataclass
class AsyncEngineConfig:
    """Settings for :class:`~gepa.async_engine.engine.AsyncStageEngine`.

    Stages
        ``rollout`` evaluates a selected parent on a minibatch with traces.
        ``propose`` builds the reflective dataset and calls the reflection LM.
        ``screen`` evaluates the child on the same minibatch.
        ``validate`` evaluates an accepted child on the validation set.
        ``patch`` reconciles a stale child with the current pool
        (``reflective`` policy only).

    Staleness
        Every work item records the pool version (number of commits) it was
        sampled at. Its *gap* is the current pool version minus that.

        ``full``       never checks the gap.
        ``guarded``    drops an item whose gap exceeds ``max_staleness``.
        ``reflective`` sends such an item's proposed text to the reflection LM
                       together with what the pool gained since, and the LM
                       either rewrites it on top of the current best candidate
                       or discards it. Falls back to ``guarded`` when no raw
                       reflection LM callable is available.

    Budget
        With ``reserve_budget`` on, metric calls are reserved when evaluation
        work is dispatched, so in-flight work cannot overshoot
        ``max_metric_calls``. With it off, sampling stops when the stop
        condition fires and in-flight items run to completion, which matches
        the synchronous engine's iteration-boundary check.
    """

    rollout: StageConfig = field(default_factory=lambda: StageConfig(workers=2))
    propose: StageConfig = field(default_factory=lambda: StageConfig(workers=4))
    screen: StageConfig = field(default_factory=lambda: StageConfig(workers=2))
    validate: StageConfig = field(default_factory=lambda: StageConfig(workers=1))
    patch: StageConfig = field(default_factory=lambda: StageConfig(workers=1))

    #: Optional cap on tasks in flight per shared resource, across stages,
    #: e.g. ``{"evaluator": 4, "proposer": 8}``.
    resource_limits: dict[str, int] = field(default_factory=dict)

    #: Upper bound on work items alive anywhere in the pipeline. ``None`` means
    #: twice the total worker count.
    max_pipeline_items: int | None = None

    staleness_policy: StalenessPolicy = "guarded"
    max_staleness: int = 4

    adaptive_workers: bool = False
    adjust_interval_seconds: float = 5.0

    reserve_budget: bool = True
    drain: DrainMode = "finish"

    #: Early-exit validation. When set in (0, 1), a validated candidate is first
    #: scored on this fraction of its validation ids. It continues to the rest
    #: only if its mean on that prefix is at least the current best candidate's
    #: mean on the same ids minus ``validate_prefix_margin``. Rejected
    #: candidates never enter the pool. Off by default because it changes which
    #: candidates the pool contains.
    validate_prefix_fraction: float | None = None
    validate_prefix_margin: float = 0.0
    #: A validation id on which this many consecutive validated candidates
    #: matched the best known score is moved behind the prefix, keeping
    #: discriminative ids in front.
    validate_prefix_pass_streak: int = 3

    #: Head loop poll interval while waiting for worker completions.
    poll_interval_seconds: float = 0.02
    #: Save ``gepa_state.bin`` every N commits (and always at the end).
    save_every_commits: int = 1
    #: Write ``async_events.jsonl`` under ``run_dir``.
    write_event_log: bool = True

    def __post_init__(self) -> None:
        if self.staleness_policy not in ("full", "guarded", "reflective"):
            raise ValueError(f"unknown staleness_policy {self.staleness_policy!r}")
        if self.max_staleness < 0:
            raise ValueError("max_staleness must be >= 0")
        if self.drain not in ("finish", "cancel"):
            raise ValueError(f"unknown drain mode {self.drain!r}")
        if self.validate_prefix_fraction is not None and not 0.0 < self.validate_prefix_fraction < 1.0:
            raise ValueError("validate_prefix_fraction must be in (0, 1)")
        if self.max_pipeline_items is None:
            total = sum(self.stage(s).max_workers or 1 for s in PIPELINE_STAGES)
            self.max_pipeline_items = 2 * total
        if self.max_pipeline_items < 1:
            raise ValueError("max_pipeline_items must be >= 1")

    def stage(self, name: str) -> StageConfig:
        return getattr(self, name)

    @classmethod
    def sequential(cls) -> AsyncEngineConfig:
        """One item in the pipeline at a time: reproduces the synchronous loop's order of operations."""
        one = lambda: StageConfig(workers=1, max_queue=1)
        return cls(
            rollout=one(),
            propose=one(),
            screen=one(),
            validate=one(),
            patch=one(),
            max_pipeline_items=1,
            staleness_policy="full",
            reserve_budget=False,
        )
