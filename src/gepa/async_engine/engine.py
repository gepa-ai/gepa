# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Stage-decoupled asynchronous optimization engine.

The synchronous loop runs one iteration at a time: select a parent, evaluate
it on a minibatch, reflect, evaluate the child, validate, commit, repeat. Every
step waits for the previous one, so an iteration costs the sum of its stage
latencies.

This engine cuts the loop at the stage boundaries and puts a buffer at each
cut. Every heavy stage has its own workers and runs whenever its buffer has
work, so different stages process different work items at the same time::

    sample -> [rollout] -> select components -> [propose] -> [screen] -> gate -> [validate] -> commit
     head      workers           head            workers      workers    head     workers      head

One head thread owns the optimizer state. It samples work orders, runs the
cheap state-touching steps, routes items between buffers, and is the only
writer of the candidate pool. Worker threads run the bracketed stages and never
see the state.

Overlap has a price. A work item carries the pool version it was sampled at,
and by the time its child is ready the pool may have moved on. The staleness
policy decides what happens to such items. Acceptance compares a child with its
own parent on the same minibatch, so a stale child that passes the gate is
still a valid candidate; the policy only controls how much outdated work the
pool absorbs.

This module imports GEPA's building blocks (adapter, state, selectors,
samplers, acceptance criteria, reflection LMs, stoppers, callbacks). It does
not import or modify :class:`gepa.core.engine.GEPAEngine`.
"""

from __future__ import annotations

import os
import queue
import time
import traceback
from collections.abc import Mapping
from math import ceil
from typing import Any, Generic

from gepa.async_engine.config import ALL_STAGES, PIPELINE_STAGES, STAGE_RESOURCE, AsyncEngineConfig
from gepa.async_engine.events import EventLog
from gepa.async_engine.items import WorkItem
from gepa.async_engine.scheduler import Completion, StagePool, WorkerController
from gepa.async_engine.stages import (
    PatchJob,
    PatchResult,
    ProposeResult,
    Reflector,
    ScreenResult,
    metric_calls,
    run_patch,
    run_propose,
    run_rollout,
    run_screen,
    run_validate,
)
from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    BudgetUpdatedEvent,
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    CandidateSelectedEvent,
    ErrorEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    IterationEndEvent,
    IterationStartEvent,
    MinibatchSampledEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ParetoFrontUpdatedEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    StateSavedEvent,
    ValsetEvaluatedEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import (
    TRAINSET_CACHE_SPLIT,
    VALSET_CACHE_SPLIT,
    EvaluationCache,
    FrontierType,
    GEPAState,
    ValsetEvaluation,
    initialize_gepa_state,
    new_iteration_id,
)
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionLM
from gepa.strategies.acceptance import AcceptanceCriterion, StrictImprovementAcceptance
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.utils import StopperProtocol

_EVAL_STAGES = ("rollout", "screen", "validate")


def _is_budget_exhausted(err: BaseException) -> bool:
    return type(err).__name__ == "BudgetExhausted"


class AsyncStageEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Runs reflective text optimization as a pipeline of independently scaled stages."""

    def __init__(
        self,
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        run_dir: str | None,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        valset: list[DataInst] | DataLoader[DataId, DataInst] | None,
        seed_candidate: dict[str, str],
        *,
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        reflection_lm: ReflectionLM | None,
        raw_reflection_lm: LanguageModel | None,
        custom_candidate_proposer: ProposalFn | None,
        perfect_score: float | None,
        skip_perfect_score: bool,
        seed: int,
        frontier_type: FrontierType,
        logger: LoggerProtocol,
        experiment_tracker: ExperimentTracker,
        stop_callback: StopperProtocol,
        callbacks: list[GEPACallback] | None = None,
        track_best_outputs: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | None = None,
        acceptance_criterion: AcceptanceCriterion | None = None,
        evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None,
        config: AsyncEngineConfig | None = None,
    ):
        if skip_perfect_score and perfect_score is None:
            raise ValueError("perfect_score must be provided when skip_perfect_score is True.")
        self.adapter = adapter
        self.run_dir = run_dir
        self.trainset = ensure_loader(trainset)
        self.valset = self.trainset if valset is None or valset is trainset else ensure_loader(valset)
        self.valset_cache_split = TRAINSET_CACHE_SPLIT if self.valset is self.trainset else VALSET_CACHE_SPLIT
        self.seed_candidate = seed_candidate
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.seed = seed
        self.frontier_type: FrontierType = frontier_type
        self.logger = logger
        self.experiment_tracker = experiment_tracker
        self.stop_callback = stop_callback
        self.callbacks = callbacks
        self.track_best_outputs = track_best_outputs
        self.raise_on_exception = raise_on_exception
        self.use_cloudpickle = use_cloudpickle
        self.val_evaluation_policy: EvaluationPolicy[DataId, DataInst] = (
            val_evaluation_policy if val_evaluation_policy is not None else FullEvaluationPolicy()
        )
        self.acceptance_criterion: AcceptanceCriterion = acceptance_criterion or StrictImprovementAcceptance()
        self._initial_evaluation_cache = evaluation_cache
        self.config = config or AsyncEngineConfig()

        self.reflector = Reflector(adapter, reflection_lm, custom_candidate_proposer)
        self.raw_reflection_lm = raw_reflection_lm
        self.staleness_policy = self.config.staleness_policy
        if self.staleness_policy == "reflective" and raw_reflection_lm is None:
            self.logger.log(
                "staleness_policy='reflective' needs a reflection LM callable to patch stale proposals; "
                "falling back to 'guarded'."
            )
            self.staleness_policy = "guarded"

        self._completions: queue.Queue[Completion] = queue.Queue()
        self.pools: dict[str, StagePool] = {}
        self.events = EventLog(run_dir, enabled=self.config.write_event_log)
        self.stats: dict[str, Any] = {}

        self._version = 0
        self._reserved = 0
        self._pipeline_items = 0
        self._orders = 0
        self._commits = 0
        self._commits_since_save = 0
        self._stopping = False
        self._stop_requested = False
        self._fatal: BaseException | None = None
        self._budget_blocked = False
        self._orders_this_version = 0
        self._gate_seen = 0
        self._gate_accepted = 0
        self._last_minibatch_size = 1
        self._last_val_cost = len(self.valset)
        self._validating: set[tuple] = set()
        self._val_pass_streak: dict[Any, int] = {}
        self._max_metric_calls = self._find_max_metric_calls(stop_callback)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the engine to stop sampling new work and drain."""
        self._stop_requested = True

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        state = self._initialize()
        self._version = len(state.program_candidates) - 1
        for name in ALL_STAGES:
            self.pools[name] = StagePool(name, STAGE_RESOURCE[name], self.config.stage(name), self._completions)
            self.events.stage(name)
        controller = (
            WorkerController(self.pools, self.config.adjust_interval_seconds) if self.config.adaptive_workers else None
        )
        self.events.emit(
            "run_start",
            version=self._version,
            workers={n: p.workers for n, p in self.pools.items()},
            staleness_policy=self.staleness_policy,
            max_staleness=self.config.max_staleness,
        )
        try:
            while True:
                self._drain_completions(state)
                if self._fatal is not None:
                    break
                if not self._stopping and (self._stop_requested or self.stop_callback(state)):
                    self._begin_stop(state)
                sampled = 0 if self._stopping or self._budget_blocked else self._sample_orders(state)
                dispatched = self._dispatch_all(state)
                self.events.tick({n: p.inflight_tasks for n, p in self.pools.items()})
                if controller is not None:
                    for stage, old, new in controller.maybe_adjust():
                        self.events.stage(stage).worker_changes.append((self.events.now(), new))
                        self.events.emit("workers_changed", stage=stage, old=old, new=new)
                if self._idle() and not sampled and not dispatched:
                    # Nothing in flight and nothing could be started: either we are
                    # draining after a stop, or the budget cannot fund another order.
                    break
        finally:
            for pool in self.pools.values():
                pool.shutdown()

        if self._fatal is not None:
            self.events.close()
            raise self._fatal

        self._save(state)
        best_idx = self.val_evaluation_policy.get_best_program(state)
        self.stats = self.events.summary(
            total_metric_calls=state.total_num_evals, commits=self._commits, orders=self._orders
        )
        self.stats["final_workers"] = {n: p.workers for n, p in self.pools.items()}
        self.events.emit("run_end", **{k: v for k, v in self.stats.items() if k != "stages"})
        self.events.close()
        if self.run_dir is not None:
            import json

            with open(os.path.join(self.run_dir, "async_stats.json"), "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, default=str)

        notify_callbacks(
            self.callbacks,
            "on_optimization_end",
            OptimizationEndEvent(
                best_candidate_idx=best_idx,
                total_iterations=state.i,
                total_metric_calls=state.total_num_evals,
                final_state=state,
            ),
        )
        summary: dict[str, Any] = {
            "best_candidate_idx": best_idx,
            "best_valset_score": self.val_evaluation_policy.get_valset_score(best_idx, state),
            "total_iterations": state.i,
            "total_candidates": len(state.program_candidates),
            "async/wall_seconds": self.stats["wall_seconds"],
            "async/stage_overlap_fraction": self.stats["stage_overlap_fraction"],
            "async/mean_tasks_in_flight": self.stats["mean_tasks_in_flight"],
        }
        self.experiment_tracker.log_summary(summary)
        return state

    # ------------------------------------------------------------------
    # initialization (seed evaluation, state, resume)
    # ------------------------------------------------------------------

    def _initialize(self) -> GEPAState[RolloutOutput, DataId]:
        valset = self.valset
        notify_callbacks(
            self.callbacks,
            "on_optimization_start",
            OptimizationStartEvent(
                seed_candidate=self.seed_candidate,
                trainset_size=len(self.trainset),
                valset_size=len(valset),
                config={
                    "perfect_score": self.perfect_score,
                    "seed": self.seed,
                    "track_best_outputs": self.track_best_outputs,
                    "engine": "async_stage",
                },
            ),
        )
        resumed = self.run_dir is not None and os.path.exists(os.path.join(self.run_dir, "gepa_state.bin"))
        seed_eval: ValsetEvaluation[RolloutOutput, DataId] | None = None
        if not resumed:
            seed_batch_fn = getattr(self.val_evaluation_policy, "get_seed_eval_batch", None)
            seed_ids = list(seed_batch_fn(valset)) if seed_batch_fn is not None else list(valset.all_ids())
            result = self.adapter.evaluate(valset.fetch(seed_ids), self.seed_candidate, capture_traces=False)
            seed_eval = ValsetEvaluation(
                outputs_by_val_id=dict(zip(seed_ids, result.outputs, strict=False)),
                scores_by_val_id=dict(zip(seed_ids, result.scores, strict=False)),
                objective_scores_by_val_id=(
                    dict(zip(seed_ids, result.objective_scores, strict=False))
                    if result.objective_scores is not None
                    else None
                ),
            )
        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            seed_valset_evaluation=seed_eval,
            track_best_outputs=self.track_best_outputs,
            frontier_type=self.frontier_type,
            evaluation_cache=self._initial_evaluation_cache,
        )
        if seed_eval is not None and state.evaluation_cache is not None:
            ids = list(seed_eval.scores_by_val_id)
            obj = (
                [seed_eval.objective_scores_by_val_id[i] for i in ids]
                if seed_eval.objective_scores_by_val_id is not None
                else None
            )
            state.evaluation_cache.put_batch(
                self.seed_candidate,
                ids,
                [seed_eval.outputs_by_val_id[i] for i in ids],
                [seed_eval.scores_by_val_id[i] for i in ids],
                obj,
                split=self.valset_cache_split,
            )
        if resumed and self.seed_candidate not in state.program_candidates:
            self.logger.log(
                "Resumed run: the given seed_candidate is not in the saved pool and is ignored by the async engine."
            )
        setter = getattr(self.adapter, "set_adapter_state", None)
        if setter is not None:
            setter(state.adapter_state)

        self.experiment_tracker.log_config(
            {
                "engine": "async_stage",
                "seed": self.seed,
                "perfect_score": self.perfect_score,
                "frontier_type": self.frontier_type,
                "trainset_size": len(self.trainset),
                "valset_size": len(valset),
                "staleness_policy": self.staleness_policy,
                "max_staleness": self.config.max_staleness,
                "workers": {s: self.config.stage(s).workers for s in ALL_STAGES},
                "adaptive_workers": self.config.adaptive_workers,
                "run_dir": self.run_dir,
            }
        )
        base_avg, base_cov = state.get_program_average_val_subset(0)
        self.experiment_tracker.log_metrics(
            {
                "val_program_average": base_avg,
                "best_score_on_valset": base_avg,
                "total_metric_calls": state.total_num_evals,
                "new_program_idx": 0,
            },
            step=state.i + 1,
        )
        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {base_avg} over {base_cov} / {len(valset)} examples"
        )
        notify_callbacks(
            self.callbacks,
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=0,
                candidate_idx=0,
                candidate=state.program_candidates[0],
                scores_by_val_id=dict(state.prog_candidate_val_subscores[0]),
                average_score=base_avg,
                num_examples_evaluated=len(state.prog_candidate_val_subscores[0]),
                total_valset_size=len(valset),
                parent_ids=[],
                is_best_program=True,
                outputs_by_val_id=None,
            ),
        )

        def budget_hook(new_total: int, delta: int) -> None:
            notify_callbacks(
                self.callbacks,
                "on_budget_updated",
                BudgetUpdatedEvent(
                    iteration=state.i + 1,
                    metric_calls_used=new_total,
                    metric_calls_delta=delta,
                    metric_calls_remaining=self._budget_left(state),
                ),
            )

        state.add_budget_hook(budget_hook)
        return state

    # ------------------------------------------------------------------
    # budget
    # ------------------------------------------------------------------

    @staticmethod
    def _find_max_metric_calls(stop_callback: Any) -> int | None:
        direct = getattr(stop_callback, "max_metric_calls", None)
        if isinstance(direct, int):
            return direct
        found = [
            m
            for s in getattr(stop_callback, "stoppers", None) or []
            if isinstance(m := getattr(s, "max_metric_calls", None), int)
        ]
        return min(found) if found else None

    def _budget_left(self, state: GEPAState) -> int | None:
        if self._max_metric_calls is None:
            return None
        return max(0, self._max_metric_calls - state.total_num_evals - self._reserved)

    def _reserve(self, state: GEPAState, item: WorkItem, n: int) -> bool:
        """Reserve ``n`` metric calls for ``item``. False when the budget cannot fund them."""
        if self.config.reserve_budget:
            left = self._budget_left(state)
            if left is not None and n > left:
                return False
        item.reserved = n
        self._reserved += n
        return True

    def _settle(self, state: GEPAState, item: WorkItem, actual: int) -> None:
        self._reserved -= item.reserved
        item.reserved = 0
        if actual:
            state.increment_evals(actual)

    # ------------------------------------------------------------------
    # head loop pieces
    # ------------------------------------------------------------------

    def _idle(self) -> bool:
        return all(p.inflight_tasks == 0 and not p.buffer for p in self.pools.values())

    def _begin_stop(self, state: GEPAState) -> None:
        self._stopping = True
        self.events.emit("stop", mode=self.config.drain, total_metric_calls=state.total_num_evals)
        if self.config.drain == "cancel":
            for pool in self.pools.values():
                while pool.buffer:
                    self._finish(state, pool.buffer.popleft(), "cancelled")

    def _drain_completions(self, state: GEPAState) -> None:
        block = any(p.inflight_tasks for p in self.pools.values())
        try:
            first = self._completions.get(timeout=self.config.poll_interval_seconds) if block else None
        except queue.Empty:
            first = None
        pending = [first] if first is not None else []
        while True:
            try:
                pending.append(self._completions.get_nowait())
            except queue.Empty:
                break
        for completion in pending:
            self._on_completion(state, completion)
            if self._fatal is not None:
                return

    def _sample_orders(self, state: GEPAState) -> int:
        rollout = self.pools["rollout"]
        sampled = 0
        assert self.config.max_pipeline_items is not None
        while self._pipeline_items < self.config.max_pipeline_items and rollout.has_room():
            # Just-in-time: only sample what free rollout workers can start now, so the
            # parent is chosen from the freshest pool rather than aging in a buffer.
            free = (rollout.workers - rollout.inflight_tasks) * rollout.config.batch_size
            if len(rollout.buffer) >= max(free, 0):
                break
            if not self._can_fund_new_order(state):
                break
            cap = self.config.max_orders_per_version
            if cap is not None and self._orders_this_version >= cap:
                if self._pipeline_items > 0:
                    break  # wait for this generation to move the pool
                self._orders_this_version = 0  # the generation produced no commit; start another
            self._new_order(state)
            sampled += 1
            if self.stop_callback(state):
                break
        return sampled

    def _can_fund_new_order(self, state: GEPAState) -> bool:
        """Whether the remaining budget can carry one more order through to a committed candidate.

        An order costs a parent rollout and a child screen, and with probability ``a`` (the running
        acceptance rate) a validation. Orders already in the pipeline that have not reached the gate
        will claim validation budget at the same rate, so that expected claim is set aside first.
        Without this, the tail of a run keeps sampling orders it can screen but never validate.
        """
        if not self.config.reserve_budget:
            return True
        left = self._budget_left(state)
        if left is None:
            return True
        accept_rate = (self._gate_accepted + 1) / (self._gate_seen + 2)
        unscreened = sum(
            len(self.pools[s].buffer) + self.pools[s].inflight_items for s in ("rollout", "propose", "patch", "screen")
        )
        expected_claims = unscreened * accept_rate * self._last_val_cost
        return left - expected_claims >= 2 * self._last_minibatch_size + self._last_val_cost

    def _new_order(self, state: GEPAState) -> None:
        state.i += 1
        trace: dict[str, Any] = {"i": state.i, "iteration_id": new_iteration_id(), "engine": "async_stage"}
        state.full_program_trace.append(trace)
        iteration = state.i + 1
        notify_callbacks(
            self.callbacks,
            "on_iteration_start",
            IterationStartEvent(iteration=iteration, state=state, trainset_loader=self.trainset),
        )
        parent_idx = self.candidate_selector.select_candidate_idx(state)
        mb_ids = list(self.batch_sampler.next_minibatch_ids(self.trainset, state))
        item = WorkItem(
            order_id=iteration,
            iteration_id=trace["iteration_id"],
            version=self._version,
            parent_idx=parent_idx,
            parent_candidate=state.program_candidates[parent_idx],
            minibatch_ids=mb_ids,
            minibatch=self.trainset.fetch(mb_ids),
            trace=trace,
            created_at=time.monotonic(),
        )
        trace.update(selected_program_candidate=parent_idx, subsample_ids=mb_ids, pool_version=self._version)
        notify_callbacks(
            self.callbacks,
            "on_candidate_selected",
            CandidateSelectedEvent(
                iteration=iteration,
                candidate_idx=parent_idx,
                candidate=item.parent_candidate,
                score=state.program_full_scores_val_set[parent_idx],
            ),
        )
        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(iteration=iteration, minibatch_ids=mb_ids, trainset_size=len(self.trainset)),
        )
        self.logger.log(
            f"Iteration {iteration}: Selected program {parent_idx} score: "
            f"{state.program_full_scores_val_set[parent_idx]} (pool version {self._version})"
        )
        self._last_minibatch_size = len(mb_ids)
        self._orders_this_version += 1
        self._orders += 1
        self._pipeline_items += 1
        self.pools["rollout"].put(item)
        self.events.emit("sample", order=iteration, parent=parent_idx, version=self._version)

    def _resource_has_room(self, resource: str) -> bool:
        limit = self.config.resource_limits.get(resource)
        if limit is None:
            return True
        used = sum(p.inflight_tasks for p in self.pools.values() if p.resource == resource)
        return used < limit

    def _downstream_has_room(self, stage: str) -> bool:
        nxt = {"rollout": ["propose"], "propose": ["screen"], "screen": ["validate"], "patch": ["screen"]}.get(
            stage, []
        )
        if stage == "propose" and self.staleness_policy == "reflective":
            nxt = [*nxt, "patch"]
        return all(self.pools[n].has_room() for n in nxt)

    def _dispatch_all(self, state: GEPAState) -> int:
        dispatched = 0
        # Downstream first, so finished work leaves the pipeline before new work enters it.
        for name in ("validate", "screen", "patch", "propose", "rollout"):
            pool = self.pools[name]
            while pool.can_dispatch() and self._resource_has_room(pool.resource) and self._downstream_has_room(name):
                items = self._admit(state, name, pool.take_batch())
                if not items:
                    continue
                pool.submit(items, self._task(state, name, items))
                stats = self.events.stage(name)
                stats.dispatched_tasks += 1
                self.events.emit(
                    "dispatch",
                    stage=name,
                    orders=[it.order_id for it in items],
                    queue=len(pool.buffer),
                    inflight=pool.inflight_tasks,
                )
                dispatched += 1
            self.events.stage(name).max_queue_depth = max(self.events.stage(name).max_queue_depth, len(pool.buffer))
        return dispatched

    def _admit(self, state: GEPAState, stage: str, items: list[WorkItem]) -> list[WorkItem]:
        """Drop items that are too stale or that the budget cannot fund, before spending on them."""
        admitted: list[WorkItem] = []
        for item in items:
            if (
                self.staleness_policy == "guarded"
                and stage in ("propose", "screen")
                and item.gap(self._version) > self.config.max_staleness
            ):
                self._finish(state, item, "discarded_stale", gap=item.gap(self._version), at=stage)
                continue
            if stage in _EVAL_STAGES and not item.reserved:
                if stage == "validate":
                    n = len(self._validate_ids_now(item))
                else:
                    n = len(item.minibatch) * (2 if stage == "screen" and item.needs_parent_eval else 1)
                if not self._reserve(state, item, n):
                    if stage == "rollout":
                        # The budget cannot fund a fresh order; sampling more would only repeat this.
                        self._budget_blocked = True
                    self._finish(state, item, "budget", at=stage)
                    continue
            admitted.append(item)
        return admitted

    def _task(self, state: GEPAState, stage: str, items: list[WorkItem]):
        adapter = self.adapter
        if stage == "rollout":
            for it in items:
                self._notify_eval_start(state, it, candidate_idx=it.parent_idx)
            return lambda: run_rollout(adapter, items)
        if stage == "propose":
            metadatas = [
                {
                    "iteration_id": it.iteration_id,
                    "parent_iteration_id": state.iteration_id_for_candidate_idx(it.parent_idx),
                }
                for it in items
            ]
            reflector = self.reflector
            return lambda: run_propose(adapter, reflector, items, metadatas)
        if stage == "screen":
            for it in items:
                self._notify_eval_start(state, it, candidate_idx=None)
            return lambda: run_screen(adapter, items)
        if stage == "validate":
            valset = self.valset
            id_lists = [self._validate_ids_now(it) for it in items]
            for it, ids in zip(items, id_lists, strict=True):
                it.val_inflight = ids
            return lambda: run_validate(adapter, valset, items, id_lists)
        if stage == "patch":
            lm = self.raw_reflection_lm
            assert lm is not None
            jobs = [self._patch_jobs(state, it) for it in items]
            return lambda: run_patch(lm, jobs)
        raise ValueError(stage)

    # ------------------------------------------------------------------
    # completions
    # ------------------------------------------------------------------

    def _on_completion(self, state: GEPAState, c: Completion) -> None:
        pool = self.pools[c.stage]
        pool.on_complete(c)
        stats = self.events.stage(c.stage)
        stats.completed_tasks += 1
        stats.completed_items += len(c.items)
        stats.busy_seconds += c.service_seconds
        for it in c.items:
            it.stage_seconds[c.stage] = it.stage_seconds.get(c.stage, 0.0) + c.service_seconds
            stats.wait_seconds += it.wait_seconds.get(c.stage, 0.0)
        self.events.emit(
            "complete",
            stage=c.stage,
            orders=[it.order_id for it in c.items],
            seconds=round(c.service_seconds, 4),
            error=repr(c.error) if c.error else None,
        )
        if c.error is not None:
            stats.failed_tasks += 1
            self._on_task_error(state, c)
            return
        if self._stopping and self.config.drain == "cancel" and c.stage != "validate":
            for it in c.items:
                self._settle(state, it, 0)
                self._finish(state, it, "cancelled")
            return
        handler = {
            "rollout": self._after_rollout,
            "propose": self._after_propose,
            "patch": self._after_patch,
            "screen": self._after_screen,
            "validate": self._after_validate,
        }[c.stage]
        for it, result in zip(c.items, c.result, strict=True):
            try:
                handler(state, it, result)
            except Exception as e:
                self._on_item_error(state, it, e)
                if self._fatal is not None:
                    return

    def _on_task_error(self, state: GEPAState, c: Completion) -> None:
        err = c.error
        assert err is not None
        if _is_budget_exhausted(err):
            self.logger.log(f"Evaluation budget exhausted during {c.stage}; draining.")
            self._stopping = True
            for it in c.items:
                self._settle(state, it, 0)
                self._finish(state, it, "budget", at=c.stage)
            return
        for it in c.items:
            self._settle(state, it, 0)
            self._on_item_error(state, it, err)

    def _on_item_error(self, state: GEPAState, item: WorkItem, err: BaseException) -> None:
        self.logger.log(f"Iteration {item.order_id}: Exception during optimization: {err}")
        self.logger.log("".join(traceback.format_exception(type(err), err, err.__traceback__)))
        notify_callbacks(
            self.callbacks,
            "on_error",
            ErrorEvent(iteration=item.order_id, exception=err, will_continue=not self.raise_on_exception),  # type: ignore[typeddict-item]
        )
        self._finish(state, item, "error")
        if self.raise_on_exception:
            self._fatal = err

    # -- rollout -> propose ---------------------------------------------

    def _after_rollout(self, state: GEPAState, item: WorkItem, batch: EvaluationBatch) -> None:
        item.parent_eval = batch
        self._settle(state, item, metric_calls(batch, len(item.minibatch)))
        self._cache_put_train(state, item.parent_candidate, item.minibatch_ids, batch)
        self._notify_eval_end(state, item, batch, candidate_idx=item.parent_idx)
        item.trace["subsample_scores"] = list(batch.scores)

        if not batch.trajectories:
            self.logger.log(f"Iteration {item.order_id}: No trajectories for parent {item.parent_idx}. Skipping.")
            self._notify_skipped(item, "no_trajectories")
            self._finish(state, item, "skipped")
            return
        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in batch.scores)
        ):
            self.logger.log(f"Iteration {item.order_id}: All subsample scores perfect. Skipping.")
            self._notify_skipped(item, "all_scores_perfect")
            self._finish(state, item, "skipped")
            return
        item.components = self.module_selector(
            state, batch.trajectories, batch.scores, item.parent_idx, item.parent_candidate
        )
        self.pools["propose"].put(item)

    # -- propose -> (patch) -> screen -----------------------------------

    def _after_propose(self, state: GEPAState, item: WorkItem, result: ProposeResult) -> None:
        iteration = item.order_id
        if result.error is not None:
            self.logger.log(f"Iteration {iteration}: Reflection failed: {result.error}")
            self._finish(state, item, "error")
            return
        assert item.components is not None
        item.reflective_dataset = result.reflective_dataset
        concrete = {k: [dict(x) for x in v] for k, v in (result.reflective_dataset or {}).items()}
        notify_callbacks(
            self.callbacks,
            "on_reflective_dataset_built",
            ReflectiveDatasetBuiltEvent(
                iteration=iteration,
                iteration_id=item.iteration_id,
                candidate_idx=item.parent_idx,
                components=item.components,
                dataset=concrete,
            ),
        )
        notify_callbacks(
            self.callbacks,
            "on_proposal_start",
            ProposalStartEvent(
                iteration=iteration,
                parent_candidate=item.parent_candidate,
                components=item.components,
                reflective_dataset=concrete,
            ),
        )
        if not result.new_texts:
            self.logger.log(f"Iteration {iteration}: Reflection returned no text updates; skipping.")
            self._finish(state, item, "no_proposal")
            return

        meta: dict[str, Any] = {"proposal_id": f"{iteration}-0"}
        for comp in result.new_texts:
            meta[f"prompt:{comp}"] = result.prompts.get(comp, "")
            meta[f"raw_lm_output:{comp}"] = result.raw_lm_outputs.get(comp, "")
        for k, v in (result.metadata or {}).items():
            meta[f"reflection_meta:{k}" if k.startswith(("prompt:", "raw_lm_output:")) else k] = v
        item.lm_metadata = meta
        item.new_texts = dict(result.new_texts)
        for name, text in item.new_texts.items():
            self.logger.log(f"Iteration {iteration}: Proposed new text for {name}: {text}")
        notify_callbacks(
            self.callbacks,
            "on_proposal_end",
            ProposalEndEvent(
                iteration=iteration,
                new_instructions=item.new_texts,
                prompts=result.prompts,
                raw_lm_outputs=result.raw_lm_outputs,
                metadata=dict(meta),
            ),
        )
        child = dict(item.parent_candidate)
        for name, text in item.new_texts.items():
            if name not in child:
                raise KeyError(f"{name} missing in candidate")
            child[name] = text
        item.child_candidate = child

        gap = item.gap(self._version)
        if self.staleness_policy == "reflective" and gap > self.config.max_staleness:
            best_idx = self.val_evaluation_policy.get_best_program(state)
            if best_idx != item.parent_idx:
                self.events.emit("route_patch", order=iteration, gap=gap, target=best_idx)
                self.pools["patch"].put(item)
                return
        self.pools["screen"].put(item)

    def _patch_jobs(self, state: GEPAState, item: WorkItem) -> list[PatchJob]:
        """Snapshot, on the head thread, what the patch worker needs to reconcile ``item`` with the pool."""
        assert item.new_texts is not None
        best_idx = self.val_evaluation_policy.get_best_program(state)
        best = state.program_candidates[best_idx]
        item.trace["patch_target"] = best_idx
        item.trace["patch_from_version"] = item.version
        newer = state.program_candidates[item.version + 1 :]
        jobs = []
        for comp, revision in item.new_texts.items():
            history: list[str] = []
            for cand in newer:
                text = cand.get(comp, "")
                if text and (not history or history[-1] != text) and text != item.parent_candidate.get(comp):
                    history.append(text)
            jobs.append(
                PatchJob(
                    component=comp,
                    base_text=item.parent_candidate.get(comp, ""),
                    revision_text=revision,
                    current_text=best.get(comp, ""),
                    history=history[-5:],
                )
            )
        # Rebase now so the version the patch was computed against is the one recorded.
        item.lm_metadata["patch:target_idx"] = best_idx
        item.lm_metadata["patch:pool_version"] = self._version
        return jobs

    def _after_patch(self, state: GEPAState, item: WorkItem, result: PatchResult) -> None:
        for comp, raw in result.raw_outputs.items():
            item.lm_metadata[f"patch:raw_lm_output:{comp}"] = raw
        if not result.texts:
            self.events.patch_discards += 1
            self.logger.log(f"Iteration {item.order_id}: Stale proposal judged redundant; discarded.")
            self._finish(state, item, "discarded_stale", via="patch")
            return
        target_idx = int(item.lm_metadata["patch:target_idx"])
        target = state.program_candidates[target_idx]
        child = dict(target)
        child.update(result.texts)
        if child == target:
            self._finish(state, item, "discarded_stale", via="patch_noop")
            return
        self.events.patched += 1
        item.patched = True
        item.trace["patched"] = True
        item.parent_idx = target_idx
        item.parent_candidate = target
        item.child_candidate = child
        item.new_texts = dict(result.texts)
        item.version = int(item.lm_metadata["patch:pool_version"])
        # The gate compares the child with its parent on the same minibatch, and the parent changed.
        cached = self._cache_get_train(state, target, item.minibatch_ids)
        if cached is not None:
            item.parent_eval = cached
            item.needs_parent_eval = False
        else:
            item.needs_parent_eval = True
        self.logger.log(f"Iteration {item.order_id}: Stale proposal rewritten on top of program {target_idx}.")
        self.pools["screen"].put(item)

    # -- screen -> gate -> validate -------------------------------------

    def _after_screen(self, state: GEPAState, item: WorkItem, result: ScreenResult) -> None:
        assert item.child_candidate is not None
        calls = metric_calls(result.child_eval, len(item.minibatch))
        if result.parent_eval is not None:
            calls += metric_calls(result.parent_eval, len(item.minibatch))
            item.parent_eval = result.parent_eval
            item.needs_parent_eval = False
            self._cache_put_train(state, item.parent_candidate, item.minibatch_ids, result.parent_eval)
        self._settle(state, item, calls)
        item.child_eval = result.child_eval
        self._cache_put_train(state, item.child_candidate, item.minibatch_ids, result.child_eval)
        self._notify_eval_end(state, item, result.child_eval, candidate_idx=None)
        before = item.parent_eval
        assert before is not None
        item.trace["subsample_scores"] = list(before.scores)
        item.trace["new_subsample_scores"] = list(result.child_eval.scores)

        def sub(b: EvaluationBatch) -> SubsampleEvaluation:
            return SubsampleEvaluation(
                scores=b.scores,
                outputs=b.outputs,
                objective_scores=list(b.objective_scores) if b.objective_scores else None,
                trajectories=b.trajectories,
            )

        proposal = CandidateProposal(
            candidate=item.child_candidate,
            parent_program_ids=[item.parent_idx],
            subsample_indices=item.minibatch_ids,
            subsample_scores_before=before.scores,
            subsample_scores_after=result.child_eval.scores,
            eval_before=sub(before),
            eval_after=sub(result.child_eval),
            tag="reflective_mutation",
            metadata=item.lm_metadata,
        )
        item.proposal = proposal
        gap = item.gap(self._version)
        self.events.gaps_at_screen.append(gap)
        item.trace["staleness_at_screen"] = gap
        old_sum, new_sum = sum(before.scores), sum(result.child_eval.scores)
        self.experiment_tracker.log_metrics(
            {
                "subsample/before": old_sum,
                "subsample/after": new_sum,
                "async/staleness_at_screen": gap,
                "total_metric_calls": state.total_num_evals,
            },
            step=item.order_id,
        )

        self._gate_seen += 1
        if not self.acceptance_criterion.should_accept(proposal, state):
            custom = getattr(self.acceptance_criterion, "reject_reason", None)
            reason = (
                custom(proposal, state)
                if custom is not None
                else f"New subsample score {new_sum} not better than old score {old_sum}"
            )
            self.logger.log(f"Iteration {item.order_id}: {reason}, skipping")
            notify_callbacks(
                self.callbacks,
                "on_candidate_rejected",
                CandidateRejectedEvent(iteration=item.order_id, old_score=old_sum, new_score=new_sum, reason=reason),
            )
            self._finish(state, item, "rejected")
            return

        if self.staleness_policy == "guarded" and gap > self.config.max_staleness:
            self._finish(state, item, "discarded_stale", gap=gap, at="validate")
            return
        key = tuple(sorted(item.child_candidate.items()))
        if item.child_candidate in state.program_candidates or key in self._validating:
            self.logger.log(f"Iteration {item.order_id}: Candidate already in the pool or being validated; skipping.")
            self._finish(state, item, "duplicate")
            return
        self._validating.add(key)
        self.events.gaps_at_validate.append(gap)
        self.logger.log(
            f"Iteration {item.order_id}: Accepted candidate (subsample score {old_sum} -> {new_sum}); running full eval."
        )
        self._gate_accepted += 1
        self._plan_validation(state, item)
        self._last_val_cost = len(item.val_todo)
        # Claim the validation budget now, while the child waits its turn, so upstream stages
        # cannot spend what this accepted child needs.
        if not self._reserve(state, item, len(item.val_todo)):
            self._validating.discard(key)
            self._finish(state, item, "budget", at="validate")
            return
        self.pools["validate"].put(item)

    def _plan_validation(self, state: GEPAState, item: WorkItem) -> None:
        assert item.child_candidate is not None
        ids = list(self.val_evaluation_policy.get_eval_batch(self.valset, state))
        if self.config.validate_prefix_fraction is not None:
            w = self.config.validate_prefix_pass_streak
            ids.sort(key=lambda i: self._val_pass_streak.get(i, 0) >= w)  # stable: settled ids go last
        item.val_ids = ids
        cache = state.evaluation_cache
        if cache is not None:
            cached, todo = cache.get_batch(item.child_candidate, ids, split=self.valset_cache_split)
        else:
            cached, todo = {}, list(ids)
        item.val_cached = cached
        item.val_todo = list(todo)

    def _prefix_ids(self, item: WorkItem) -> list[Any]:
        frac = self.config.validate_prefix_fraction
        assert frac is not None
        return item.val_ids[: max(1, ceil(frac * len(item.val_ids)))]

    def _validate_ids_now(self, item: WorkItem) -> list[Any]:
        """The validation ids the next validate task should evaluate for ``item``."""
        remaining = [i for i in item.val_todo if i not in item.val_partial]
        if self.config.validate_prefix_fraction is None or item.val_prefix_done:
            return remaining
        prefix = set(self._prefix_ids(item))
        return [i for i in remaining if i in prefix]

    def _after_validate(self, state: GEPAState, item: WorkItem, batch: EvaluationBatch | None) -> None:
        assert item.child_candidate is not None
        evaluated, item.val_inflight = item.val_inflight, []
        calls = 0
        if batch is not None:
            calls = metric_calls(batch, len(evaluated))
            obj = list(batch.objective_scores) if batch.objective_scores else None
            for j, vid in enumerate(evaluated):
                item.val_partial[vid] = (batch.outputs[j], batch.scores[j], obj[j] if obj is not None else None)
            if state.evaluation_cache is not None:
                state.evaluation_cache.put_batch(
                    item.child_candidate, evaluated, batch.outputs, batch.scores, obj, split=self.valset_cache_split
                )
        self._settle(state, item, calls)
        item.val_metric_calls += calls

        if self.config.validate_prefix_fraction is not None and not item.val_prefix_done:
            item.val_prefix_done = True
            if not self._prefix_passes(state, item):
                self._validating.discard(tuple(sorted(item.child_candidate.items())))
                self._finish(state, item, "validation_rejected")
                return
            if self._validate_ids_now(item):
                self.pools["validate"].put(item)
                return

        outputs_by: dict[Any, Any] = {}
        scores_by: dict[Any, float] = {}
        objective_by: dict[Any, Any] | None = None
        for vid, entry in item.val_cached.items():
            outputs_by[vid], scores_by[vid] = entry.output, entry.score
            if entry.objective_scores is not None:
                objective_by = objective_by or {}
                objective_by[vid] = entry.objective_scores
        for vid, (out, score, obj_scores) in item.val_partial.items():
            outputs_by[vid], scores_by[vid] = out, score
            if obj_scores is not None:
                objective_by = objective_by or {}
                objective_by[vid] = obj_scores
        item.valset_evaluation = ValsetEvaluation(
            outputs_by_val_id=outputs_by, scores_by_val_id=scores_by, objective_scores_by_val_id=objective_by
        )
        self._commit(state, item)

    def _prefix_passes(self, state: GEPAState, item: WorkItem) -> bool:
        best_idx = self.val_evaluation_policy.get_best_program(state)
        best_scores = state.prog_candidate_val_subscores[best_idx]
        mine: list[float] = []
        theirs: list[float] = []
        for vid in self._prefix_ids(item):
            if vid not in best_scores:
                continue
            if vid in item.val_partial:
                mine.append(item.val_partial[vid][1])
            elif vid in item.val_cached:
                mine.append(item.val_cached[vid].score)
            else:
                continue
            theirs.append(best_scores[vid])
        if not mine:
            return True
        my_mean, best_mean = sum(mine) / len(mine), sum(theirs) / len(theirs)
        ok = my_mean >= best_mean - self.config.validate_prefix_margin
        self.events.emit("validate_prefix", order=item.order_id, mean=my_mean, best_mean=best_mean, passed=ok)
        if not ok:
            self.logger.log(
                f"Iteration {item.order_id}: Validation prefix mean {my_mean:.4f} below best {best_mean:.4f}; "
                "not validating the rest."
            )
        return ok

    # ------------------------------------------------------------------
    # commit: the only writer of the candidate pool
    # ------------------------------------------------------------------

    def _commit(self, state: GEPAState, item: WorkItem) -> None:
        assert item.child_candidate is not None and item.valset_evaluation is not None and item.proposal is not None
        evaluation = item.valset_evaluation
        self._validating.discard(tuple(sorted(item.child_candidate.items())))
        # Validation metric calls were already counted as they completed.
        discovery_calls = state.total_num_evals - item.val_metric_calls
        state.num_full_ds_evals += 1

        before = {p for front in state.get_pareto_front_mapping().values() for p in front}
        new_idx = state.update_state_with_new_program(
            parent_program_idx=[item.parent_idx],
            new_program=item.child_candidate,
            valset_evaluation=evaluation,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=discovery_calls,
            iteration_id=item.iteration_id,
        )
        gap = item.gap(self._version)
        self._version += 1
        self._orders_this_version = 0
        self._commits += 1
        self.events.gaps_at_commit.append(gap)
        after = {p for front in state.get_pareto_front_mapping().values() for p in front}

        score = self.val_evaluation_policy.get_valset_score(new_idx, state)
        best_idx = self.val_evaluation_policy.get_best_program(state)
        is_best = new_idx == best_idx
        iteration = item.order_id
        item.trace.update(
            new_program_idx=new_idx,
            new_program_indices=[new_idx],
            evaluated_val_indices=sorted(evaluation.scores_by_val_id.keys(), key=str),
            proposal_accepted=True,
            staleness_at_commit=gap,
        )
        for vid, s in evaluation.scores_by_val_id.items():
            front_score = state.pareto_front_valset.get(vid)
            passed = front_score is not None and s >= front_score
            self._val_pass_streak[vid] = self._val_pass_streak.get(vid, 0) + 1 if passed else 0

        notify_callbacks(
            self.callbacks,
            "on_pareto_front_updated",
            ParetoFrontUpdatedEvent(
                iteration=iteration, new_front=sorted(after), displaced_candidates=sorted(before - after)
            ),
        )
        if is_best:
            self.logger.log(f"Iteration {iteration}: Found a better program on the valset with score {score}.")
        notify_callbacks(
            self.callbacks,
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=iteration,
                candidate_idx=new_idx,
                candidate=item.child_candidate,
                scores_by_val_id=dict(evaluation.scores_by_val_id),
                average_score=score,
                num_examples_evaluated=len(evaluation.scores_by_val_id),
                total_valset_size=len(self.valset),
                parent_ids=[item.parent_idx],
                is_best_program=is_best,
                outputs_by_val_id=dict(evaluation.outputs_by_val_id) if evaluation.outputs_by_val_id else None,
            ),
        )
        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            new_program_idx=new_idx,
            valset_evaluation=evaluation,
            objective_scores=state.prog_candidate_objective_scores[new_idx],
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=best_idx,
            valset_size=len(self.valset),
            val_evaluation_policy=self.val_evaluation_policy,
        )
        self.experiment_tracker.log_metrics(
            {"async/staleness_at_commit": gap, "async/pool_version": self._version}, step=iteration
        )
        notify_callbacks(
            self.callbacks,
            "on_candidate_accepted",
            CandidateAcceptedEvent(
                iteration=iteration,
                new_candidate_idx=new_idx,
                new_score=sum(item.proposal.subsample_scores_after or []),
                parent_ids=[item.parent_idx],
            ),
        )
        self.events.emit(
            "commit", order=iteration, candidate=new_idx, parent=item.parent_idx, gap=gap, score=score, best=is_best
        )
        self._commits_since_save += 1
        self._finish(state, item, "accepted")
        self._save_if_due(state)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _finish(self, state: GEPAState, item: WorkItem, outcome: str, **fields: Any) -> None:
        if item.reserved:
            self._reserved -= item.reserved
            item.reserved = 0
        item.outcome = outcome
        item.trace["outcome"] = outcome
        item.trace.setdefault("proposal_accepted", outcome == "accepted")
        item.trace["stage_seconds"] = {k: round(v, 4) for k, v in item.stage_seconds.items()}
        item.trace["wait_seconds"] = {k: round(v, 4) for k, v in item.wait_seconds.items()}
        self.events.outcomes[outcome] += 1
        self._pipeline_items -= 1
        self.events.emit(
            "finish",
            order=item.order_id,
            outcome=outcome,
            latency=round(time.monotonic() - item.created_at, 4),
            **fields,
        )
        notify_callbacks(
            self.callbacks,
            "on_iteration_end",
            IterationEndEvent(iteration=item.order_id, state=state, proposal_accepted=outcome == "accepted"),
        )

    def _save_if_due(self, state: GEPAState, force: bool = False) -> None:
        if force or self._commits_since_save >= self.config.save_every_commits:
            self._save(state)

    def _save(self, state: GEPAState) -> None:
        if self.run_dir is None:
            return
        getter = getattr(self.adapter, "get_adapter_state", None)
        if getter is not None:
            state.adapter_state = dict(getter())
        state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
        self._commits_since_save = 0
        notify_callbacks(self.callbacks, "on_state_saved", StateSavedEvent(iteration=state.i + 1, run_dir=self.run_dir))

    def _cache_put_train(self, state: GEPAState, candidate: dict[str, str], ids: list[Any], b: EvaluationBatch) -> None:
        if state.evaluation_cache is None:
            return
        obj = list(b.objective_scores) if b.objective_scores else None
        state.evaluation_cache.put_batch(candidate, ids, b.outputs, b.scores, obj, split=TRAINSET_CACHE_SPLIT)

    def _cache_get_train(self, state: GEPAState, candidate: dict[str, str], ids: list[Any]) -> EvaluationBatch | None:
        if state.evaluation_cache is None:
            return None
        cached, todo = state.evaluation_cache.get_batch(candidate, ids, split=TRAINSET_CACHE_SPLIT)
        if todo:
            return None
        entries = [cached[i] for i in ids]
        objective = [e.objective_scores for e in entries]
        return EvaluationBatch(
            outputs=[e.output for e in entries],
            scores=[e.score for e in entries],
            trajectories=None,
            objective_scores=objective if all(o is not None for o in objective) else None,  # type: ignore[arg-type]
            num_metric_calls=0,
        )

    def _parents_of(self, state: GEPAState, idx: int) -> list[int]:
        return [p for p in state.parent_program_for_candidate[idx] if p is not None]

    def _notify_eval_start(self, state: GEPAState, item: WorkItem, candidate_idx: int | None) -> None:
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=item.order_id,
                candidate_idx=candidate_idx,
                batch_size=len(item.minibatch),
                capture_traces=True,
                parent_ids=self._parents_of(state, candidate_idx) if candidate_idx is not None else [item.parent_idx],
                inputs=item.minibatch,
                is_seed_candidate=candidate_idx == 0,
            ),
        )

    def _notify_eval_end(
        self, state: GEPAState, item: WorkItem, batch: EvaluationBatch, candidate_idx: int | None
    ) -> None:
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=item.order_id,
                candidate_idx=candidate_idx,
                scores=batch.scores,
                has_trajectories=bool(batch.trajectories),
                parent_ids=self._parents_of(state, candidate_idx) if candidate_idx is not None else [item.parent_idx],
                outputs=batch.outputs,
                trajectories=batch.trajectories,
                objective_scores=batch.objective_scores,
                is_seed_candidate=candidate_idx == 0,
            ),
        )

    def _notify_skipped(self, item: WorkItem, reason: str) -> None:
        assert item.parent_eval is not None
        notify_callbacks(
            self.callbacks,
            "on_evaluation_skipped",
            EvaluationSkippedEvent(
                iteration=item.order_id,
                candidate_idx=item.parent_idx,
                reason=reason,
                scores=item.parent_eval.scores,
                is_seed_candidate=item.parent_idx == 0,
            ),
        )


def describe_pipeline() -> Mapping[str, str]:
    """Stage name -> the resource it draws from, in pipeline order."""
    return {s: STAGE_RESOURCE[s] for s in PIPELINE_STAGES}
