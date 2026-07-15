"""Ensemble helpers — compose multiple engines over the same task.

Two flavors of each helper:

- ``optimize_*`` — convenience. Mirror
  :func:`gepa.optimize_anything.optimize_anything`'s flat shape
  (``seed_candidate`` positional, ``evaluator=`` / ``objective=`` /
  ``background=`` / ``dataset=`` / ``valset=`` / ``test_set=`` keywords) plus
  ``configs=``, one :class:`OptimizeAnythingConfig` per engine. Build a fresh
  :class:`EvalServer` per engine internally (each with its own budget from
  ``OptimizeAnythingConfig.max_evals`` / ``OptimizeAnythingConfig.max_token_cost``).
  Budgets are pre-partitioned per config; if an engine finishes early its
  leftover budget is not redistributed.
- ``optimize_*_with_server`` — primitive. Caller owns the
  :class:`EvalServer`(s). All four helpers take **one server per config**
  (one :class:`BudgetTracker` per engine or per stage), so budget is
  fair-shared across the composition — pre-partition ``total / N`` the
  way you want it spent. Useful when an outer framework (e.g. terrarium)
  wants every inner eval routed through its own server while still
  composing multiple engines.

Helpers
-------
- :func:`optimize_sequential` / :func:`optimize_sequential_with_server` —
  pipeline. Each engine's best becomes the next engine's seed.
  Monotonic: an engine that regresses doesn't poison the chain (the
  running best is carried forward).
- :func:`optimize_adaptive_sequential_with_server` — rotate through engines
  on score plateaus while sharing one eval budget and carrying the running
  best forward.
- :func:`optimize_parallel` / :func:`optimize_parallel_with_server` —
  run N engines concurrently, return all results.
- :func:`optimize_best_of` / :func:`optimize_best_of_with_server` —
  parallel + ``max(results, key=best_score)``.
- :func:`optimize_vote` / :func:`optimize_vote_with_server` — parallel,
  then re-score each engine's ``best_candidate`` once via ``evaluate``
  (outside any budget) and return the candidate with the highest
  re-score. Useful when engines have disjoint scoring quirks and you
  want a final consensus.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engine import Result
from gepa.oa.task import Task

if TYPE_CHECKING:
    from gepa.oa.eval_server import EvalServer


def optimize_anything_from_task(
    task: Task,
    evaluate: Callable[..., tuple[float, dict[str, Any]]],
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Optimize a prebuilt :class:`Task` by unpacking it into :func:`optimize_anything`.

    A convenience for callers that already hold a ``Task`` — e.g. the ensemble
    helpers, which carry one ``Task`` forward across stages. The task's ``name``
    is preserved by threading it through ``config`` when ``config.name`` is unset.
    """
    # Lazy import avoids a circular import (gepa.optimize_anything imports this module).
    from gepa.optimize_anything import optimize_anything

    if config is None:
        config = OptimizeAnythingConfig()
    if config.name is None:
        config = replace(config, name=task.name)
    return optimize_anything(
        task.seed_candidate,
        evaluator=evaluate,
        dataset=task.train_set,
        valset=task.val_set,
        objective=task.objective,
        background=task.background,
        test_set=task.test_set,
        config=config,
    )


def optimize_anything_with_server(
    server: EvalServer,
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Run optimization against a caller-owned :class:`EvalServer`.

    The caller is responsible for ``server.start()`` and ``server.stop()``.
    Server-shaped config fields such as ``max_evals``, ``output_dir``, and
    ``tracker`` are ignored because they already belong to the server.
    """
    # Lazy import avoids a circular import (gepa.optimize_anything imports this module).
    from gepa.optimize_anything import _build_engine, _run_engine

    if config is None:
        config = OptimizeAnythingConfig()
    engine = _build_engine(config)
    return _run_engine(server, engine, owns_server=False)


def _build_task(
    seed_candidate: str | None,
    *,
    objective: str | None,
    background: str | None,
    dataset: list[Any] | None,
    valset: list[Any] | None,
    test_set: list[Any] | None,
    name: str | None,
) -> Task:
    """Build the :class:`Task` an ensemble threads across its stages.

    ``name`` groups every stage's outputs under one directory
    (``outputs/optimize_anything/<name>/<engine>/<timestamp>``); when omitted,
    a shared pipeline name is generated so the stages still land together.
    """
    if name is None:
        name = f"pipeline-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return Task(
        name=name,
        seed_candidate=seed_candidate,
        objective=objective or "",
        background=background or "",
        train_set=dataset,
        val_set=valset,
        test_set=test_set,
    )


def optimize_sequential(
    seed_candidate: str | None = None,
    *,
    evaluator: Callable[..., tuple[float, dict[str, Any]]],
    configs: list[OptimizeAnythingConfig],
    dataset: list[Any] | None = None,
    valset: list[Any] | None = None,
    objective: str | None = None,
    background: str | None = None,
    test_set: list[Any] | None = None,
    name: str | None = None,
) -> Result:
    """Run ``configs`` in order; each engine's best becomes the next seed.

    Takes the same task fields as :func:`gepa.optimize_anything.optimize_anything`,
    plus ``configs`` — one :class:`OptimizeAnythingConfig` per stage.

    Monotonic: if engine ``k+1`` regresses, the chain still feeds the
    running-best candidate (not engine ``k``'s output verbatim) to
    engine ``k+2``.

    Returns the :class:`Result` of the last engine run, with metadata
    augmented by ``all_results`` (one Result per config, stage-ordered) and
    ``best_stage_score`` / ``best_stage_candidate``.
    """
    task = _build_task(
        seed_candidate,
        objective=objective,
        background=background,
        dataset=dataset,
        valset=valset,
        test_set=test_set,
        name=name,
    )
    return _sequential_from_task(task, evaluator, configs)


def _sequential_from_task(
    task: Task,
    evaluate: Callable[..., tuple[float, dict[str, Any]]],
    configs: list[OptimizeAnythingConfig],
) -> Result:
    if not configs:
        raise ValueError("optimize_sequential requires at least one config")

    all_results: list[Result] = []
    current_task = task
    best_so_far: Result | None = None

    for cfg in configs:
        result = optimize_anything_from_task(current_task, evaluate, cfg)
        all_results.append(result)
        if best_so_far is None or result.best_score > best_so_far.best_score:
            best_so_far = result
        # Always feed the running best forward, not the latest run's output —
        # protects the pipeline from a single regressing stage.
        assert best_so_far is not None
        current_task = replace(current_task, seed_candidate=best_so_far.best_candidate)

    assert best_so_far is not None
    final = all_results[-1]
    final.metadata["all_results"] = all_results
    final.metadata["best_stage_score"] = best_so_far.best_score
    final.metadata["best_stage_candidate"] = best_so_far.best_candidate
    return final


def optimize_parallel(
    seed_candidate: str | None = None,
    *,
    evaluator: Callable[..., tuple[float, dict[str, Any]]],
    configs: list[OptimizeAnythingConfig],
    dataset: list[Any] | None = None,
    valset: list[Any] | None = None,
    objective: str | None = None,
    background: str | None = None,
    test_set: list[Any] | None = None,
    name: str | None = None,
    max_workers: int | None = None,
) -> list[Result]:
    """Run each config concurrently against the same task.

    Takes the same task fields as :func:`gepa.optimize_anything.optimize_anything`,
    plus ``configs`` — one :class:`OptimizeAnythingConfig` per engine.

    Each call gets its own :class:`EvalServer` and budget — pre-partition
    ``max_evals`` / ``max_token_cost`` across the configs the way you want
    them spent. Returns the raw list of results (same order as ``configs``).

    ``max_workers`` defaults to ``len(configs)``.
    """
    task = _build_task(
        seed_candidate,
        objective=objective,
        background=background,
        dataset=dataset,
        valset=valset,
        test_set=test_set,
        name=name,
    )
    return _parallel_from_task(task, evaluator, configs, max_workers=max_workers)


def _parallel_from_task(
    task: Task,
    evaluate: Callable[..., tuple[float, dict[str, Any]]],
    configs: list[OptimizeAnythingConfig],
    *,
    max_workers: int | None = None,
) -> list[Result]:
    if not configs:
        raise ValueError("optimize_parallel requires at least one config")

    workers = max_workers if max_workers is not None else len(configs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: optimize_anything_from_task(task, evaluate, c), configs))


def optimize_best_of(
    seed_candidate: str | None = None,
    *,
    evaluator: Callable[..., tuple[float, dict[str, Any]]],
    configs: list[OptimizeAnythingConfig],
    dataset: list[Any] | None = None,
    valset: list[Any] | None = None,
    objective: str | None = None,
    background: str | None = None,
    test_set: list[Any] | None = None,
    name: str | None = None,
    max_workers: int | None = None,
) -> Result:
    """Run all configs in parallel, return the result with the highest ``best_score``.

    Takes the same task fields as :func:`gepa.optimize_anything.optimize_anything`,
    plus ``configs`` — one :class:`OptimizeAnythingConfig` per engine.

    ``best_score`` comes from each engine's own scoring during its run — no
    re-scoring. Use :func:`optimize_vote` if you want a fair cross-engine
    comparison via a single re-score pass.
    """
    task = _build_task(
        seed_candidate,
        objective=objective,
        background=background,
        dataset=dataset,
        valset=valset,
        test_set=test_set,
        name=name,
    )
    results = _parallel_from_task(task, evaluator, configs, max_workers=max_workers)
    winner = max(results, key=lambda r: r.best_score)
    winner.metadata["all_results"] = results
    return winner


def optimize_vote(
    seed_candidate: str | None = None,
    *,
    evaluator: Callable[..., tuple[float, dict[str, Any]]],
    configs: list[OptimizeAnythingConfig],
    dataset: list[Any] | None = None,
    valset: list[Any] | None = None,
    objective: str | None = None,
    background: str | None = None,
    test_set: list[Any] | None = None,
    name: str | None = None,
    max_workers: int | None = None,
) -> Result:
    """Run all configs in parallel, then re-score each engine's best candidate
    once via ``evaluator`` (outside any budget) and return the highest-scoring.

    Takes the same task fields as :func:`gepa.optimize_anything.optimize_anything`,
    plus ``configs`` — one :class:`OptimizeAnythingConfig` per engine.

    Unlike :func:`optimize_best_of`, the re-score pass uses a single shared
    scoring function, so engines with different internal eval semantics get
    compared fairly. The returned :class:`Result` is the original winning
    engine's result with metadata augmented:

    - ``vote_scores``: list of re-scores aligned with ``configs``.
    - ``vote_candidates``: each engine's best_candidate.
    - ``vote_winner_idx``: index into ``configs`` of the winning engine.
    - ``all_results``: the raw list of engine results.
    """
    task = _build_task(
        seed_candidate,
        objective=objective,
        background=background,
        dataset=dataset,
        valset=valset,
        test_set=test_set,
        name=name,
    )
    results = _parallel_from_task(task, evaluator, configs, max_workers=max_workers)
    vote_scores: list[float] = []
    for r in results:
        try:
            score, _ = evaluator(r.best_candidate)
        except Exception:
            score = float("-inf")
        vote_scores.append(float(score))
    winner_idx = max(range(len(results)), key=lambda i: vote_scores[i])
    winner = results[winner_idx]
    winner.metadata["vote_scores"] = vote_scores
    winner.metadata["vote_candidates"] = [r.best_candidate for r in results]
    winner.metadata["vote_winner_idx"] = winner_idx
    winner.metadata["all_results"] = results
    return winner


# ── _with_server variants ──────────────────────────────────────────────
#
# Mirror the four convenience helpers above, but caller owns the
# :class:`EvalServer`(s). All four take ``list[EvalServer]`` (one server per
# config) so budget is fair-shared across the composition — pre-partition
# ``total / N`` the way you want it spent.
#
# Caller is responsible for ``server.start()`` / ``server.stop()``;
# these helpers do not touch lifecycle.


def optimize_sequential_with_server(
    servers: list[EvalServer],
    configs: list[OptimizeAnythingConfig],
) -> Result:
    """Sequential variant for caller-owned :class:`EvalServer`s.

    ``servers`` and ``configs`` are zipped — stage ``i`` runs config ``i``
    against server ``i``. Each server has its own :class:`BudgetTracker`,
    so budget is **fair-shared across stages** (caller pre-partitions, e.g.
    ``total / N`` per stage). A productive early stage cannot monopolize the
    budget pool — symmetric with the parallel-family helpers.

    Between stages the function seeds the next stage's
    ``seed_candidate`` with the running-best candidate by mutating
    ``servers[i].task`` in place, exactly like :func:`optimize_sequential`.
    The mutation is benign: only ``seed_candidate`` changes, and a server
    doesn't read it again after construction (only the next engine does,
    when picking up the seed).

    Returns the :class:`Result` of the last stage with ``all_results``
    (stage-ordered), ``best_stage_score``, and ``best_stage_candidate``
    in metadata.
    """
    if not configs:
        raise ValueError("optimize_sequential_with_server requires at least one config")
    if len(servers) != len(configs):
        raise ValueError(f"servers and configs must have the same length (got {len(servers)} and {len(configs)})")

    original_tasks = [s.task for s in servers]
    all_results: list[Result] = []
    best_so_far: Result | None = None

    try:
        for i, cfg in enumerate(configs):
            srv = servers[i]
            # Seed this stage with the running-best candidate, if any.
            if best_so_far is not None:
                srv.task = replace(srv.task, seed_candidate=best_so_far.best_candidate)
            result = optimize_anything_with_server(srv, cfg)
            all_results.append(result)
            if best_so_far is None or result.best_score > best_so_far.best_score:
                best_so_far = result
    finally:
        for s, t in zip(servers, original_tasks, strict=True):
            s.task = t

    assert best_so_far is not None
    final = all_results[-1]
    final.metadata["all_results"] = all_results
    final.metadata["best_stage_score"] = best_so_far.best_score
    final.metadata["best_stage_candidate"] = best_so_far.best_candidate
    return final


def optimize_adaptive_sequential_with_server(
    server: EvalServer,
    configs: list[OptimizeAnythingConfig],
    *,
    plateau_evals: int,
    patience: int = 1,
    min_evals_per_stage: int = 0,
    improvement_epsilon: float = 0.0,
    cycle: bool = True,
    max_switches: int | None = None,
) -> Result:
    """Run engine slices and switch when the current engine plateaus.

    This is a scheduler over existing optimize_anything engines, not a new optimizer. It
    repeatedly gives the active engine a bounded eval slice, observes whether
    that engine's aggregate ``Result.best_score`` improved the scheduler's
    running best, and switches to the next engine after ``patience``
    non-improving slices once ``min_evals_per_stage`` has been spent in the
    current stage.

    All slices share ``server`` and therefore one hard eval budget. The running
    best candidate is always fed forward, so a regressing slice cannot poison a
    later engine. The returned result is the last executed slice augmented
    with scheduler metadata and ``best_stage_candidate`` / ``best_stage_score``
    for the best aggregate engine result seen by the scheduler. This avoids
    treating per-example ``EvalServer.best_score`` as a candidate-level score
    on dataset tasks.
    """
    if not configs:
        raise ValueError("optimize_adaptive_sequential_with_server requires at least one config")
    if plateau_evals <= 0:
        raise ValueError("plateau_evals must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if min_evals_per_stage < 0:
        raise ValueError("min_evals_per_stage cannot be negative")

    original_task = server.task
    original_max_evals = server.budget.max_evals
    stage_results: list[Result] = []
    schedule: list[dict[str, object]] = []
    best_so_far: Result | None = None
    current_idx = 0
    current_stage_evals = 0
    no_improvement_slices = 0
    switches = 0
    idle_slices_in_round = 0
    adapter_cost = 0.0

    try:
        while not server.budget.exhausted:
            if current_idx >= len(configs):
                break

            remaining = server.budget.remaining
            if remaining is not None and remaining <= 0:
                break
            if stage_results and remaining is not None and remaining < min_evals_per_stage:
                break
            slice_evals = plateau_evals if remaining is None else min(plateau_evals, remaining)
            if slice_evals <= 0:
                break

            # The scheduler shares one *eval* budget across stages (carved into
            # per-slice windows here). Proposer cost is NOT shared: each stage's
            # engine reads its own ``configs[i].max_token_cost`` and enforces it
            # itself. ``adapter_cost`` is accumulated only for reporting.
            server.budget.max_evals = (
                server.budget.used + slice_evals
                if original_max_evals is None
                else min(original_max_evals, server.budget.used + slice_evals)
            )

            before_evals = server.budget.used
            before_best = best_so_far.best_score if best_so_far is not None else float("-inf")
            result = optimize_anything_with_server(server, configs[current_idx])
            server.budget.max_evals = original_max_evals
            after_evals = server.budget.used
            eval_delta = after_evals - before_evals
            current_stage_evals += eval_delta
            stage_results.append(result)
            result_cost = float(result.metadata.get("adapter_cost", 0.0))
            adapter_cost += result_cost

            improved = result.best_score > before_best + improvement_epsilon
            if best_so_far is None or result.best_score > best_so_far.best_score + improvement_epsilon:
                best_so_far = result

            if improved:
                no_improvement_slices = 0
                idle_slices_in_round = 0
            else:
                no_improvement_slices += 1
                idle_slices_in_round = idle_slices_in_round + 1 if eval_delta == 0 else 0

            schedule.append(
                {
                    "engine_idx": current_idx,
                    "engine": str(configs[current_idx].engine),
                    "eval_start": before_evals,
                    "eval_end": after_evals,
                    "eval_delta": eval_delta,
                    "best_before": before_best,
                    "best_after": best_so_far.best_score if best_so_far is not None else float("-inf"),
                    "improved": improved,
                    "adapter_cost": result_cost,
                }
            )

            if idle_slices_in_round >= len(configs):
                break

            should_switch = (
                no_improvement_slices >= patience and current_stage_evals >= min_evals_per_stage and len(configs) > 1
            )
            if should_switch:
                if max_switches is not None and switches >= max_switches:
                    break
                next_idx = current_idx + 1
                if next_idx >= len(configs):
                    if not cycle:
                        break
                    next_idx = 0
                switches += 1
                current_idx = next_idx
                current_stage_evals = 0
                no_improvement_slices = 0

            # Always seed the next slice with the global best from the shared
            # server. This keeps scheduler semantics monotonic across engines.
            if best_so_far is not None:
                server.task = replace(server.task, seed_candidate=best_so_far.best_candidate)
    finally:
        server.budget.max_evals = original_max_evals
        server.task = original_task

    if not stage_results:
        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={
                "stage_results": [],
                "adaptive_schedule": schedule,
                "adaptive_stop_reason": "budget_exhausted" if server.budget.exhausted else "no_slices_run",
            },
        )

    final = stage_results[-1]
    if best_so_far is not None:
        final.best_candidate = best_so_far.best_candidate
        final.best_score = best_so_far.best_score
    final.metadata["stage_results"] = stage_results
    final.metadata["adaptive_schedule"] = schedule
    final.metadata["adaptive_switches"] = switches
    final.metadata["adapter_cost"] = adapter_cost
    final.metadata["best_stage_score"] = best_so_far.best_score if best_so_far is not None else final.best_score
    final.metadata["best_stage_candidate"] = (
        best_so_far.best_candidate if best_so_far is not None else final.best_candidate
    )
    final.metadata["adaptive_stop_reason"] = "budget_exhausted" if server.budget.exhausted else "scheduler_stopped"
    return final


def optimize_parallel_with_server(
    servers: list[EvalServer],
    configs: list[OptimizeAnythingConfig],
    *,
    max_workers: int | None = None,
) -> list[Result]:
    """Parallel variant for caller-owned :class:`EvalServer`s.

    ``servers`` and ``configs`` are zipped — server ``i`` runs config ``i``.
    Each server has its own :class:`BudgetTracker`; pre-partition the budget
    across servers the way you want it spent.

    ``max_workers`` defaults to ``len(configs)``.
    """
    if not configs:
        raise ValueError("optimize_parallel_with_server requires at least one config")
    if len(servers) != len(configs):
        raise ValueError(f"servers and configs must have the same length (got {len(servers)} and {len(configs)})")

    workers = max_workers if max_workers is not None else len(configs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda sc: optimize_anything_with_server(sc[0], sc[1]),
                zip(servers, configs, strict=True),
            )
        )


def optimize_best_of_with_server(
    servers: list[EvalServer],
    configs: list[OptimizeAnythingConfig],
    *,
    max_workers: int | None = None,
) -> Result:
    """Best-of variant for caller-owned :class:`EvalServer`s.

    Run all (server, config) pairs in parallel, return the result with
    the highest ``best_score``. ``best_score`` comes from each engine's
    own scoring during its run — no re-scoring. Use
    :func:`optimize_vote_with_server` for cross-engine re-score.
    """
    results = optimize_parallel_with_server(servers, configs, max_workers=max_workers)
    winner = max(results, key=lambda r: r.best_score)
    winner.metadata["all_results"] = results
    return winner


def optimize_vote_with_server(
    servers: list[EvalServer],
    configs: list[OptimizeAnythingConfig],
    evaluate: Callable[..., tuple[float, dict[str, Any]]],
    *,
    max_workers: int | None = None,
) -> Result:
    """Vote variant for caller-owned :class:`EvalServer`s.

    Run all (server, config) pairs in parallel, then re-score each
    engine's ``best_candidate`` once via ``evaluate`` (outside any
    budget) and return the highest-scoring. ``evaluate`` is called
    directly — it does not go through any of the inner servers, so the
    re-score pass does not consume their budgets.

    Returned :class:`Result` is the winning engine's result with
    ``vote_scores`` / ``vote_candidates`` / ``vote_winner_idx`` /
    ``all_results`` in metadata.
    """
    results = optimize_parallel_with_server(servers, configs, max_workers=max_workers)
    vote_scores: list[float] = []
    for r in results:
        try:
            score, _ = evaluate(r.best_candidate)
        except Exception:
            score = float("-inf")
        vote_scores.append(float(score))
    winner_idx = max(range(len(results)), key=lambda i: vote_scores[i])
    winner = results[winner_idx]
    winner.metadata["vote_scores"] = vote_scores
    winner.metadata["vote_candidates"] = [r.best_candidate for r in results]
    winner.metadata["vote_winner_idx"] = winner_idx
    winner.metadata["all_results"] = results
    return winner
