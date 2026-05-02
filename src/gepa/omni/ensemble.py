"""Ensemble helpers — compose multiple backends over the same task.

Two flavors of each helper:

- ``optimize_*`` — convenience. Take ``(task, evaluate, configs)``, build
  a fresh :class:`EvalServer` per backend internally (each with its own
  budget from ``OmniConfig.max_evals`` / ``OmniConfig.max_token_cost``).
  Budgets are pre-partitioned per config; if a backend finishes early its
  leftover budget is not redistributed.
- ``optimize_*_with_server`` — primitive. Caller owns the
  :class:`EvalServer`(s). For the parallel-family helpers the caller
  pre-builds one server per config (one budget partition per backend).
  For sequential the caller pre-builds **one** server that all stages
  share (single budget pool — per-config budgets are ignored). Useful
  when an outer framework (e.g. terrarium) wants every inner eval routed
  through its own server while still composing multiple backends.

Helpers
-------
- :func:`optimize_sequential` / :func:`optimize_sequential_with_server` —
  pipeline. Each backend's best becomes the next backend's seed.
  Monotonic: a backend that regresses doesn't poison the chain (the
  running best is carried forward).
- :func:`optimize_parallel` / :func:`optimize_parallel_with_server` —
  run N backends concurrently, return all results.
- :func:`optimize_best_of` / :func:`optimize_best_of_with_server` —
  parallel + ``max(results, key=best_score)``.
- :func:`optimize_vote` / :func:`optimize_vote_with_server` — parallel,
  then re-score each backend's ``best_candidate`` once via ``evaluate``
  (outside any budget) and return the candidate with the highest
  re-score. Useful when backends have disjoint scoring quirks and you
  want a final consensus.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING

from gepa.omni.api import optimize_anything, optimize_anything_with_server

if TYPE_CHECKING:
    from gepa.omni.backend import Result
    from gepa.omni.config import OmniConfig
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import EvalFn, Task


def optimize_sequential(
    task: Task,
    evaluate: EvalFn,
    configs: list[OmniConfig],
) -> Result:
    """Run ``configs`` in order; each backend's best becomes the next seed.

    Monotonic: if backend ``k+1`` regresses, the chain still feeds the
    running-best candidate (not backend ``k``'s output verbatim) to
    backend ``k+2``.

    Returns the :class:`Result` of the last backend run, with metadata
    augmented by ``stage_results`` (one Result per config) and
    ``best_stage_score``.
    """
    if not configs:
        raise ValueError("optimize_sequential requires at least one config")

    stage_results: list[Result] = []
    current_task = task
    best_so_far: Result | None = None

    for cfg in configs:
        result = optimize_anything(current_task, evaluate, cfg)
        stage_results.append(result)
        if best_so_far is None or result.best_score > best_so_far.best_score:
            best_so_far = result
        # Always feed the running best forward, not the latest run's output —
        # protects the pipeline from a single regressing stage.
        assert best_so_far is not None
        current_task = replace(current_task, initial_candidate=best_so_far.best_candidate)

    assert best_so_far is not None
    final = stage_results[-1]
    final.metadata["stage_results"] = stage_results
    final.metadata["best_stage_score"] = best_so_far.best_score
    final.metadata["best_stage_candidate"] = best_so_far.best_candidate
    return final


def optimize_parallel(
    task: Task,
    evaluate: EvalFn,
    configs: list[OmniConfig],
    *,
    max_workers: int | None = None,
) -> list[Result]:
    """Run each config concurrently against the same task.

    Each call gets its own :class:`EvalServer` and budget — pre-partition
    ``max_evals`` / ``max_token_cost`` across the configs the way you want
    them spent. Returns the raw list of results (same order as ``configs``).

    ``max_workers`` defaults to ``len(configs)``.
    """
    if not configs:
        raise ValueError("optimize_parallel requires at least one config")

    workers = max_workers if max_workers is not None else len(configs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: optimize_anything(task, evaluate, c), configs))


def optimize_best_of(
    task: Task,
    evaluate: EvalFn,
    configs: list[OmniConfig],
    *,
    max_workers: int | None = None,
) -> Result:
    """Run all configs in parallel, return the result with the highest ``best_score``.

    ``best_score`` comes from each backend's own scoring during its run — no
    re-scoring. Use :func:`optimize_vote` if you want a fair cross-backend
    comparison via a single re-score pass.
    """
    results = optimize_parallel(task, evaluate, configs, max_workers=max_workers)
    winner = max(results, key=lambda r: r.best_score)
    winner.metadata["all_results"] = results
    return winner


def optimize_vote(
    task: Task,
    evaluate: EvalFn,
    configs: list[OmniConfig],
    *,
    max_workers: int | None = None,
) -> Result:
    """Run all configs in parallel, then re-score each backend's best candidate
    once via ``evaluate`` (outside any budget) and return the highest-scoring.

    Unlike :func:`optimize_best_of`, the re-score pass uses a single shared
    scoring function, so backends with different internal eval semantics get
    compared fairly. The returned :class:`Result` is the original winning
    backend's result with metadata augmented:

    - ``vote_scores``: list of re-scores aligned with ``configs``.
    - ``vote_candidates``: each backend's best_candidate.
    - ``vote_winner_idx``: index into ``configs`` of the winning backend.
    - ``all_results``: the raw list of backend results.
    """
    results = optimize_parallel(task, evaluate, configs, max_workers=max_workers)
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


# ── _with_server variants ──────────────────────────────────────────────
#
# Mirror the four convenience helpers above, but caller owns the
# :class:`EvalServer`(s). The parallel-family helpers take a list of
# servers (one per config — pre-partitioned budgets). Sequential takes a
# single shared server (one budget pool — per-config budgets ignored).
#
# Caller is responsible for ``server.start()`` / ``server.stop()``;
# these helpers do not touch lifecycle.


def optimize_sequential_with_server(
    server: EvalServer,
    configs: list[OmniConfig],
) -> Result:
    """Sequential variant for a caller-owned :class:`EvalServer`.

    All stages share ``server`` and thus a single budget pool — per-config
    ``max_evals`` / ``max_token_cost`` are ignored (the server's
    :class:`BudgetTracker` is the single hard cap).

    Between stages the function mutates ``server.task`` to seed the next
    stage's ``initial_candidate`` with the running-best candidate, exactly
    like :func:`optimize_sequential`. The mutation is benign: only
    ``initial_candidate`` changes, and the server doesn't read it again
    after construction (only the next backend does, when picking up the
    seed).

    Returns the :class:`Result` of the last stage with ``stage_results``,
    ``best_stage_score``, and ``best_stage_candidate`` in metadata.
    """
    if not configs:
        raise ValueError("optimize_sequential_with_server requires at least one config")

    original_task = server.task
    stage_results: list[Result] = []
    best_so_far: Result | None = None

    try:
        for cfg in configs:
            result = optimize_anything_with_server(server, cfg)
            stage_results.append(result)
            if best_so_far is None or result.best_score > best_so_far.best_score:
                best_so_far = result
            # Always feed the running best forward, not the latest run's
            # output — protects the pipeline from a single regressing stage.
            assert best_so_far is not None
            server.task = replace(server.task, initial_candidate=best_so_far.best_candidate)
    finally:
        server.task = original_task

    assert best_so_far is not None
    final = stage_results[-1]
    final.metadata["stage_results"] = stage_results
    final.metadata["best_stage_score"] = best_so_far.best_score
    final.metadata["best_stage_candidate"] = best_so_far.best_candidate
    return final


def optimize_parallel_with_server(
    servers: list[EvalServer],
    configs: list[OmniConfig],
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
    configs: list[OmniConfig],
    *,
    max_workers: int | None = None,
) -> Result:
    """Best-of variant for caller-owned :class:`EvalServer`s.

    Run all (server, config) pairs in parallel, return the result with
    the highest ``best_score``. ``best_score`` comes from each backend's
    own scoring during its run — no re-scoring. Use
    :func:`optimize_vote_with_server` for cross-backend re-score.
    """
    results = optimize_parallel_with_server(servers, configs, max_workers=max_workers)
    winner = max(results, key=lambda r: r.best_score)
    winner.metadata["all_results"] = results
    return winner


def optimize_vote_with_server(
    servers: list[EvalServer],
    configs: list[OmniConfig],
    evaluate: EvalFn,
    *,
    max_workers: int | None = None,
) -> Result:
    """Vote variant for caller-owned :class:`EvalServer`s.

    Run all (server, config) pairs in parallel, then re-score each
    backend's ``best_candidate`` once via ``evaluate`` (outside any
    budget) and return the highest-scoring. ``evaluate`` is called
    directly — it does not go through any of the inner servers, so the
    re-score pass does not consume their budgets.

    Returned :class:`Result` is the winning backend's result with
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
