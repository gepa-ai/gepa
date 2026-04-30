"""Ensemble helpers — compose multiple backends over the same task.

All helpers take the same ``(task, evaluate, configs)`` shape and call
:func:`gepa.omni.optimize_anything` under the hood. Budgets are pre-partitioned
per :class:`OmniConfig`; there's no shared budget pool across backends. If a
backend finishes early, its leftover budget is not redistributed — split the
budget the way you want it spent.

Helpers
-------
- :func:`optimize_sequential` — pipeline. Each backend's best becomes the next
  backend's seed. Monotonic: a backend that regresses doesn't poison the
  chain (the running best is carried forward).
- :func:`optimize_parallel` — run N backends concurrently, return all results.
- :func:`optimize_best_of` — parallel + ``max(results, key=best_score)``.
- :func:`optimize_vote` — parallel, then re-score each backend's
  ``best_candidate`` once via ``evaluate`` (outside any budget) and return the
  candidate with the highest re-score. Useful when backends have disjoint
  scoring quirks and you want a final consensus.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING

from gepa.omni.api import optimize_anything

if TYPE_CHECKING:
    from gepa.omni.backend import Result
    from gepa.omni.config import OmniConfig
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
