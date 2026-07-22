"""Public ``optimize_anything`` API.

A call has three parts:

- **What to optimize** — the task fields passed directly to
  :func:`optimize_anything` (``seed_candidate``, ``objective``,
  ``background``, and optional ``dataset`` / ``valset`` / ``test_set``).
- **How to score it** — ``evaluator``, a function returning ``(score, info)``.
- **How to optimize** — :class:`OptimizeAnythingConfig`, which chooses the
  :class:`Engine`, names the run, and sets the budget.

Example:

    from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything

    def evaluate(candidate: str) -> tuple[float, dict]:
        return run_candidate(candidate), {"diagnostics": "..."}

    result = optimize_anything(
        "initial prompt or program",
        evaluator=evaluate,
        objective="Improve the candidate's score.",
        config=OptimizeAnythingConfig(engine="gepa", max_evals=100),
    )
"""

from __future__ import annotations

import time
import uuid
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from gepa.oa.budget import BudgetExhausted, BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engine import Engine, Result
from gepa.oa.engines import AutoResearchEngine, BestOfNEngine, GepaEngine, MetaHarnessEngine
from gepa.oa.ensemble import (
    optimize_adaptive_sequential,
    optimize_adaptive_sequential_with_server,
    optimize_anything_from_task,
    optimize_anything_with_server,
    optimize_best_of,
    optimize_best_of_with_server,
    optimize_parallel,
    optimize_parallel_with_server,
    optimize_sequential,
    optimize_sequential_with_server,
    optimize_vote,
    optimize_vote_with_server,
)
from gepa.oa.eval_server import EvalServer, _resolve_id
from gepa.oa.proposers import ClaudeCodeAgentProposer
from gepa.oa.registry import (
    get_engine_cls,
    get_task,
    list_engines,
    list_tasks,
    register_engine,
    register_task,
    register_task_factory,
)
from gepa.oa.task import Task


def optimize_anything(
    seed_candidate: str | None = None,
    *,
    evaluator: Callable[..., tuple[float, dict[str, Any]]],
    dataset: list[Any] | None = None,
    valset: list[Any] | None = None,
    objective: str | None = None,
    background: str | None = None,
    test_set: list[Any] | None = None,
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Optimize a text candidate (prompt, code, instructions, ...) against a score.

    The signature mirrors :func:`gepa.gepa_launcher.optimize_anything`
    (``seed_candidate`` / ``evaluator`` / ``dataset`` / ``valset`` / ``objective``
    / ``background``) so the two entry points share one shape; the new API only
    swaps ``config`` for an :class:`OptimizeAnythingConfig` and adds ``test_set``.

    Args:
        seed_candidate: The seed text to evolve from. ``None`` for seedless
            mode, where the engine bootstraps the first candidate from
            ``objective`` / ``background``.
        evaluator: Scoring function returning ``(score, info)``. Use
            ``(candidate) -> (score, info)`` for a single task, or
            ``(candidate, example) -> (score, info)`` when dataset/val/test
            examples are provided. ``score`` is a float (higher is better) and
            ``info`` is a free-form dict surfaced to the engine as feedback.
        dataset: Optional training examples used during optimization. Items
            are opaque — any object ``evaluator`` understands. Pairs the
            dataset-shaped ``(candidate, example)`` signature with ``evaluator``.
        valset: Optional validation examples used for candidate selection.
        objective: Short goal statement (e.g. "Maximize sum of circle radii").
            Surfaced verbatim by every engine as the optimization goal.
        background: Long-form context — problem statement, evaluation rules,
            domain notes. Surfaced verbatim by every engine.
        test_set: Optional held-out test examples. When provided, the seed and
            optimized candidates are each scored on it outside the budget — the
            test set never enters the eval server, so engines and agents cannot
            see it. When omitted, test scoring is skipped entirely. Reported
            under ``result.metadata["test_score(s)"]`` (optimized) and
            ``result.metadata["baseline_test_score(s)"]`` (seed).
        config: Engine selection, run ``name``, budgets, output directory, and
            engine-specific options. Defaults to ``OptimizeAnythingConfig()``;
            at least one of ``config.max_evals`` or ``config.max_token_cost``
            must be set. When ``config.name`` is omitted, a name is generated
            from the engine, a short uuid, and a timestamp.

    Returns:
        A :class:`Result` with ``best_candidate``, ``best_score``,
        ``total_evals``, ``eval_log``, and ``metadata``.
    """
    if config is None:
        config = OptimizeAnythingConfig()
    if config.max_evals is None and config.max_token_cost is None:
        warnings.warn(
            "Neither config.max_evals nor config.max_token_cost is set; the run is unbounded "
            "and will stop only on the engine's own stop conditions.",
            stacklevel=2,
        )

    task = Task(
        name=config.name or _default_run_name(config.engine),
        seed_candidate=seed_candidate,
        objective=objective or "",
        background=background or "",
        train_set=dataset,
        val_set=valset,
        test_set=test_set,
    )
    engine = _build_engine(config)
    output_dir = _resolve_output_dir(config.output_dir, task=task, engine_name=engine.name)
    # The eval budget caps eval calls only. The proposer-cost cap
    # (config.max_token_cost) is read and enforced by the engine itself.
    budget = BudgetTracker(max_evals=config.max_evals)
    server = EvalServer(
        task,
        evaluator,
        budget,
        max_concurrency=config.max_concurrency,
        output_dir=output_dir,
    )
    return _run_engine(server, engine, owns_server=True)


def _run_engine(server: EvalServer, engine: Engine, *, owns_server: bool) -> Result:
    """Run ``engine`` and attach shared eval-server metadata to its result."""
    if owns_server:
        server.start()

    task = server.task

    # Held-out test is scored outside the budget, directly against the seed
    # candidate (baseline) and, after optimization, the best candidate. The
    # test_set is optional — when absent these passes are skipped entirely.
    baseline_test = _score_test(server, task, task.seed_candidate)

    start = time.time()
    try:
        result = engine.run(task, server)
    except BudgetExhausted:
        result = Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
        )
    finally:
        if owns_server:
            server.stop()

    result.total_evals = server.budget.used
    result.eval_log = server.eval_log
    result.metadata.setdefault("wall_time", time.time() - start)
    result.metadata["budget"] = server.budget.status()
    adapter_cost = float(result.metadata.get("adapter_cost", 0.0))
    result.metadata["total_cost"] = server.total_cost + adapter_cost
    result.metadata["progress_log"] = server.progress_log
    result.metadata["engine"] = engine.name
    if server.output_dir is not None:
        result.metadata["output_dir"] = str(server.output_dir)

    try:
        engine.process_result(result, server.output_dir)
    except Exception:
        # Engine artifact persistence is best-effort.
        pass

    # Reuse the baseline pass when the engine made no improvement, otherwise
    # score the optimized candidate. Both are optional and skipped without a
    # test_set.
    if result.best_candidate == task.seed_candidate:
        optimized_test = baseline_test
    else:
        optimized_test = _score_test(server, task, result.best_candidate)

    if baseline_test is not None:
        result.metadata["baseline_test_scores"], result.metadata["baseline_test_score"] = baseline_test
    if optimized_test is not None:
        result.metadata["test_scores"], result.metadata["test_score"] = optimized_test

    return result


def _score_test(server: EvalServer, task: Task, candidate: str | None) -> tuple[dict[str, float], float] | None:
    """Score ``candidate`` on the held-out ``test_set``, outside the budget.

    Returns ``(per_example_scores, average)`` or ``None`` when there is no
    test_set or no candidate to score. Test examples are evaluated directly
    via ``eval_fn`` — they never touch the eval server's budget or pool.
    """
    if not task.test_set or not candidate:
        return None
    scores: dict[str, float] = {}
    for i, ex in enumerate(task.test_set):
        eid = _resolve_id(ex, f"test_{i}")
        if eid in scores:
            eid = f"test_{i}"
        try:
            score, _ = server.eval_fn(candidate, ex)
            scores[eid] = score
        except Exception:
            scores[eid] = 0.0
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return scores, avg


def _build_engine(config: OptimizeAnythingConfig) -> Engine:
    """Instantiate ``config.engine`` or return it if it is already an engine."""
    if isinstance(config.engine, str):
        cls = get_engine_cls(config.engine)
        return cls(config)
    if config.engine_config:
        warnings.warn(
            "OptimizeAnythingConfig.engine_config is ignored when OptimizeAnythingConfig.engine is a constructed "
            "Engine instance.",
            stacklevel=3,
        )
    return config.engine


def _default_run_name(engine: str | Engine) -> str:
    """Generate a run name from the engine, a short uuid, and a timestamp."""
    backend = engine if isinstance(engine, str) else getattr(engine, "name", "engine")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{backend}-{uuid.uuid4().hex[:8]}-{ts}"


def _resolve_output_dir(output_dir: str | Path | None, *, task: Task, engine_name: str) -> Path:
    """Return an explicit output dir or create the standard timestamped one."""
    if output_dir is not None:
        path = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = Path("outputs") / "optimize_anything" / task.name / engine_name / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "AutoResearchEngine",
    "BestOfNEngine",
    "BudgetExhausted",
    "BudgetTracker",
    "ClaudeCodeAgentProposer",
    "Engine",
    "EvalServer",
    "GepaEngine",
    "MetaHarnessEngine",
    "OptimizeAnythingConfig",
    "Result",
    "Task",
    "get_engine_cls",
    "get_task",
    "list_engines",
    "list_tasks",
    "optimize_adaptive_sequential",
    "optimize_adaptive_sequential_with_server",
    "optimize_anything",
    "optimize_anything_with_server",
    "optimize_best_of",
    "optimize_best_of_with_server",
    "optimize_parallel",
    "optimize_parallel_with_server",
    "optimize_sequential",
    "optimize_sequential_with_server",
    "optimize_anything_from_task",
    "optimize_vote",
    "optimize_vote_with_server",
    "register_engine",
    "register_task",
    "register_task_factory",
]
