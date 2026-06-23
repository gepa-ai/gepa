"""Public ``optimize_anything`` API.

A call has three parts:

- **What to optimize** — the task fields passed directly to
  :func:`optimize_anything` (``name``, ``initial_candidate``, ``objective``,
  ``background``, and optional ``train_set`` / ``val_set`` / ``test_set``).
- **How to score it** — ``evaluate``, a function returning ``(score, info)``.
- **How to optimize** — :class:`OptimizeAnythingConfig`, which chooses the
  :class:`Engine` and sets the budget.

Example:

    from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything

    def evaluate(candidate: str) -> tuple[float, dict]:
        return run_candidate(candidate), {"diagnostics": "..."}

    result = optimize_anything(
        name="my_task",
        initial_candidate="initial prompt or program",
        objective="Improve the candidate's score.",
        evaluate=evaluate,
        config=OptimizeAnythingConfig(engine="gepa", max_evals=100),
    )

Use :func:`optimize_anything_with_server` when an outer system already owns an
:class:`EvalServer`.
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from gepa.oa.budget import BudgetExhausted, BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engine import Engine, Result
from gepa.oa.engines import AutoResearchEngine, BestOfNEngine, GepaEngine, MetaHarnessEngine
from gepa.oa.ensemble import (
    optimize_adaptive_sequential_with_server,
    optimize_best_of,
    optimize_best_of_with_server,
    optimize_parallel,
    optimize_parallel_with_server,
    optimize_sequential,
    optimize_sequential_with_server,
    optimize_vote,
    optimize_vote_with_server,
)
from gepa.oa.eval_server import EvalServer
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
from gepa.oa.task import EvalFn, Task


def optimize_anything(
    *,
    name: str,
    initial_candidate: str,
    evaluate: EvalFn,
    objective: str = "",
    background: str = "",
    train_set: list[Any] | None = None,
    val_set: list[Any] | None = None,
    test_set: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Optimize a text candidate (prompt, code, instructions, ...) against a score.

    Args:
        name: Unique identifier for this task (e.g. ``"circle_packing"``). Used
            for logging and to lay out the default output directory.
        initial_candidate: The seed text to evolve from.
        evaluate: Scoring function returning ``(score, info)``. Use
            ``(candidate) -> (score, info)`` for a single task, or
            ``(candidate, example) -> (score, info)`` when train/val/test
            examples are provided. ``score`` is a float (higher is better) and
            ``info`` is a free-form dict surfaced to the engine as feedback.
        objective: Short goal statement (e.g. "Maximize sum of circle radii").
            Surfaced verbatim by every engine as the optimization goal.
        background: Long-form context — problem statement, evaluation rules,
            domain notes. Surfaced verbatim by every engine.
        train_set: Training examples used during optimization. Items are opaque
            — any object ``evaluate`` understands. Pairs the dataset-shaped
            ``(candidate, example)`` signature with ``evaluate``.
        val_set: Validation examples used for candidate selection.
        test_set: Held-out test examples scored after the run, outside the
            budget. Reported under ``result.metadata["test_score(s)"]``.
        metadata: Free-form per-task context (run mode, candidate type, etc.).
            The whole dict is handed to the engine, so engines may read any key
            from ``task.metadata``. Only JSON-scalar values (``str``/``int``/
            ``float``/``bool``) are surfaced to proposer agents via the eval
            server's ``GET /task`` endpoint; lists, dicts, and arbitrary objects
            are *not* exposed there (they stay engine-side only). A few keys are
            reserved and read by specific engines: ``"val_set"`` is used by the
            GEPA engine as a validation-set fallback, and
            ``"optimize_anything_handoffs"`` is consumed by the AutoResearch
            engine.
        config: Engine selection, budgets, output directory, and
            engine-specific options. Defaults to ``OptimizeAnythingConfig()``;
            at least one of ``config.max_evals`` or ``config.max_token_cost``
            must be set.

    Returns:
        A :class:`Result` with ``best_candidate``, ``best_score``,
        ``total_evals``, ``eval_log``, and ``metadata``.
    """
    task = Task(
        name=name,
        initial_candidate=initial_candidate,
        objective=objective,
        background=background,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        metadata=metadata if metadata is not None else {},
    )
    return optimize_task(task, evaluate, config)


def optimize_task(
    task: Task,
    evaluate: EvalFn,
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Optimize a prebuilt :class:`Task` with the configured engine.

    The Task-based counterpart of :func:`optimize_anything`: that function
    builds a :class:`Task` from its arguments and forwards it here. Call this
    directly when you already hold a ``Task`` — e.g. the ensemble helpers,
    which carry one ``Task`` forward across stages. See ``evaluate`` /
    ``config`` and the return value in :func:`optimize_anything`.
    """
    if config is None:
        config = OptimizeAnythingConfig()
    if config.max_evals is None and config.max_token_cost is None:
        raise ValueError("At least one of config.max_evals or config.max_token_cost must be specified")

    engine = _build_engine(config)
    output_dir = _resolve_output_dir(config.output_dir, task=task, engine_name=engine.name)
    # The eval budget caps eval calls only. The proposer-cost cap
    # (config.max_token_cost) is read and enforced by the engine itself.
    budget = BudgetTracker(max_evals=config.max_evals)
    server = EvalServer(
        task,
        evaluate,
        budget,
        tracker=config.tracker,
        max_concurrency=config.max_concurrency,
        output_dir=output_dir,
    )
    return _run_engine(server, engine, owns_server=True)


def optimize_anything_with_server(
    server: EvalServer,
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Run optimization against a caller-owned :class:`EvalServer`.

    The caller is responsible for ``server.start()`` and ``server.stop()``.
    Server-shaped config fields such as ``max_evals``, ``output_dir``, and
    ``tracker`` are ignored because they already belong to the server.
    """
    if config is None:
        config = OptimizeAnythingConfig()

    engine = _build_engine(config)
    return _run_engine(server, engine, owns_server=False)


def _run_engine(server: EvalServer, engine: Engine, *, owns_server: bool) -> Result:
    """Run ``engine`` and attach shared eval-server metadata to its result."""
    if owns_server:
        server.start()

    task = server.task
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

    if task.test_set and result.best_candidate:
        test_scores: dict[str, float] = {}
        for eid, ex in server.iter_split("test"):
            try:
                score, _ = server.eval_fn(result.best_candidate, ex)
                test_scores[eid] = score
            except Exception:
                test_scores[eid] = 0.0
        result.metadata["test_scores"] = test_scores
        result.metadata["test_score"] = sum(test_scores.values()) / len(test_scores) if test_scores else 0.0

    return result


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
    "EvalFn",
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
    "optimize_adaptive_sequential_with_server",
    "optimize_anything",
    "optimize_anything_with_server",
    "optimize_best_of",
    "optimize_best_of_with_server",
    "optimize_parallel",
    "optimize_parallel_with_server",
    "optimize_sequential",
    "optimize_sequential_with_server",
    "optimize_task",
    "optimize_vote",
    "optimize_vote_with_server",
    "register_engine",
    "register_task",
    "register_task_factory",
]
