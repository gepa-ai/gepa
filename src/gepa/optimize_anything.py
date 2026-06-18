"""Public ``optimize_anything`` API.

The API has three pieces:

- :class:`Task` describes what to optimize.
- ``evaluate`` scores a candidate and returns ``(score, info)``.
- :class:`OptimizeAnythingConfig` chooses the optimization :class:`Engine`
  and sets budgets.

Example:

    from gepa.optimize_anything import OptimizeAnythingConfig, Task, optimize_anything

    def evaluate(candidate: str) -> tuple[float, dict]:
        return run_candidate(candidate), {"diagnostics": "..."}

    result = optimize_anything(
        task=Task(
            name="my_task",
            initial_candidate="initial prompt or program",
            objective="Improve the candidate's score.",
        ),
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
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from gepa.core.result import GEPAResult

_NEW_API_ARGS = {"task", "evaluate", "config"}
_LEGACY_API_ARGS = {"seed_candidate", "evaluator", "dataset", "valset", "objective", "background"}

_LEGACY_FALLBACK_MESSAGE = """\
gepa.optimize_anything.optimize_anything received a legacy seed_candidate/evaluator
call and is transparently dispatching to gepa.legacy_optimize_anything so existing
code keeps working. This backward-compatible fallback is deprecated and may be
removed in a future release.

Migrate to the engine-based API:

    from gepa.optimize_anything import OptimizeAnythingConfig, Task, optimize_anything

    result = optimize_anything(
        task=Task(name="my_task", initial_candidate="...", objective="..."),
        evaluate=evaluate,
        config=OptimizeAnythingConfig(engine="gepa", max_evals=100),
    )

or import the legacy API explicitly to silence this warning:

    from gepa.legacy_optimize_anything import optimize_anything
"""


def optimize_anything(
    *args: Any,
    **kwargs: Any,
) -> Result | GEPAResult:
    """Run the engine-based API, falling back to the legacy API when needed.

    Calls using the new ``task`` / ``evaluate`` / ``config`` shape run the
    engine-based optimizer. Calls using the legacy ``seed_candidate=...`` /
    ``evaluator=...`` shape are transparently dispatched to
    :func:`gepa.legacy_optimize_anything.optimize_anything` with a
    :class:`DeprecationWarning`, so existing code keeps working. Import
    :mod:`gepa.legacy_optimize_anything` directly to use the legacy API without
    the warning. Calls that mix the two APIs are rejected with a ``TypeError``.
    """
    legacy_args = sorted(_LEGACY_API_ARGS & set(kwargs))
    if legacy_args:
        new_args = sorted((_NEW_API_ARGS - {"config"}) & set(kwargs))
        if new_args:
            raise TypeError(
                f"cannot mix new-API argument(s) {new_args} with legacy argument(s) {legacy_args}; "
                "use either the Task/evaluate API or gepa.legacy_optimize_anything, not both"
            )
        warnings.warn(_LEGACY_FALLBACK_MESSAGE, DeprecationWarning, stacklevel=2)
        from gepa.legacy_optimize_anything import optimize_anything as _legacy_optimize_anything

        return _legacy_optimize_anything(*args, **kwargs)

    return _run_engine_optimization(*args, **kwargs)


def _run_engine_optimization(
    task: Task,
    evaluate: EvalFn,
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Run optimization for ``task`` using the configured engine.

    Args:
        task: What to optimize. ``task.objective`` and ``task.background`` are
            passed to the engine.
        evaluate: Scoring function. Use ``(candidate) -> (score, info)`` for a
            single task, or ``(candidate, example) -> (score, info)`` when the
            task has train/val/test examples.
        config: Engine selection, budgets, output directory, and
            engine-specific options. Defaults to ``OptimizeAnythingConfig()``.

    Returns:
        A :class:`Result` with ``best_candidate``, ``best_score``,
        ``total_evals``, ``eval_log``, and metadata.
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
    "optimize_vote",
    "optimize_vote_with_server",
    "register_engine",
    "register_task",
    "register_task_factory",
]
