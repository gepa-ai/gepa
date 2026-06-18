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

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, cast

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

_MISSING = object()
_NEW_API_ARGS = {"task", "evaluate", "config"}
_LEGACY_API_ARGS = {"seed_candidate", "evaluator", "dataset", "valset", "objective", "background"}
_LEGACY_MIGRATION_MESSAGE = """\
gepa.optimize_anything.optimize_anything now uses the engine-based API:

    from gepa.optimize_anything import OptimizeAnythingConfig, Task, optimize_anything

    result = optimize_anything(
        task=Task(
            name="my_task",
            initial_candidate="initial prompt or program",
            objective="Improve the candidate's score.",
        ),
        evaluate=evaluate,
        config=OptimizeAnythingConfig(engine="gepa", max_evals=100),
    )

The previous seed_candidate/evaluator API has moved to:

    from gepa.legacy_optimize_anything import EngineConfig, GEPAConfig, optimize_anything

    result = optimize_anything(
        seed_candidate="initial prompt or program",
        evaluator=evaluate,
        objective="Improve the candidate's score.",
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=100)),
    )

This call was not run. Update the call to the new Task/evaluate API, or import
from gepa.legacy_optimize_anything explicitly to keep using the legacy API.
"""


def optimize_anything(
    *args: Any,
    **kwargs: Any,
) -> Result:
    """Validate the public call shape and run the engine-based API.

    The legacy ``seed_candidate=...`` / ``evaluator=...`` call shape is no
    longer accepted from this module. Import
    :mod:`gepa.legacy_optimize_anything` explicitly for that API.
    """
    legacy_args = sorted(_LEGACY_API_ARGS & set(kwargs))
    if legacy_args:
        _raise_optimize_anything_call_error(
            f"legacy argument(s) are no longer accepted here: {', '.join(legacy_args)}",
            deprecation=True,
        )

    unknown_args = sorted(set(kwargs) - _NEW_API_ARGS)
    if unknown_args:
        _raise_optimize_anything_call_error(f"unknown argument(s): {', '.join(unknown_args)}")

    if len(args) > 3:
        _raise_optimize_anything_call_error(
            f"expected at most 3 positional arguments (task, evaluate, config), got {len(args)}"
        )

    task = _pop_arg("task", 0, args, kwargs)
    evaluate = _pop_arg("evaluate", 1, args, kwargs)
    config = _pop_arg("config", 2, args, kwargs, default=None)

    if task is _MISSING:
        _raise_optimize_anything_call_error("missing required argument: task")
    if evaluate is _MISSING:
        _raise_optimize_anything_call_error("missing required argument: evaluate")
    if not isinstance(task, Task):
        _raise_optimize_anything_call_error(
            "task must be a gepa.optimize_anything.Task. "
            "If you passed a seed candidate string/dict, use gepa.legacy_optimize_anything instead.",
            deprecation=True,
        )
    if not callable(evaluate):
        _raise_optimize_anything_call_error("evaluate must be callable")
    if config is not None and not isinstance(config, OptimizeAnythingConfig):
        _raise_optimize_anything_call_error("config must be an OptimizeAnythingConfig or None")

    return _run_engine_optimization(task=task, evaluate=cast(EvalFn, evaluate), config=config)


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


def _pop_arg(
    name: str,
    position: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    default: Any = _MISSING,
) -> Any:
    has_positional = len(args) > position
    has_keyword = name in kwargs
    if has_positional and has_keyword:
        _raise_optimize_anything_call_error(f"{name!r} was provided both positionally and by keyword")
    if has_positional:
        return args[position]
    if has_keyword:
        return kwargs[name]
    return default


def _raise_optimize_anything_call_error(reason: str, *, deprecation: bool = False) -> NoReturn:
    message = f"Invalid optimize_anything call: {reason}.\n\n{_LEGACY_MIGRATION_MESSAGE}"
    print(message, file=sys.stderr)
    if deprecation:
        warnings.warn(message, DeprecationWarning, stacklevel=3)
    raise TypeError(message)


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
