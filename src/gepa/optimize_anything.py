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

import dataclasses
import time
import uuid
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

# Legacy launcher surface, re-exported so v0.1.x-era code that did
# ``from gepa.optimize_anything import GEPAConfig, ...`` keeps working against
# this module. New code should prefer :class:`OptimizeAnythingConfig`.
from gepa.gepa_launcher import (
    Candidate,
    EngineConfig,
    Evaluator,
    EvaluatorWrapper,
    GEPAConfig,
    GEPAResult,
    Image,
    LogContext,
    MergeConfig,
    OptimizationState,
    RefinerConfig,
    ReflectionConfig,
    SideInfo,
    TrackingConfig,
    get_log_context,
    log,
    make_litellm_lm,
    set_log_context,
)
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
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Callable[..., Any] | None = None,
    batch_evaluator: Callable[..., Any] | None = None,
    dataset: list[Any] | None = None,
    valset: list[Any] | None = None,
    objective: str | None = None,
    background: str | None = None,
    test_set: list[Any] | None = None,
    config: OptimizeAnythingConfig | None = None,
) -> Result:
    """Optimize a text candidate (prompt, code, instructions, ...) against a score.

    The signature mirrors :func:`gepa.gepa_launcher.optimize_anything`
    (``seed_candidate`` / ``evaluator`` / ``batch_evaluator`` / ``dataset`` /
    ``valset`` / ``objective`` / ``background``) so the two entry points share
    one shape; the new API swaps ``config`` for an
    :class:`OptimizeAnythingConfig` and adds ``test_set``.

    Args:
        seed_candidate: The seed to evolve from — either a single text string
            or, for multi-component optimization, a ``{component: text}`` dict
            (:data:`Candidate`) whose components are co-optimized. Dict seeds
            are supported by the ``gepa`` engine; the other engines treat the
            seed as a single text. ``None`` for seedless mode, where the engine
            bootstraps the first candidate from ``objective`` / ``background``.
        evaluator: Scoring function returning ``(score, info)`` — a bare
            ``score`` is also accepted, with ``info`` defaulting to ``{}``. Use
            ``(candidate) -> (score, info)`` for a single task, or
            ``(candidate, example) -> (score, info)`` when dataset/val/test
            examples are provided. ``score`` is a float (higher is better) and
            ``info`` is a free-form dict surfaced to the engine as feedback.
            Optional when ``batch_evaluator`` is provided (at least one is
            required).
        batch_evaluator: Grouped scoring hook (launcher parity): receives ALL
            ``(candidate, example)`` pairs of an evaluation stage in ONE call —
            e.g. to submit a provider batch job or fan out over your own
            infrastructure — and returns one result per pair (``score`` or
            ``(score, info)``). Multi-pair evaluations (minibatch, valset, and
            held-out test passes) prefer it; single-pair evaluations use
            ``evaluator``, or route through the batch function as singleton
            batches when ``evaluator`` is omitted. If its signature accepts an
            ``opt_states`` keyword, the gepa engine forwards the aligned
            per-pair optimization states.
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
            from the engine, a short uuid, and a timestamp. A legacy
            :class:`GEPAConfig` is also accepted and converted; the run uses
            the gepa engine and returns the launcher's
            :class:`~gepa.core.result.GEPAResult`.

    Returns:
        A :class:`Result` with ``best_candidate``, ``best_score``,
        ``total_evals``, ``eval_log``, and ``metadata`` — or, when ``config``
        is a legacy :class:`GEPAConfig`, the underlying
        :class:`~gepa.core.result.GEPAResult` (``candidates``, ``best_idx``,
        ``val_aggregate_scores``, ...) so v0.1.x callers get the result shape
        they were written against.
    """
    legacy_config = False
    if config is None:
        config = OptimizeAnythingConfig()
    elif not isinstance(config, OptimizeAnythingConfig):
        if test_set is not None:
            raise ValueError(
                "test_set requires an OptimizeAnythingConfig; the legacy GEPAConfig API has no held-out test pass."
            )
        config = _from_legacy_config(config)
        legacy_config = True
    if evaluator is None and batch_evaluator is None:
        raise ValueError("Provide evaluator=, batch_evaluator=, or both.")
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
        batch_evaluate=batch_evaluator,
        max_concurrency=config.max_concurrency,
        output_dir=output_dir,
    )
    result = _run_engine(server, engine, owns_server=True)
    if legacy_config:
        # v0.1.x callers always get the launcher's result type back — the gepa
        # engine stashes the underlying GEPAResult in the result metadata. If
        # the eval budget died before GEPA core saved any state (e.g. budget
        # smaller than the valset), synthesize a single-candidate snapshot.
        # The public signature advertises only the new API; legacy
        # GEPAConfig-in/GEPAResult-out is a runtime affordance, so the legacy
        # result rides out through an Any-typed local.
        legacy_result: Any = result.metadata.get("gepa_result") or _legacy_result_from(result, seed_candidate)
        return legacy_result
    return result


def _from_legacy_config(config: Any) -> OptimizeAnythingConfig:
    """Convert a legacy launcher :class:`GEPAConfig` into the OA-shaped config.

    The whole config rides through ``engine_config`` field-for-field (so nested
    configs and callbacks survive as objects); the launcher's eval budget
    becomes ``max_evals`` — ``None`` stays unbounded, as in the launcher.
    """
    from gepa.gepa_launcher import GEPAConfig

    if not isinstance(config, GEPAConfig):
        raise TypeError(f"config must be an OptimizeAnythingConfig or GEPAConfig, got {type(config).__name__}")
    return OptimizeAnythingConfig(
        engine="gepa",
        max_evals=config.engine.max_metric_calls,
        engine_config={f.name: getattr(config, f.name) for f in dataclasses.fields(config)},
    )


def _legacy_result_from(result: Result, seed_candidate: Any) -> GEPAResult:
    """Fallback :class:`GEPAResult` when the gepa engine had none to report.

    Happens when the eval budget is exhausted before GEPA core saved any
    state (e.g. the budget is smaller than the seed's valset pass). Returns a
    single-candidate snapshot of the best the eval server saw, so legacy
    callers still get the result shape they were written against.
    """
    best = result.best_candidate or seed_candidate
    if isinstance(best, dict):
        candidate, key = best, None
    else:
        candidate, key = {"current_candidate": best or ""}, "current_candidate"
    score = float(result.best_score) if result.best_score is not None else float("-inf")
    return GEPAResult(
        candidates=[candidate],
        parents=[[None]],
        val_aggregate_scores=[score],
        val_subscores=[{}],
        per_val_instance_best_candidates={},
        discovery_eval_counts=[result.total_evals or 0],
        total_metric_calls=result.total_evals,
        _str_candidate_key=key,
    )


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
    if server.batch_fn is not None:
        # Grouped test pass: the user's batch function gets all pairs in one call.
        try:
            per_example = [score for score, _info in server.batch_fn([(candidate, ex) for ex in task.test_set])]
        except Exception:
            per_example = [0.0] * len(task.test_set)
    else:
        per_example = []
        for ex in task.test_set:
            try:
                per_example.append(server.eval_fn(candidate, ex)[0])
            except Exception:
                per_example.append(0.0)
    scores: dict[str, float] = {}
    for i, (ex, score) in enumerate(zip(task.test_set, per_example, strict=True)):
        eid = _resolve_id(ex, f"test_{i}")
        if eid in scores:
            eid = f"test_{i}"
        scores[eid] = score
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
    # Legacy launcher surface (v0.1.x `gepa.optimize_anything` API), re-exported
    # for backward compatibility.
    "Candidate",
    "ClaudeCodeAgentProposer",
    "Engine",
    "EngineConfig",
    "EvalServer",
    "Evaluator",
    "EvaluatorWrapper",
    "GEPAConfig",
    "GEPAResult",
    "GepaEngine",
    "Image",
    "LogContext",
    "MergeConfig",
    "MetaHarnessEngine",
    "OptimizationState",
    "OptimizeAnythingConfig",
    "RefinerConfig",
    "ReflectionConfig",
    "Result",
    "SideInfo",
    "Task",
    "TrackingConfig",
    "get_engine_cls",
    "get_log_context",
    "get_task",
    "list_engines",
    "list_tasks",
    "log",
    "make_litellm_lm",
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
    "set_log_context",
]


def __getattr__(name: str) -> Any:
    """Permanent compatibility shim for the v0.1.x ``gepa.optimize_anything`` API.

    The commonly-imported launcher names are re-exported explicitly at the top
    of this module. This fallback catches any *other* public symbol a pinned
    v0.1.x caller may still import (e.g. ``EvaluatorWrapper``) and forwards it
    from :mod:`gepa.gepa_launcher`, so old imports keep working unmodified
    instead of hitting a bare ``ImportError``. Truly-unknown names raise
    ``AttributeError`` with a pointer to where the launcher surface now lives.
    """
    if not name.startswith("_"):
        import gepa.gepa_launcher as _launcher

        obj = getattr(_launcher, name, None)
        if obj is not None:
            return obj
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        "The legacy launcher surface now lives in 'gepa.gepa_launcher'."
    )
