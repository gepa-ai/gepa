"""``gepa.omni.optimize_anything`` — the unified backend-pluggable entry point.

A single optimization call dispatches to any registered backend (GEPA,
Claude Code, Meta-Harness, …) over a shared eval-server contract.

Two entry points:

- :func:`optimize_anything` — convenience. Build a server from
  ``(task, evaluate, config)``, run it, return the result.
- :func:`optimize_anything_with_server` — primitive. Caller owns the
  ``EvalServer`` (construction + ``start()`` + ``stop()``). Useful when an
  outer framework (e.g. terrarium) already manages the eval server's
  budget, output dir, and tracker, and just wants omni's backend
  dispatching layered on top.

Example::

    from gepa.omni import OmniConfig, Task, optimize_anything

    def evaluate(candidate: str) -> tuple[float, dict]:
        return run(candidate), {}

    result = optimize_anything(
        task=Task(name="my_task", initial_candidate="...", objective="Maximize foo."),
        evaluate=evaluate,
        config=OmniConfig(
            backend="gepa",
            max_evals=200,
            max_token_cost=5.0,
            config={"reflection": {"reflection_lm": "openai/gpt-5"}},
        ),
    )
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path

from gepa.omni.backend import Backend, Result
from gepa.omni.budget import BudgetExhausted, BudgetTracker
from gepa.omni.config import OmniConfig
from gepa.omni.eval_server import EvalServer
from gepa.omni.registry import get_backend_cls
from gepa.omni.task import EvalFn, Task


def optimize_anything(
    task: Task,
    evaluate: EvalFn,
    config: OmniConfig | None = None,
) -> Result:
    """Run optimization for a task using the configured backend.

    Args:
        task: What to optimize. ``task.objective`` and ``task.background`` are
            surfaced verbatim by every backend.
        evaluate: How to score a candidate. ``(candidate) -> (score, info)``
            for single-task; ``(candidate, example) -> (score, info)`` for
            dataset tasks. Required.
        config: All other knobs. Defaults to ``OmniConfig()``.

    Returns:
        :class:`Result` with ``best_candidate``, ``best_score``, ``total_evals``,
        ``eval_log``, and a metadata dict (``total_cost``, ``budget``,
        ``progress_log``, ``output_dir``, plus backend-specific keys).
    """
    if config is None:
        config = OmniConfig()
    if config.max_evals is None and config.max_token_cost is None:
        raise ValueError("At least one of config.max_evals or config.max_token_cost must be specified")
    if config.effort is not None and config.max_thinking_tokens is not None:
        raise ValueError("config.effort and config.max_thinking_tokens are mutually exclusive")

    backend = _build_backend(config)
    out_path = _resolve_output_dir(config.output_dir, task=task, backend_name=backend.name)
    budget = BudgetTracker(max_evals=config.max_evals, max_token_cost=config.max_token_cost)
    server = EvalServer(
        task,
        evaluate,
        budget,
        tracker=config.tracker,
        max_concurrency=config.max_concurrency,
        output_dir=out_path,
    )
    return _execute(server, backend, owns_server=True)


def optimize_anything_with_server(
    server: EvalServer,
    config: OmniConfig | None = None,
) -> Result:
    """Run optimization against a caller-owned :class:`EvalServer`.

    The primitive form of :func:`optimize_anything`. The server already
    encapsulates task, eval function, budget, output dir, concurrency, and
    tracker — so the server-shaped fields on ``OmniConfig`` (``max_evals``,
    ``max_token_cost``, ``output_dir``, ``max_concurrency``, ``tracker``)
    are **ignored** here; only ``backend`` and the cross-backend conveniences
    (``run_dir``, ``effort``, ``max_thinking_tokens``, ``sandbox``,
    ``stop_at_score``, ``config``) are read.

    The caller owns lifecycle: construct the server, call ``server.start()``
    before this function, and ``server.stop()`` after (or use a
    context-manager wrapper). The function does not start or stop the
    server — re-entry on the same server between calls is expected.

    Args:
        server: A constructed and started :class:`EvalServer`.
        config: Backend selector and backend-specific config. Defaults to
            ``OmniConfig()`` (= ``backend="gepa"``).

    Returns:
        :class:`Result` shaped exactly like :func:`optimize_anything`.

    Example::

        server = EvalServer(task, evaluate, BudgetTracker(max_evals=200))
        server.start()
        try:
            result = optimize_anything_with_server(
                server, OmniConfig(backend="gepa"),
            )
        finally:
            server.stop()
    """
    if config is None:
        config = OmniConfig()
    if config.effort is not None and config.max_thinking_tokens is not None:
        raise ValueError("config.effort and config.max_thinking_tokens are mutually exclusive")

    backend = _build_backend(config)
    return _execute(server, backend, owns_server=False)


def _execute(server: EvalServer, backend: Backend, *, owns_server: bool) -> Result:
    """Run a backend against a server and fill the result metadata.

    Shared core of :func:`optimize_anything` and
    :func:`optimize_anything_with_server`. Assumes config validation and
    server/backend construction already happened.

    When ``owns_server=True``, this function calls ``server.start()`` /
    ``server.stop()``. Otherwise it leaves lifecycle to the caller.
    """
    if owns_server:
        server.start()

    task = server.task
    start = time.time()
    try:
        result = backend.run(task, server)
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
    result.metadata["backend"] = backend.name
    if server.output_dir is not None:
        result.metadata["output_dir"] = str(server.output_dir)

    try:
        backend.process_result(result, server.output_dir)
    except Exception:
        # Backend artifact persistence is best-effort; never let it kill an
        # otherwise-successful run.
        pass

    if task.test_set and result.best_candidate:
        eval_fn = server.eval_fn
        test_scores: dict[str, float] = {}
        for eid, ex in server.iter_split("test"):
            try:
                score, _ = eval_fn(result.best_candidate, ex)
                test_scores[eid] = score
            except Exception:
                test_scores[eid] = 0.0
        result.metadata["test_scores"] = test_scores
        result.metadata["test_score"] = sum(test_scores.values()) / len(test_scores) if test_scores else 0.0

    return result


def _build_backend(config: OmniConfig) -> Backend:
    """Resolve ``config.backend`` into an instance.

    - String: look up the registered class and call ``cls(config)``. The
      backend's ``__init__`` consumes the keys it understands and warns
      about unknown keys in ``config.config``.
    - Backend instance: return as-is. ``config.config`` is ignored (the
      caller already configured the backend); we warn so it's not silent.
    """
    if isinstance(config.backend, str):
        cls = get_backend_cls(config.backend)
        return cls(config)
    if config.config:
        warnings.warn(
            "OmniConfig.config is ignored when OmniConfig.backend is a constructed Backend instance.",
            stacklevel=3,
        )
    return config.backend


def _resolve_output_dir(output_dir: str | Path | None, *, task: Task, backend_name: str) -> Path:
    """Return ``output_dir`` if explicit, else a fresh timestamped default.

    Default: ``outputs/omni/<task>/<backend>/<YYYY-MM-DD_HH-MM-SS>/``.
    """
    if output_dir is not None:
        path = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = Path("outputs") / "omni" / task.name / backend_name / ts
    path.mkdir(parents=True, exist_ok=True)
    return path
