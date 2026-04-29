"""``gepa.omni.optimize_anything`` — the unified backend-pluggable entry point.

A single optimization call dispatches to any registered backend (GEPA,
Claude Code, Meta-Harness, …) over a shared eval-server contract.

Example::

    from gepa.omni import OmniConfig, Task, optimize_anything

    def evaluate(candidate: str) -> tuple[float, dict]:
        return run(candidate), {}

    result = optimize_anything(
        task=Task(
            name="my_task",
            initial_candidate="...",
            objective="Maximize foo.",
        ),
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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

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

    backend_obj = _build_backend(config)

    out_path = _resolve_output_dir(config.output_dir, task=task, backend_name=backend_obj.name)

    budget = BudgetTracker(max_evals=config.max_evals, max_token_cost=config.max_token_cost)
    server = EvalServer(
        task,
        evaluate,
        budget,
        tracker=config.tracker,
        max_concurrency=config.max_concurrency,
        output_dir=out_path,
    )
    server.start()

    start = time.time()
    try:
        result = backend_obj.run(task, server)
    except BudgetExhausted:
        result = Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=budget.used,
            eval_log=server.eval_log,
        )
    finally:
        server.stop()

    result.total_evals = budget.used
    result.eval_log = server.eval_log
    result.metadata.setdefault("wall_time", time.time() - start)
    result.metadata["budget"] = budget.status()
    adapter_cost = float(result.metadata.get("adapter_cost", 0.0))
    result.metadata["total_cost"] = server.total_cost + adapter_cost
    result.metadata["progress_log"] = server.progress_log
    result.metadata["backend"] = backend_obj.name
    result.metadata["output_dir"] = str(out_path)

    try:
        backend_obj.process_result(result, out_path)
    except Exception:
        # Backend artifact persistence is best-effort; never let it kill an
        # otherwise-successful run.
        pass

    if task.test_set and result.best_candidate:
        eval_fn: Callable[..., tuple[float, dict[str, Any]]] = server.eval_fn
        test_scores: dict[str, float] = {}
        for ex in task.test_set:
            try:
                score, _ = eval_fn(result.best_candidate, ex)
                test_scores[ex.id] = score
            except Exception:
                test_scores[ex.id] = 0.0
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
