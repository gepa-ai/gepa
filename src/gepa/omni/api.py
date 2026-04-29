"""``gepa.omni.optimize_anything`` — the unified backend-pluggable entry point.

A single optimization call dispatches to any registered backend (GEPA,
Claude Code, Meta-Harness, …) without changing the user-facing surface.

Example::

    from gepa.omni import optimize_anything, Task

    def evaluate(candidate: str) -> tuple[float, dict]:
        return run(candidate), {}

    result = optimize_anything(
        task=Task(
            name="my_task",
            initial_candidate="...",
            objective="Maximize foo.",
        ),
        evaluate=evaluate,
        backend="gepa",
        config={"reflection": {"reflection_lm": "openai/gpt-5"}},
        max_evals=200,
        max_token_cost=5.0,
    )
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gepa.omni.backend import Backend, Result
from gepa.omni.budget import BudgetExhausted, BudgetTracker
from gepa.omni.eval_server import EvalServer
from gepa.omni.registry import make_backend
from gepa.omni.task import EvalFn, Task


def optimize_anything(
    task: Task,
    evaluate: EvalFn,
    backend: str | Backend = "gepa",
    config: dict[str, Any] | None = None,
    *,
    max_evals: int | None = 100,
    max_token_cost: float | None = None,
    max_concurrency: int = 8,
    output_dir: str | Path | None = None,
    tracker: Any | None = None,
    stop_at_score: float | None = None,
    effort: str | None = None,
    max_thinking_tokens: int | None = None,
    sandbox: bool | None = None,
) -> Result:
    """Run optimization for a task using the chosen backend.

    Args:
        task: What to optimize. ``task.objective`` and ``task.background`` are
            surfaced verbatim by every backend.
        evaluate: How to score a candidate. ``(candidate) -> (score, info)``
            for single-task; ``(candidate, example) -> (score, info)`` for
            dataset tasks. Required.
        backend: Name of a registered backend (``"gepa"`` / ``"claude_code"``
            / ``"meta_harness"``) or a constructed :class:`Backend` instance.
        config: Backend-specific configuration. Keys depend on the backend
            (e.g. ``{"engine": {...}, "reflection": {...}}`` for GEPA, or
            ``{"model": "opus", "max_iterations": 10}`` for meta_harness).
            Ignored when ``backend`` is already a constructed instance.
        max_evals: Server-side cap on eval calls.
        max_token_cost: Cap on cumulative LLM USD spend. Threaded into
            whichever per-backend knob enforces it (GEPA's
            ``max_reflection_cost``, Claude Code's ``--max-budget-usd``, etc.).
        max_concurrency: Eval server thread-pool size.
        output_dir: When set, the eval server persists per-eval JSON,
            ``progress_log.jsonl``, and ``summary.json`` here, and the chosen
            backend writes its artifacts under ``<output_dir>/<backend.name>/``.
        tracker: Optional experiment tracker (wandb/mlflow wrapper).
        stop_at_score, effort, max_thinking_tokens, sandbox: Top-level cross-
            backend conveniences. Threaded into matching backend attributes
            when the backend hasn't already been configured for them.

    Returns:
        :class:`Result` with ``best_candidate``, ``best_score``, ``total_evals``,
        ``eval_log``, and a metadata dict (``total_cost``, ``budget``,
        ``progress_log``, plus backend-specific keys).
    """
    if max_evals is None and max_token_cost is None:
        raise ValueError("At least one of max_evals or max_token_cost must be specified")

    backend_obj = make_backend(backend, config)

    out_path: Path | None = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        if hasattr(backend_obj, "run_dir") and not getattr(backend_obj, "run_dir", None):
            backend_obj.run_dir = str(out_path / backend_obj.name)

    _thread_top_level(
        backend_obj,
        stop_at_score=stop_at_score,
        effort=effort,
        max_thinking_tokens=max_thinking_tokens,
        sandbox=sandbox,
    )

    budget = BudgetTracker(max_evals=max_evals, max_token_cost=max_token_cost)
    server = EvalServer(
        task,
        evaluate,
        budget,
        tracker=tracker,
        max_concurrency=max_concurrency,
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

    if out_path is not None:
        try:
            backend_obj.process_result(result, out_path)
        except Exception:
            # Backend artifact persistence is best-effort; never let it kill
            # an otherwise-successful run.
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


def _thread_top_level(
    backend_obj: Backend,
    *,
    stop_at_score: float | None,
    effort: str | None,
    max_thinking_tokens: int | None,
    sandbox: bool | None,
) -> None:
    """Set top-level shared knobs on the backend when it has matching slots
    that haven't already been populated. User overrides on the backend itself
    always win."""
    forwards = {
        "stop_at_score": stop_at_score,
        "effort": effort,
        "max_thinking_tokens": max_thinking_tokens,
        "sandbox": sandbox,
    }
    for attr, value in forwards.items():
        if value is None:
            continue
        if not hasattr(backend_obj, attr):
            continue
        if getattr(backend_obj, attr) is not None:
            continue
        setattr(backend_obj, attr, value)
