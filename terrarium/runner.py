"""Main entry point: load a task + adapter, run evolution, report results."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

from terrarium.adapter import Adapter, Result
from terrarium.budget import BudgetExhausted, BudgetTracker
from terrarium.eval_server import EvalServer
from terrarium.registry import get_task
from terrarium.task import Task
from terrarium.tracking import TerrariumTracker, TrackingConfig


def run(
    task: str | Task,
    adapter: str | Adapter,
    *,
    max_evals: int = 100,
    max_concurrency: int = 8,
    tracking: TrackingConfig | None = None,
) -> Result:
    """Run an evolution system on a task.

    This is the main terrarium entry point. It:
    1. Loads the task and adapter (from name/path or object).
    2. Creates an EvalServer (the single budget choke point).
    3. Starts the HTTP endpoint (for external adapters).
    4. Calls adapter.evolve(task, server, max_evals).

    All adapters — in-process or external — go through the same server.

    Args:
        task: Task name (e.g. "circle_packing") or a Task object.
        adapter: Path to an adapter Python file, or an Adapter object.
            The file must define a ``create_adapter()`` function.
        max_evals: Maximum number of eval calls allowed.

    Returns:
        Result from the evolution run.

    Example::

        result = terrarium.run("circle_packing", "my_adapter.py", max_evals=100)
        print(result.best_score)
    """
    if isinstance(task, str):
        task = get_task(task)

    if isinstance(adapter, str):
        adapter = load_adapter(adapter)

    tracker = TerrariumTracker(tracking) if tracking else None
    budget = BudgetTracker(max_evals=max_evals)
    server = EvalServer(task, budget, tracker=tracker, max_concurrency=max_concurrency)
    server.start()

    if tracker:
        tracker.start({"task": task.name, "max_evals": max_evals})
        # Inject GEPA-level callback for iteration/valset metrics
        from terrarium.adapters.gepa import GEPAAdapter

        if isinstance(adapter, GEPAAdapter):
            adapter.callbacks.append(tracker.create_callback())

    start = time.time()

    try:
        result = adapter.evolve(task, server, max_evals)
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
    result.metadata["wall_time"] = time.time() - start
    result.metadata["budget"] = budget.status()

    if tracker:
        tracker.log_summary({
            "best_score": result.best_score,
            "total_evals": result.total_evals,
            "wall_time": result.metadata["wall_time"],
        })
        tracker.end()

    return result


def load_adapter(path: str) -> Adapter:
    """Load an Adapter from a Python file.

    The file must define ``create_adapter() -> Adapter``.
    """
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("_terrarium_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_terrarium_adapter"] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "create_adapter"):
        raise AttributeError(f"Adapter file {path} must define create_adapter() -> Adapter")

    return mod.create_adapter()


def main() -> None:
    """CLI: ``python -m terrarium <task_name> <adapter_path> [--max-evals N]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="terrarium",
        description="Run an evolution system on a Terrarium task.",
    )
    parser.add_argument("task", help="Task name (e.g. circle_packing, aime_math)")
    parser.add_argument("adapter", help="Path to adapter .py file (must define create_adapter())")
    parser.add_argument("--max-evals", type=int, default=100, help="Maximum eval budget (default: 100)")
    parser.add_argument("--output", "-o", help="Write result JSON to this file")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--wandb-project", default="terrarium", help="wandb project name (default: terrarium)")
    parser.add_argument("--wandb-entity", default=None, help="wandb entity/team")
    parser.add_argument("--mlflow", action="store_true", help="Enable mlflow tracking")
    parser.add_argument("--mlflow-tracking-uri", default=None, help="mlflow tracking URI")
    parser.add_argument("--mlflow-experiment", default="terrarium", help="mlflow experiment name (default: terrarium)")
    parser.add_argument("--mlflow-run-name", default=None, help="mlflow run name")
    parser.add_argument("--max-concurrency", type=int, default=8, help="Max parallel evaluations (default: 8)")

    args = parser.parse_args()

    tracking_config = None
    if args.wandb or args.mlflow:
        tracking_config = TrackingConfig(
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            use_mlflow=args.mlflow,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
            wandb_tags=[args.task],
        )

    result = run(
        args.task,
        args.adapter,
        max_evals=args.max_evals,
        max_concurrency=args.max_concurrency,
        tracking=tracking_config,
    )

    summary = {
        "task": args.task,
        "adapter": args.adapter,
        "best_score": result.best_score,
        "total_evals": result.total_evals,
        "metadata": result.metadata,
    }

    print(json.dumps(summary, indent=2, default=str))

    if args.output:
        out = {**summary, "best_candidate": result.best_candidate}
        Path(args.output).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nFull result written to {args.output}")


if __name__ == "__main__":
    main()
