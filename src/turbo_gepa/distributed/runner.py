"""
Distributed runner utilities for spawning TurboGEPA workers.

Each worker process is responsible for a subset of islands and shares cache/log
directories via a mounted filesystem (e.g., Modal volumes).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
from typing import Any, Callable, Sequence

from turbo_gepa.adapters.default_adapter import DefaultAdapter

FactoryType = Callable[[], Any]


def _resolve_factory(factory: str | FactoryType) -> FactoryType:
    if callable(factory):
        return factory  # type: ignore[return-value]
    if ":" not in factory:
        raise ValueError("Factory must be in the form 'module:function'")
    module_name, func_name = factory.split(":", 1)
    module = importlib.import_module(module_name)
    factory_obj = getattr(module, func_name, None)
    if factory_obj is None or not callable(factory_obj):
        raise ValueError(f"Factory callable '{factory}' not found")
    return factory_obj  # type: ignore[return-value]


def _prepare_adapter(
    adapter: DefaultAdapter,
    *,
    worker_id: int,
    worker_count: int,
    islands_per_worker: int | None,
) -> None:
    adapter.config.worker_id = worker_id
    adapter.config.worker_count = worker_count
    if islands_per_worker is not None:
        adapter.config.islands_per_worker = islands_per_worker
    if not getattr(adapter.config, "migration_backend", None):
        adapter.config.migration_backend = "volume"
    if not getattr(adapter.config, "migration_path", None):
        adapter.config.migration_path = os.path.join(adapter.base_cache_dir, "migrations")


def run_worker_from_factory(
    *,
    factory: str | FactoryType,
    package: str | None = None,
    worker_id: int,
    worker_count: int,
    islands_per_worker: int | None = None,
    seeds: Sequence[str] | None = None,
    max_rounds: int | None = None,
    max_evaluations: int | None = None,
    display_progress: bool = True,
    enable_auto_stop: bool = True,
    control_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Instantiate an adapter via ``factory`` and run the assigned islands.

    The factory callable may return either a DefaultAdapter or a tuple of
    (DefaultAdapter, Sequence[str|Candidate]) to provide custom seeds.
    """

    if package:
        import importlib
        importlib.import_module(package)
    factory_fn = _resolve_factory(factory)
    built = factory_fn()
    adapter: DefaultAdapter
    default_seeds: Sequence[str] | None = None
    if isinstance(built, tuple) and len(built) == 2:
        adapter = built[0]
        default_seeds = built[1]
    else:
        adapter = built
    if not isinstance(adapter, DefaultAdapter):
        raise TypeError("Factory must return a DefaultAdapter or (DefaultAdapter, seeds)")
    seed_values = seeds if seeds is not None else default_seeds
    _prepare_adapter(
        adapter,
        worker_id=worker_id,
        worker_count=worker_count,
        islands_per_worker=islands_per_worker,
    )
    if control_dir:
        resolved_control = os.path.abspath(control_dir)
        os.makedirs(resolved_control, exist_ok=True)
        adapter.control_dir = resolved_control  # type: ignore[attr-defined]
        adapter.config.control_dir = resolved_control
    if run_id:
        adapter._forced_run_token = run_id  # type: ignore[attr-defined]
    payload = asyncio.run(
        adapter.optimize_async(
            seed_values or ["You are a helpful assistant."],
            max_rounds=max_rounds,
            max_evaluations=max_evaluations,
            display_progress=display_progress,
            enable_auto_stop=enable_auto_stop,
        )
    )
    return payload


def _load_seeds_from_path(path: str | None) -> list[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Seed file must contain a JSON list")
    return [str(item) for item in data]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a TurboGEPA worker for distributed islands.")
    parser.add_argument("--factory", required=True, help="Import path to callable returning DefaultAdapter or (adapter, seeds).")
    parser.add_argument("--worker-id", type=int, required=True, help="Zero-based worker identifier.")
    parser.add_argument("--worker-count", type=int, required=True, help="Total number of workers participating.")
    parser.add_argument("--islands-per-worker", type=int, help="Explicit islands per worker (defaults to even split).")
    parser.add_argument("--seeds-json", help="Path to JSON file containing a list of seed prompts.")
    parser.add_argument(
        "--seed",
        dest="seed_values",
        action="append",
        help="Seed prompt literal (repeatable). Overrides seeds-json and factory defaults.",
    )
    parser.add_argument("--control-dir", help="Shared directory for control/stop files.")
    parser.add_argument("--run-id", help="Global run ID shared by all workers (defaults to random).")
    parser.add_argument("--max-rounds", type=int, default=None, help="Optional maximum rounds per worker.")
    parser.add_argument("--max-evaluations", type=int, default=None, help="Optional evaluation budget per worker.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    parser.add_argument("--no-auto-stop", action="store_true", help="Disable stop governor auto-stop.")
    parser.add_argument("--output-json", help="Optional path to dump the run metadata.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    explicit_seeds = args.seed_values or _load_seeds_from_path(args.seeds_json)
    result = run_worker_from_factory(
        factory=args.factory,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
        islands_per_worker=args.islands_per_worker,
        seeds=explicit_seeds,
        max_rounds=args.max_rounds,
        max_evaluations=args.max_evaluations,
        display_progress=not args.quiet,
        enable_auto_stop=not args.no_auto_stop,
        control_dir=args.control_dir,
        run_id=args.run_id,
    )
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result.get("run_metadata", {}), handle, indent=2)


if __name__ == "__main__":
    main()
