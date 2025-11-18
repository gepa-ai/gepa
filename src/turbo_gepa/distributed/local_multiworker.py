"""
Local multiworker helper for TurboGEPA.

This module provides a thin convenience wrapper around
``run_worker_from_factory`` so you can launch multiple workers as separate
processes on a single machine without shell scripting.
"""

from __future__ import annotations

from multiprocessing import Process
from typing import Sequence

from .runner import run_worker_from_factory


def run_local_multiworker(
    *,
    factory: str,
    worker_count: int,
    islands_per_worker: int | None = None,
    seeds: Sequence[str] | None = None,
    max_rounds: int | None = None,
    max_evaluations: int | None = None,
    control_dir: str | None = None,
    run_id: str | None = None,
    display_progress: bool = True,
    enable_auto_stop: bool = True,
) -> None:
    """
    Launch multiple TurboGEPA workers as local processes.

    This helper is a convenience for local experimentation. It mirrors the
    arguments of ``run_worker_from_factory`` but spawns ``worker_count``
    separate processes, each with its own event loop and HTTP client.

    Args:
        factory: Import path to a callable returning DefaultAdapter or
            (DefaultAdapter, seeds), e.g. ``\"examples.modal_turbo_aime:adapter_factory\"``.
        worker_count: Number of worker processes to launch.
        islands_per_worker: Optional islands per worker override.
        seeds: Optional shared seed list for all workers.
        max_rounds: Optional maximum rounds per worker.
        max_evaluations: Optional evaluation budget per worker.
        control_dir: Shared control directory for stop files.
        run_id: Global run identifier shared by all workers.
        display_progress: Whether each worker logs progress.
        enable_auto_stop: Whether to enable the stop governor.
    """
    processes: list[Process] = []

    for worker_id in range(worker_count):
        proc = Process(
            target=run_worker_from_factory,
            kwargs={
                "factory": factory,
                "worker_id": worker_id,
                "worker_count": worker_count,
                "islands_per_worker": islands_per_worker,
                "seeds": seeds,
                "max_rounds": max_rounds,
                "max_evaluations": max_evaluations,
                "display_progress": display_progress,
                "enable_auto_stop": enable_auto_stop,
                "control_dir": control_dir,
                "run_id": run_id,
            },
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

