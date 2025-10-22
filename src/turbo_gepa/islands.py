"""
Multi-process island orchestration for TurboGEPA.

The implementation provides thin wrappers around ``multiprocessing`` queues to
enable non-blocking migrations between islands. Actual evaluation happens in
the orchestrator loop, keeping island management lightweight.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Iterable, List
from queue import Empty, Full

from .interfaces import Candidate


@dataclass
class IslandContext:
    """Holds migration queues for an island."""

    inbound: "mp.Queue[Candidate]"
    outbound: "mp.Queue[Candidate]"


def spawn_islands(
    n_islands: int,
    worker: Callable[[IslandContext], None],
    *,
    start_method: str = "spawn",
) -> List[mp.Process]:
    """Create and start island processes."""
    ctx = mp.get_context(start_method)
    processes: List[mp.Process] = []
    queues = [ctx.Queue() for _ in range(n_islands)]
    for idx in range(n_islands):
        inbound = queues[idx]
        outbound = queues[(idx + 1) % n_islands]
        process = ctx.Process(target=worker, args=(IslandContext(inbound, outbound),), daemon=True)
        process.start()
        processes.append(process)
    return processes


def migrate_out(context: IslandContext, candidates: Iterable[Candidate]) -> None:
    """Send elites to the next island without blocking."""
    for candidate in candidates:
        try:
            context.outbound.put_nowait(candidate)
        except Full:
            break


def integrate_in(context: IslandContext) -> List[Candidate]:
    """Import elites received from the previous island."""
    received: List[Candidate] = []
    while True:
        try:
            received.append(context.inbound.get_nowait())
        except Empty:
            break
    return received
