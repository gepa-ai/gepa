"""
Async island orchestration for TurboGEPA.

Uses asyncio tasks instead of multiprocessing for concurrent island optimization.
Islands share the same process but run concurrently via async/await.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, List

from .interfaces import Candidate


@dataclass
class IslandContext:
    """Holds migration queues for an island."""

    inbound: "asyncio.Queue[Candidate]"
    outbound: "asyncio.Queue[Candidate]"
    island_id: int
    metrics_queue: "asyncio.Queue[dict] | None" = None  # Optional metrics reporting


async def spawn_islands(
    n_islands: int,
    worker: Callable[[IslandContext], Awaitable[None]],
    metrics_queue: "asyncio.Queue[dict] | None" = None,
) -> List[asyncio.Task]:
    """Create and start island tasks running concurrently.

    Args:
        n_islands: Number of islands to spawn
        worker: Async function to run on each island
    """
    tasks: List[asyncio.Task] = []
    queues = [asyncio.Queue() for _ in range(n_islands)]

    for idx in range(n_islands):
        inbound = queues[idx]
        outbound = queues[(idx + 1) % n_islands]
        context = IslandContext(inbound, outbound, island_id=idx, metrics_queue=metrics_queue)
        task = asyncio.create_task(worker(context))
        tasks.append(task)

    return tasks


def migrate_out(context: IslandContext, candidates: Iterable[Candidate]) -> None:
    """Send elites to the next island without blocking."""
    for candidate in candidates:
        try:
            context.outbound.put_nowait(candidate)
        except asyncio.QueueFull:
            break


def integrate_in(context: IslandContext) -> List[Candidate]:
    """Import elites received from the previous island."""
    received: List[Candidate] = []
    while True:
        try:
            received.append(context.inbound.get_nowait())
        except asyncio.QueueEmpty:
            break
    return received
