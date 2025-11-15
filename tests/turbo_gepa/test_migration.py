from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from turbo_gepa.config import Config
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.islands import IslandContext
from turbo_gepa.migrations import FileMigrationBackend, LocalQueueMigrationBackend
from turbo_gepa.orchestrator import Orchestrator


class _SilentLogger:
    def log(self, *args, **kwargs) -> None:  # pragma: no cover - simple sink
        return


class _DummyMigrationBackend:
    def __init__(self) -> None:
        self.published: list[tuple[int, list[str]]] = []
        self.incoming: list[list[Candidate]] = []

    def publish(self, from_island: int, candidates: list[Candidate]) -> None:
        self.published.append((from_island, [c.text for c in candidates]))

    def consume(self, island_id: int) -> list[Candidate]:
        return self.incoming.pop(0) if self.incoming else []


@pytest.mark.asyncio
async def test_orchestrator_maybe_migrate_enqueues_incoming(tmp_path) -> None:
    """Prove that incoming migrations are recorded and enqueued locally."""

    config = Config(
        n_islands=4,
        migration_k=2,
        migration_period=1,
        cache_path=str(tmp_path / "cache"),
        log_path=str(tmp_path / "logs"),
        promote_objective="quality",
    )

    orch = object.__new__(Orchestrator)
    orch.config = config
    orch.island_id = 1
    orch.eval_batches_completed = 1
    orch.show_progress = False
    orch.queue = deque()
    orch.logger = _SilentLogger()
    orch._candidate_island_meta = {}
    orch._migration_events = deque(maxlen=256)

    backend = _DummyMigrationBackend()
    incoming = Candidate("incoming-child", meta={"origin_island": 0})
    backend.incoming.append([incoming])
    orch.migration_backend = backend

    # Archive returns a single elite so the orchestrator publishes + shares it.
    shared_candidate = Candidate("shared-parent", meta={"origin_island": orch.island_id})
    shared_result = EvalResult(objectives={"quality": 0.9}, traces=[], n_examples=1, shard_fraction=1.0)
    orch.archive = SimpleNamespace(
        select_for_generation=lambda *_, **__: [shared_candidate],
        pareto_entries=lambda: [SimpleNamespace(candidate=shared_candidate, result=shared_result)],
    )

    enqueued: list[Candidate] = []

    def _enqueue(candidates):
        enqueued.extend(candidates)

    orch.enqueue = _enqueue

    await Orchestrator._maybe_migrate(orch)

    assert backend.published == [(orch.island_id, ["shared-parent"])]
    assert any(c.text == "shared-parent" for c in enqueued)
    assert any(c.text == "incoming-child" for c in enqueued)
    assert incoming.meta["origin_island"] == 0
    assert incoming.meta["current_island"] == orch.island_id
    assert orch._migration_events, "expected migration events recorded"


def test_file_migration_backend_round_trip(tmp_path) -> None:
    backend = FileMigrationBackend(str(tmp_path), total_islands=4)
    outbound = Candidate("migrant", meta={"origin_island": 2})
    backend.publish(2, [outbound])
    received = backend.consume(3)
    assert [cand.text for cand in received] == ["migrant"]
    received_second = backend.consume(0)
    assert [cand.text for cand in received_second] == ["migrant"]


@pytest.mark.asyncio
async def test_local_queue_backend_broadcasts_all_islands() -> None:
    queues = [asyncio.Queue() for _ in range(3)]
    context = IslandContext(
        inbound=queues[1],
        outbound=queues[2],
        island_id=1,
        all_queues=queues,
    )
    backend = LocalQueueMigrationBackend(context)
    outbound = Candidate("local-migrant", meta={"origin_island": 1})
    backend.publish(1, [outbound])

    received_ids = []
    for idx in (0, 2):
        item = await queues[idx].get()
        received_ids.append((idx, item.text, item.meta.get("origin_island")))

    assert received_ids == [(0, "local-migrant", 1), (2, "local-migrant", 1)]
