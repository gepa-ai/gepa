"""Test async island orchestration."""

import asyncio

import pytest

from turbo_gepa.interfaces import Candidate
from turbo_gepa.islands import IslandContext, integrate_in, migrate_out, spawn_islands


def test_spawn_islands():
    """Test that islands can be spawned and run concurrently."""

    async def async_test():
        completed_islands = []

        async def simple_worker(context: IslandContext) -> None:
            """Simple worker that just records completion."""
            await asyncio.sleep(0.01)  # Simulate work
            completed_islands.append(context.island_id)

        tasks = await spawn_islands(n_islands=3, worker=simple_worker)
        await asyncio.gather(*tasks)

        assert len(completed_islands) == 3
        assert set(completed_islands) == {0, 1, 2}

    asyncio.run(async_test())


def test_island_migration():
    """Test that candidates can migrate between islands."""

    async def async_test():
        received_by_island = {}

        async def migration_worker(context: IslandContext) -> None:
            """Worker that sends and receives candidates."""
            # Send a candidate to next island
            candidate = Candidate(
                text=f"Candidate from island {context.island_id}",
                meta={"source_island": context.island_id},
            )
            migrate_out(context, [candidate])

            # Wait a bit for migration to happen
            await asyncio.sleep(0.05)

            # Receive candidates from previous island
            received = integrate_in(context)
            received_by_island[context.island_id] = received

        tasks = await spawn_islands(n_islands=3, worker=migration_worker)
        await asyncio.gather(*tasks)

        # Each island should have received a candidate from the previous island
        # Island 0 receives from island 2, island 1 from island 0, etc.
        assert len(received_by_island) == 3
        for island_id in range(3):
            received = received_by_island[island_id]
            assert len(received) >= 1, f"Island {island_id} should have received candidates"
            # Check the source island ID (ring topology)
            prev_island = (island_id - 1) % 3
            assert received[0].meta["source_island"] == prev_island

    asyncio.run(async_test())


def test_migrate_out_sync():
    """Test migrate_out function with a mock context."""
    queue = asyncio.Queue()
    context = IslandContext(
        inbound=asyncio.Queue(),
        outbound=queue,
        island_id=0,
    )

    candidates = [
        Candidate(text="Candidate 1", meta={}),
        Candidate(text="Candidate 2", meta={}),
    ]

    migrate_out(context, candidates)

    # Should have enqueued both candidates
    assert queue.qsize() == 2


def test_integrate_in_sync():
    """Test integrate_in function with a mock context."""
    queue = asyncio.Queue()
    context = IslandContext(
        inbound=queue,
        outbound=asyncio.Queue(),
        island_id=0,
    )

    # Add some candidates to the queue
    candidates = [
        Candidate(text="Candidate 1", meta={}),
        Candidate(text="Candidate 2", meta={}),
    ]
    for c in candidates:
        queue.put_nowait(c)

    # Integrate them
    received = integrate_in(context)

    assert len(received) == 2
    assert received[0].text == "Candidate 1"
    assert received[1].text == "Candidate 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
