import asyncio
import tempfile

from turbo_gepa.cache import DiskCache
from turbo_gepa.interfaces import Candidate, EvalResult


def test_disk_cache_round_trip():
    candidate = Candidate(text="prompt")
    result = EvalResult(
        objectives={"quality": 0.5},
        traces=[{"out": "foo", "example_id": "ex-1"}],
        n_examples=1,
        example_ids=["ex-1"],
    )

    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            await cache.set(candidate, "ex-1", result)
            cached = await cache.get(candidate, "ex-1")
            assert cached is not None
            assert cached.objectives["quality"] == 0.5
            assert cached.example_ids == ["ex-1"]

    asyncio.run(run())
