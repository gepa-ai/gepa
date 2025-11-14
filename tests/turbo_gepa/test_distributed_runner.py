from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest

from turbo_gepa.distributed.runner import run_worker_from_factory

FACTORY_PATH = "tests.turbo_gepa.distributed_fixtures:build_test_adapter"


def _worker_entry(cache_dir: str, log_dir: str, worker_id: int, worker_count: int) -> None:
    os.environ["TURBOGEPA_CACHE_PATH"] = cache_dir
    os.environ["TURBOGEPA_LOG_PATH"] = log_dir
    run_worker_from_factory(
        factory=FACTORY_PATH,
        worker_id=worker_id,
        worker_count=worker_count,
        islands_per_worker=2,
        max_rounds=1,
        max_evaluations=8,
        display_progress=False,
        enable_auto_stop=False,
    )


@pytest.mark.skipif(os.name == "nt", reason="multiprocessing semantics differ on Windows")
def test_distributed_runner_multiprocess(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[mp.Process] = []
    for worker_id in range(2):
        p = mp.Process(
            target=_worker_entry,
            args=(str(cache_dir), str(log_dir), worker_id, 2),
        )
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    migration_dir = cache_dir / "migrations"
    assert migration_dir.exists()
    # Ensure at least one evaluation record was written
    shard_dirs = list(cache_dir.glob("**/*.jsonl"))
    assert shard_dirs, "Expected shared cache to contain evaluation records"
