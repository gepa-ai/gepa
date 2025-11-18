"""
Quick local benchmark for run_local_multiworker using the AIME adapter factory.

Usage:
    python examples/local_multiworker_bench.py

This will launch 4 worker processes, each running one island, for a total
of 4 islands across processes. It prints the wall-clock time for the
multiworker run. Requires OPENROUTER_* env to be configured as usual.
"""

from __future__ import annotations

import os
import time

from turbo_gepa.distributed import run_local_multiworker


def main() -> None:
    # Configure the AIME factory via environment vars so workers share settings.
    os.environ.setdefault("AIME_DATASET_SIZE", "30")
    os.environ.setdefault("AIME_EVAL_CONCURRENCY", "20")
    os.environ.setdefault("AIME_TARGET_QUALITY", "0.733")
    os.environ.setdefault("AIME_MAX_RUNTIME", "240")
    os.environ.setdefault("AIME_VERIFICATION_SPEED_BIAS", "0.8")
    os.environ.setdefault("AIME_N_ISLANDS", "4")

    factory_str = "examples.local_turbo_aime_factory:adapter_factory"
    control_dir = ".turbo_gepa_multi/control"
    run_id = "local_multi_4workers"

    start = time.time()
    run_local_multiworker(
        factory=factory_str,
        worker_count=4,
        islands_per_worker=1,
        max_rounds=None,
        max_evaluations=None,
        control_dir=control_dir,
        run_id=run_id,
        display_progress=True,
        enable_auto_stop=True,
    )
    elapsed = time.time() - start
    print(f"\nLocal multiworker (4 workers, 4 islands) wall-time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

