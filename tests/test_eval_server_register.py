# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Concurrency regression for ``EvalServer._register_candidate``.

``log_progress`` can be called from many threads at once (HTTP handlers,
batched valset evals). Id allocation used to be an unlocked read-modify-write
of ``_next_candidate_id`` / ``_candidate_registry``, which under contention
could hand out duplicate ids or drop registry entries. Allocation now happens
under ``self._lock``.
"""

from __future__ import annotations

import threading

from gepa.oa.budget import BudgetTracker
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task


def _server():
    task = Task(name="reg", seed_candidate="seed")
    return EvalServer(task, evaluate=lambda candidate: (0.0, {}), budget=BudgetTracker(max_evals=None))


def _run_concurrently(fns):
    barrier = threading.Barrier(len(fns))

    def wrapped(fn):
        barrier.wait()  # maximize the overlap window
        fn()

    threads = [threading.Thread(target=wrapped, args=(fn,)) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_distinct_candidates_get_unique_contiguous_ids():
    server = _server()
    n = 64
    results: list[int] = []
    results_lock = threading.Lock()

    def register(i):
        def go():
            cid = server._register_candidate(f"candidate-{i}")
            with results_lock:
                results.append(cid)

        return go

    _run_concurrently([register(i) for i in range(n)])

    # No duplicate ids, and the id space is exactly 0..n-1 (no gaps, no
    # clobbered registry entries).
    assert len(results) == n
    assert sorted(results) == list(range(n))
    assert server._next_candidate_id == n


def test_same_candidate_registered_concurrently_gets_one_id():
    server = _server()
    n = 64
    results: list[int] = []
    results_lock = threading.Lock()

    def register():
        cid = server._register_candidate("the-one-candidate")
        with results_lock:
            results.append(cid)

    _run_concurrently([register for _ in range(n)])

    # Every concurrent registration of the same candidate resolves to a single
    # shared id; exactly one id was ever allocated.
    assert len(results) == n
    assert set(results) == {0}
    assert server._next_candidate_id == 1
