# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end tests for the vectorized (batched) proposal pipeline.

``num_parallel_proposals`` drives how many proposals each iteration produces.
N=1 is the default (behavior-preserving); N>1 exercises the batched engine path
where one iteration evaluates/reflects/evaluates a batch of tasks, with the
reflection LM calls issued together.
"""

import threading
import time

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)


def _evaluator(candidate):
    # Score is the length of the candidate text — gives the optimizer a gradient
    # to follow so multiple candidates get accepted across iterations.
    return float(len(str(candidate))), {}


class ConcurrencyTrackingLM:
    """Reflection LM that records the peak number of concurrent in-flight calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.total_calls = 0

    def __call__(self, prompt):
        with self._lock:
            self.in_flight += 1
            self.total_calls += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Block briefly so that concurrently-issued calls actually overlap
            # (a constant-time mock would finish before its siblings start).
            time.sleep(0.02)
            return "```\nan improved and noticeably longer candidate instruction\n```"
        finally:
            with self._lock:
                self.in_flight -= 1


def _run(num_parallel_proposals, lm, max_metric_calls=24):
    return optimize_anything(
        seed_candidate="seed",
        evaluator=_evaluator,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=max_metric_calls,
                num_parallel_proposals=num_parallel_proposals,
            ),
            reflection=ReflectionConfig(reflection_lm=lm, reflection_minibatch_size=1),
        ),
    )


def test_batched_run_completes_and_optimizes():
    """A run with num_parallel_proposals=3 completes and explores candidates."""
    lm = ConcurrencyTrackingLM()
    result = _run(num_parallel_proposals=3, lm=lm)

    assert result is not None
    assert lm.total_calls > 0


def test_single_and_batched_are_both_valid():
    """N=1 and N>1 both produce a valid result (the batched path is the only path)."""
    r1 = _run(num_parallel_proposals=1, lm=ConcurrencyTrackingLM())
    r3 = _run(num_parallel_proposals=3, lm=ConcurrencyTrackingLM())
    assert r1 is not None
    assert r3 is not None


def test_batched_reflection_runs_concurrently():
    """With N>1, reflection LM calls within an iteration overlap (vectorized)."""
    lm = ConcurrencyTrackingLM()
    _run(num_parallel_proposals=4, lm=lm, max_metric_calls=40)
    # If reflection were issued strictly sequentially, peak concurrency would be 1.
    # The batched reflector fans the per-task calls out at once.
    assert lm.max_in_flight >= 2
