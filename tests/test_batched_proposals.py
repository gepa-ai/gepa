# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end tests for the vectorized (batched) proposal pipeline.

``num_parallel_proposals`` drives how many proposals each iteration produces.
N=1 is the default (behavior-preserving); N>1 exercises the batched engine path
where one iteration evaluates/reflects/evaluates a batch of tasks. The reflection
LM calls are issued together via the LM's batched completion when available.
"""

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


class CountingLM:
    """Custom reflection callable that counts invocations."""

    def __init__(self):
        self.total_calls = 0

    def __call__(self, prompt):
        self.total_calls += 1
        return "```\nan improved and noticeably longer candidate instruction\n```"


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
    lm = CountingLM()
    result = _run(num_parallel_proposals=3, lm=lm)

    assert result is not None
    assert lm.total_calls > 0


def test_single_and_batched_are_both_valid():
    """N=1 and N>1 both produce a valid result (the batched path is the only path)."""
    r1 = _run(num_parallel_proposals=1, lm=CountingLM())
    r3 = _run(num_parallel_proposals=3, lm=CountingLM())
    assert r1 is not None
    assert r3 is not None
