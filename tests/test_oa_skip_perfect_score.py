# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Perfect-score skipping in the optimize_anything launcher.

``skip_perfect_score`` is on by default. Without an explicit ``perfect_score``
the launcher assumes 1.0, logs that, and turns skipping off once a higher
score shows up, so unbounded scores never stall the search.
"""

import pytest

from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, TrackingConfig, optimize_anything

ASSUMED = "skip_perfect_score: assuming a perfect score of 1.0"
DISABLED = "disabling skip_perfect_score for this run"
SKIPPED = "All subsample scores perfect"


class _ListLogger:
    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str):
        self.messages.append(message)

    def has(self, text: str) -> bool:
        return any(text in m for m in self.messages)


def _run(score: float, **reflection_kwargs):
    logger = _ListLogger()
    reflection_calls = []

    def reflection_lm(prompt):
        reflection_calls.append(prompt)
        return f"improved candidate v{len(reflection_calls)}"

    def evaluator(candidate, example):
        return score, {"note": "ok"}

    data = [f"ex{i}" for i in range(4)]
    optimize_anything(
        seed_candidate="seed",
        evaluator=evaluator,
        dataset=data,
        valset=data,
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=16, parallel=False, seed=0),
            reflection=ReflectionConfig(reflection_lm=reflection_lm, reflection_minibatch_size=2, **reflection_kwargs),
            tracking=TrackingConfig(logger=logger),
        ),
    )
    return logger, reflection_calls


def test_default_skips_perfect_minibatches_and_logs_the_assumed_bound():
    logger, reflection_calls = _run(1.0)
    assert logger.has(ASSUMED)
    assert logger.has(SKIPPED)
    assert not logger.has(DISABLED)
    assert reflection_calls == []


def test_score_above_the_assumed_bound_turns_skipping_off():
    logger, reflection_calls = _run(2.0)
    assert logger.has(ASSUMED)
    assert logger.has("Observed score 2.0 > assumed perfect score 1.0")
    assert not logger.has(SKIPPED)
    assert reflection_calls


def test_explicit_perfect_score_is_used_without_the_assumption():
    logger, reflection_calls = _run(2.0, perfect_score=2.0)
    assert not logger.has(ASSUMED)
    assert not logger.has(DISABLED)
    assert logger.has(SKIPPED)
    assert reflection_calls == []


@pytest.mark.parametrize("score", [1.0, 2.0])
def test_skip_perfect_score_false_always_reflects(score):
    logger, reflection_calls = _run(score, skip_perfect_score=False)
    assert not logger.has(ASSUMED)
    assert not logger.has(SKIPPED)
    assert reflection_calls
