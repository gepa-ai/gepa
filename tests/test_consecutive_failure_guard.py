# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the consecutive-failure guard (max_consecutive_failures).

The guard tracks consecutive iterations where the proposer fails to produce
a candidate (proposal is None), which covers:
- Reflection LM errors (caught internally by the proposer → returns None)
- Exceptions that propagate to the main loop (gepa.optimize custom adapters)
"""

import pytest

from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    optimize_anything,
)


def _good_evaluator(candidate):
    return 0.5, {}


def _mock_lm_always_fail(prompt):
    raise ValueError("reflection LM is broken")


class TestConsecutiveFailureGuard:
    def test_raises_after_n_consecutive_failures(self):
        """RuntimeError is raised once max_consecutive_failures is reached.

        The reflection LM always raises, so the proposer always returns None.
        """
        with pytest.raises(RuntimeError, match="consecutive iterations"):
            optimize_anything(
                seed_candidate="x",
                evaluator=_good_evaluator,
                config=GEPAConfig(
                    engine=EngineConfig(
                        max_metric_calls=50,
                        raise_on_exception=False,
                        max_consecutive_failures=3,
                    ),
                    reflection=ReflectionConfig(reflection_lm=_mock_lm_always_fail),
                ),
            )

    def test_error_message_includes_exceptions_list(self):
        """The RuntimeError message lists all collected intermediate exceptions."""
        with pytest.raises(RuntimeError, match="Exceptions collected"):
            optimize_anything(
                seed_candidate="x",
                evaluator=_good_evaluator,
                config=GEPAConfig(
                    engine=EngineConfig(
                        max_metric_calls=50,
                        raise_on_exception=False,
                        max_consecutive_failures=2,
                    ),
                    reflection=ReflectionConfig(reflection_lm=_mock_lm_always_fail),
                ),
            )

    def test_error_message_mentions_raise_on_exception(self):
        """The RuntimeError message hints at raise_on_exception for debugging."""
        with pytest.raises(RuntimeError, match="raise_on_exception"):
            optimize_anything(
                seed_candidate="x",
                evaluator=_good_evaluator,
                config=GEPAConfig(
                    engine=EngineConfig(
                        max_metric_calls=50,
                        raise_on_exception=False,
                        max_consecutive_failures=1,
                    ),
                    reflection=ReflectionConfig(reflection_lm=_mock_lm_always_fail),
                ),
            )

    def test_disabled_when_zero(self):
        """Setting max_consecutive_failures=0 disables the guard entirely."""
        result = optimize_anything(
            seed_candidate="x",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=5,
                    raise_on_exception=False,
                    max_consecutive_failures=0,  # disabled
                ),
                reflection=ReflectionConfig(reflection_lm=_mock_lm_always_fail),
            ),
        )
        assert result is not None
        assert len(result.candidates) == 1  # only seed, all iterations produced nothing

    def test_counter_resets_on_proposal_generated(self):
        """Counter resets when the proposer produces a candidate.

        2 failures then 1 success means no guard trigger with
        max_consecutive_failures=3.
        """
        call_count = [0]

        def lm_transient_then_good(prompt):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("transient failure")
            return "```\nx\n```"

        result = optimize_anything(
            seed_candidate="x",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=10,
                    raise_on_exception=False,
                    max_consecutive_failures=3,
                ),
                reflection=ReflectionConfig(reflection_lm=lm_transient_then_good),
            ),
        )
        assert result is not None

    def test_default_is_5(self):
        """max_consecutive_failures defaults to 5 in EngineConfig."""
        assert EngineConfig().max_consecutive_failures == 5

    def test_raise_on_exception_true_disables_guard(self):
        """When raise_on_exception=True, the consecutive-failure guard is inactive.

        The guard only operates when raise_on_exception=False.  With True, any
        exception that reaches the main loop is re-raised immediately — but LM
        errors caught internally by the proposer still return None silently
        (the main loop never sees them).  The run completes by budget.
        """
        # With raise_on_exception=True and an always-failing LM, the proposer
        # returns None on every iteration (LM error caught internally).
        # The main loop continues to the budget without raising.
        result = optimize_anything(
            seed_candidate="x",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=3,
                    raise_on_exception=True,
                    max_consecutive_failures=1,  # guard inactive when raise_on_exception=True
                ),
                reflection=ReflectionConfig(reflection_lm=_mock_lm_always_fail),
            ),
        )
        # Completes with only seed — guard didn't fire (it's disabled)
        assert result is not None

    def test_gepa_optimize_api_accepts_param(self):
        """max_consecutive_failures is accepted and functional via gepa.optimize().

        Uses a custom GEPAAdapter whose evaluate() always fails after seed eval —
        adapter errors propagate through the main loop exception handler.
        """
        import gepa
        from gepa.core.adapter import EvaluationBatch

        call_count = [0]

        class FlakeyAdapter:
            propose_new_texts = None

            def evaluate(self, batch, candidate, capture_traces=False):
                call_count[0] += 1
                if call_count[0] == 1:
                    return EvaluationBatch(
                        outputs=[0.5] * len(batch), scores=[0.5] * len(batch)
                    )
                raise ValueError("adapter broken")

            def make_reflective_dataset(self, candidate, eval_batch, components):
                return {c: [] for c in components}

        with pytest.raises(RuntimeError, match="consecutive iterations"):
            gepa.optimize(
                seed_candidate={"p": "x"},
                trainset=[{"q": "a"}],
                valset=[{"q": "b"}],
                adapter=FlakeyAdapter(),
                reflection_lm="openai/gpt-5",
                max_metric_calls=50,
                raise_on_exception=False,
                max_consecutive_failures=2,
            )
