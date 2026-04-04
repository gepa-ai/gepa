# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for GEPACallback support in optimize_anything."""

from unittest.mock import MagicMock

from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    optimize_anything,
)


def _good_evaluator(candidate):
    return 0.5, {}


def _mock_lm(prompt):
    return "```\ncandidate\n```"


class TestOptimizeAnythingCallbacks:
    def test_callbacks_receive_events(self):
        """Callbacks passed via GEPAConfig receive optimization events."""
        events_received: list[str] = []

        class Recorder:
            def on_optimization_start(self, event):
                events_received.append("optimization_start")

            def on_iteration_start(self, event):
                events_received.append("iteration_start")

            def on_iteration_end(self, event):
                events_received.append("iteration_end")

            def on_optimization_end(self, event):
                events_received.append("optimization_end")

        result = optimize_anything(
            seed_candidate="x",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                callbacks=[Recorder()],
                engine=EngineConfig(max_metric_calls=5),
                reflection=ReflectionConfig(reflection_lm=_mock_lm),
            ),
        )

        assert result is not None
        assert "optimization_start" in events_received
        assert "iteration_start" in events_received
        assert "iteration_end" in events_received
        assert "optimization_end" in events_received

    def test_multiple_callbacks(self):
        """Multiple callbacks all receive events."""
        counts = [0, 0]

        class Counter1:
            def on_iteration_end(self, event):
                counts[0] += 1

        class Counter2:
            def on_iteration_end(self, event):
                counts[1] += 1

        optimize_anything(
            seed_candidate="x",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                callbacks=[Counter1(), Counter2()],
                engine=EngineConfig(max_metric_calls=5),
                reflection=ReflectionConfig(reflection_lm=_mock_lm),
            ),
        )

        assert counts[0] > 0
        assert counts[0] == counts[1]

    def test_no_callbacks_default(self):
        """Without callbacks, optimization still works."""
        result = optimize_anything(
            seed_candidate="x",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=3),
                reflection=ReflectionConfig(reflection_lm=_mock_lm),
            ),
        )
        assert result is not None

    def test_proposal_events_received(self):
        """Callbacks receive on_proposal_start and on_proposal_end events."""
        events: list[str] = []

        class ProposalRecorder:
            def on_proposal_start(self, event):
                events.append("proposal_start")

            def on_proposal_end(self, event):
                events.append("proposal_end")

        optimize_anything(
            seed_candidate="x",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                callbacks=[ProposalRecorder()],
                engine=EngineConfig(max_metric_calls=5),
                reflection=ReflectionConfig(reflection_lm=_mock_lm),
            ),
        )

        assert "proposal_start" in events
        assert "proposal_end" in events
