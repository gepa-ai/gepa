# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Integration tests: mutation_rate validation on ReflectionConfig and optimize()."""

from __future__ import annotations

import pytest

from gepa.optimize_anything import ReflectionConfig


class TestMutationRateValidation:
    def test_reflection_config_accepts_valid_rate(self) -> None:
        """ReflectionConfig accepts mutation_rate in [0.0, 1.0]."""
        config = ReflectionConfig(mutation_rate=0.5)
        assert config.mutation_rate == 0.5

    def test_reflection_config_default_is_1(self) -> None:
        """Default mutation_rate is 1.0 (full rewrite allowed)."""
        config = ReflectionConfig()
        assert config.mutation_rate == 1.0

    def test_optimize_rejects_mutation_rate_above_1(self) -> None:
        """gepa.optimize() raises ValueError for mutation_rate > 1.0."""
        import gepa

        with pytest.raises(ValueError, match="mutation_rate must be between"):
            gepa.optimize(
                seed_candidate={"instructions": "test"},
                trainset=[{"input": "x", "answer": "y", "additional_context": {}}],
                reflection_lm=lambda p: "ok",
                mutation_rate=1.5,
                max_metric_calls=1,
            )

    def test_optimize_rejects_negative_mutation_rate(self) -> None:
        """gepa.optimize() raises ValueError for negative mutation_rate."""
        import gepa

        with pytest.raises(ValueError, match="mutation_rate must be between"):
            gepa.optimize(
                seed_candidate={"instructions": "test"},
                trainset=[{"input": "x", "answer": "y", "additional_context": {}}],
                reflection_lm=lambda p: "ok",
                mutation_rate=-0.1,
                max_metric_calls=1,
            )
