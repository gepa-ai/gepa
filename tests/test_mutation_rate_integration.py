# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Integration tests: mutation_rate validation on ReflectionConfig and optimize()."""

from __future__ import annotations

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
