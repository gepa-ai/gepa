# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for refiner functionality with the number guessing optimization scenario."""

import tempfile
from pathlib import Path

import pytest

from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    RefinerConfig,
    optimize_anything,
)

# Golden number to guess
GOLDEN_NUMBER = 42


def create_fitness_fn(call_counter: dict):
    """Create a fitness_fn that counts calls and scores based on distance from golden number."""

    def fitness_fn(candidate: dict[str, str], **kwargs) -> tuple[float, dict]:
        call_counter["count"] += 1

        try:
            guess = int(candidate["number"])
        except (ValueError, KeyError):
            guess = 0

        off_by = abs(guess - GOLDEN_NUMBER)
        score = -off_by

        side_info = {
            "guess": guess,
            "golden": GOLDEN_NUMBER,
            "off_by": off_by,
        }

        return score, side_info

    return fitness_fn


class TestRefiner:
    """Tests for refiner functionality."""

    def test_refiner_without_caching(self):
        """Test refiner works without caching."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=RefinerConfig(
                refiner_lm="openrouter/openai/gpt-4.1-nano",
                max_refinements=2,
            ),
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            fitness_fn=fitness_fn,
            objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0.",
            config=config,
        )

        assert result is not None
        print(f"\n[Refiner no cache] Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}")
        print(f"Best candidate: {result.best_candidate}")
        print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")

    def test_refiner_with_memory_cache(self):
        """Test refiner works with memory caching."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=RefinerConfig(
                refiner_lm="openrouter/openai/gpt-4.1-nano",
                max_refinements=2,
            ),
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            fitness_fn=fitness_fn,
            objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0.",
            config=config,
            cache_evaluation_storage="memory",
        )

        assert result is not None
        print(f"\n[Refiner + memory cache] Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}")
        print(f"Best candidate: {result.best_candidate}")
        print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")

    def test_refiner_with_disk_cache(self):
        """Test refiner works with disk caching."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            call_counter = {"count": 0}
            fitness_fn = create_fitness_fn(call_counter)

            config = GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=5,
                    cache_evaluation=True,
                    run_dir=tmp_dir,
                ),
                reflection=ReflectionConfig(
                    reflection_lm="openrouter/openai/gpt-4.1-nano",
                ),
                refiner=RefinerConfig(
                    refiner_lm="openrouter/openai/gpt-4.1-nano",
                    max_refinements=2,
                ),
            )

            result = optimize_anything(
                seed_candidate={"number": "50"},
                fitness_fn=fitness_fn,
                objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0.",
                config=config,
                cache_evaluation_storage="disk",
            )

            assert result is not None
            print(f"\n[Refiner + disk cache] Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}")
            print(f"Best candidate: {result.best_candidate}")
            print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")

            # Verify cache files created
            cache_dir = Path(tmp_dir) / "fitness_cache"
            assert cache_dir.exists()
            cache_files = list(cache_dir.glob("*.pkl"))
            print(f"Cache files: {len(cache_files)}")

    def test_refiner_cache_reduces_calls(self):
        """Test that caching reduces actual fitness_fn calls with refiner."""
        call_counter_no_cache = {"count": 0}
        call_counter_with_cache = {"count": 0}

        # Run without cache
        fitness_fn_no_cache = create_fitness_fn(call_counter_no_cache)
        config_no_cache = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=3,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=RefinerConfig(
                refiner_lm="openrouter/openai/gpt-4.1-nano",
                max_refinements=3,
            ),
        )

        result_no_cache = optimize_anything(
            seed_candidate={"number": "50"},
            fitness_fn=fitness_fn_no_cache,
            objective="Guess the golden integer. off_by shows distance from target.",
            config=config_no_cache,
        )

        # Run with cache
        fitness_fn_with_cache = create_fitness_fn(call_counter_with_cache)
        config_with_cache = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=3,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=RefinerConfig(
                refiner_lm="openrouter/openai/gpt-4.1-nano",
                max_refinements=3,
            ),
        )

        result_with_cache = optimize_anything(
            seed_candidate={"number": "50"},
            fitness_fn=fitness_fn_with_cache,
            objective="Guess the golden integer. off_by shows distance from target.",
            config=config_with_cache,
            cache_evaluation_storage="memory",
        )

        print(f"\n[Comparison] No cache: {call_counter_no_cache['count']} calls, With cache: {call_counter_with_cache['count']} calls")

        # With caching, we should have equal or fewer actual fitness calls
        assert call_counter_with_cache["count"] <= call_counter_no_cache["count"]


if __name__ == "__main__":
    print("Running refiner manual test...")

    call_counter = {"count": 0}
    fitness_fn = create_fitness_fn(call_counter)

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=5,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm="openrouter/openai/gpt-4.1-nano",
        ),
        refiner=RefinerConfig(
            refiner_lm="openrouter/openai/gpt-4.1-nano",
            max_refinements=2,
        ),
    )

    result = optimize_anything(
        seed_candidate={"number": "50"},
        fitness_fn=fitness_fn,
        objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0. Return ONLY an integer.",
        config=config,
        cache_evaluation_storage="memory",
    )

    print(f"\nBest candidate: {result.best_candidate}")
    print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")
    print(f"Total metric calls: {result.total_metric_calls}")
    print(f"Actual fitness_fn calls: {call_counter['count']}")
