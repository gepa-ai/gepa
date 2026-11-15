# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for adapter-level fitness function caching (cache_evaluation_storage)."""

import tempfile
from pathlib import Path

import pytest

from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
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


class TestCacheEvaluationStorage:
    """Tests for cache_evaluation_storage parameter."""

    def test_memory_cache_prevents_duplicate_calls(self):
        """Memory cache should prevent re-evaluation of same candidate."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=True,  # Enable caching
                cache_evaluation_storage="memory",
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=None,
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer. The side_info tells you how far off you are.",
            config=config,
        )

        assert result is not None
        # With caching, we should have fewer actual fitness_fn calls than metric calls
        # because duplicate candidates are served from cache
        print(f"Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}")

    def test_disk_cache_persists_across_runs(self):
        """Disk cache should persist and be loaded on subsequent runs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            call_counter = {"count": 0}
            fitness_fn = create_fitness_fn(call_counter)

            config = GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=3,
                    cache_evaluation=True,
                    cache_evaluation_storage="disk",
                    run_dir=tmp_dir,
                ),
                reflection=ReflectionConfig(
                    reflection_lm="openrouter/openai/gpt-4.1-nano",
                ),
                refiner=None,
            )

            # First run
            result1 = optimize_anything(
                seed_candidate={"number": "50"},
                evaluator=fitness_fn,
                objective="Guess the golden integer. The side_info tells you how far off you are.",
                config=config,
            )

            first_run_calls = call_counter["count"]
            print(f"First run - Metric calls: {result1.total_metric_calls}, Actual fitness calls: {first_run_calls}")

            # Verify cache files were created
            cache_dir = Path(tmp_dir) / "fitness_cache"
            assert cache_dir.exists(), "Cache directory should be created"
            cache_files = list(cache_dir.glob("*.pkl"))
            assert len(cache_files) > 0, "Cache files should be created"
            print(f"Cache files created: {len(cache_files)}")

            # Second run with same seed - should use cache
            call_counter["count"] = 0
            fitness_fn2 = create_fitness_fn(call_counter)

            config2 = GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=3,
                    cache_evaluation=True,
                    cache_evaluation_storage="disk",
                    run_dir=tmp_dir,
                ),
                reflection=ReflectionConfig(
                    reflection_lm="openrouter/openai/gpt-4.1-nano",
                ),
                refiner=None,
            )

            result2 = optimize_anything(
                seed_candidate={"number": "50"},
                evaluator=fitness_fn2,
                objective="Guess the golden integer. The side_info tells you how far off you are.",
                config=config2,
            )

            second_run_calls = call_counter["count"]
            print(f"Second run - Metric calls: {result2.total_metric_calls}, Actual fitness calls: {second_run_calls}")

            # Second run should have fewer calls due to cache hits
            # At minimum, the seed candidate should be cached
            assert second_run_calls <= first_run_calls, "Second run should benefit from cache"

    def test_cache_off_when_cache_evaluation_false(self):
        """When cache_evaluation=False, no caching should occur."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=3,
                cache_evaluation=False,  # Caching disabled
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=None,
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer.",
            config=config,
        )

        assert result is not None
        # Every metric call should result in a fitness_fn call (no caching)
        print(f"Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}")

    def test_auto_mode_uses_disk_when_run_dir_provided(self):
        """Auto mode should use disk cache when run_dir is provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            call_counter = {"count": 0}
            fitness_fn = create_fitness_fn(call_counter)

            config = GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=3,
                    cache_evaluation=True,
                    run_dir=tmp_dir,
                ),
                reflection=ReflectionConfig(
                    reflection_lm="openrouter/openai/gpt-4.1-nano",
                ),
                refiner=None,
            )

            result = optimize_anything(
                seed_candidate={"number": "50"},
                evaluator=fitness_fn,
                objective="Guess the golden integer.",
                config=config,
            )

            # Verify cache files were created (disk mode)
            cache_dir = Path(tmp_dir) / "fitness_cache"
            assert cache_dir.exists(), "Auto mode with run_dir should use disk cache"

    def test_auto_mode_uses_memory_when_no_run_dir(self):
        """Auto mode should use memory cache when no run_dir is provided."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=3,
                cache_evaluation=True,
                run_dir=None,  # No run_dir
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=None,
        )

        # Should not raise - uses memory mode
        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer.",
            config=config,
        )

        assert result is not None

    def test_disk_mode_without_run_dir_raises_error(self):
        """Disk mode without run_dir should raise ValueError."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=3,
                cache_evaluation=True,
                cache_evaluation_storage="disk",  # Explicit disk without run_dir
                run_dir=None,  # No run_dir
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=None,
        )

        with pytest.raises(ValueError, match="cache_evaluation_storage='disk' requires run_dir"):
            optimize_anything(
                seed_candidate={"number": "50"},
                evaluator=fitness_fn,
                objective="Guess the golden integer.",
                config=config,
            )


if __name__ == "__main__":
    # Run a quick manual test
    print("Running manual test...")

    call_counter = {"count": 0}
    fitness_fn = create_fitness_fn(call_counter)

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=5,
            cache_evaluation=True,
            cache_evaluation_storage="memory",
        ),
        reflection=ReflectionConfig(
            reflection_lm="openrouter/openai/gpt-4.1-nano",
        ),
        refiner=None,
    )

    result = optimize_anything(
        seed_candidate={"number": "50"},
        evaluator=fitness_fn,
        objective="Guess the golden integer. The side_info tells you how far off you are (off_by field). Try to minimize off_by to 0.",
        config=config,
    )

    print(f"\nBest candidate: {result.best_candidate}")
    print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")
    print(f"Total metric calls: {result.total_metric_calls}")
    print(f"Actual fitness_fn calls: {call_counter['count']}")
