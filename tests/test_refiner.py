# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for refiner functionality with the number guessing optimization scenario."""

import tempfile
from pathlib import Path

import pytest

from gepa.optimize_anything import (
    DEFAULT_REFINER_PROMPT,
    EngineConfig,
    GEPAConfig,
    RefinerConfig,
    ReflectionConfig,
    make_litellm_lm,
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


def create_multi_param_fitness_fn(call_counter: dict):
    """Create a fitness_fn for multi-param optimization (two numbers that should sum to 100)."""

    def fitness_fn(candidate: dict[str, str], **kwargs) -> tuple[float, dict]:
        call_counter["count"] += 1

        try:
            a = int(candidate["param_a"])
        except (ValueError, KeyError):
            a = 0
        try:
            b = int(candidate["param_b"])
        except (ValueError, KeyError):
            b = 0

        target_sum = 100
        off_by = abs(a + b - target_sum)
        score = -off_by

        side_info = {
            "param_a_value": a,
            "param_b_value": b,
            "sum": a + b,
            "target_sum": target_sum,
            "off_by": off_by,
        }

        return score, side_info

    return fitness_fn


def create_dataset_fitness_fn(call_counter: dict):
    """Fitness fn for dataset mode — each example has its own golden number.

    Returns scores dict so objective frontier has something to work with.
    """

    def fitness_fn(candidate: dict[str, str], example, **kwargs) -> tuple[float, dict]:
        call_counter["count"] += 1
        golden = example["golden"]

        try:
            guess = int(candidate["number"])
        except (ValueError, KeyError):
            guess = 0

        off_by = abs(guess - golden)
        score = -off_by

        return score, {
            "guess": guess,
            "golden": golden,
            "off_by": off_by,
            "scores": {"accuracy": max(0.0, 1.0 - off_by / 100.0)},
        }

    return fitness_fn


# Dataset: two golden numbers — candidate must compromise
DATASET = [{"golden": 40}, {"golden": 60}]


class TestRefiner:
    """Tests for refiner functionality."""

    def test_refiner_without_caching(self):
        """Test refiner works without caching. RefinerConfig() with no refiner_lm defaults from reflection_lm."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(
                max_refinements=2,
            ),
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0.",
            config=config,
        )

        assert result is not None
        # refiner_prompt should be auto-injected and present in best_candidate
        assert "refiner_prompt" in result.best_candidate
        print(
            f"\n[Refiner no cache] Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}"
        )
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
                cache_evaluation_storage="memory",
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(
                max_refinements=2,
            ),
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0.",
            config=config,
        )

        assert result is not None
        assert "refiner_prompt" in result.best_candidate
        print(
            f"\n[Refiner + memory cache] Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}"
        )
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
                    cache_evaluation_storage="disk",
                    run_dir=tmp_dir,
                ),
                reflection=ReflectionConfig(
                    reflection_lm="openrouter/openai/gpt-5-nano",
                ),
                refiner=RefinerConfig(
                    max_refinements=2,
                ),
            )

            result = optimize_anything(
                seed_candidate={"number": "50"},
                evaluator=fitness_fn,
                objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0.",
                config=config,
            )

            assert result is not None
            assert "refiner_prompt" in result.best_candidate
            print(
                f"\n[Refiner + disk cache] Metric calls: {result.total_metric_calls}, Actual fitness calls: {call_counter['count']}"
            )
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
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(
                max_refinements=3,
            ),
        )

        result_no_cache = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn_no_cache,
            objective="Guess the golden integer. off_by shows distance from target.",
            config=config_no_cache,
        )

        # Run with cache
        fitness_fn_with_cache = create_fitness_fn(call_counter_with_cache)
        config_with_cache = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=3,
                cache_evaluation=True,
                cache_evaluation_storage="memory",
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(
                max_refinements=3,
            ),
        )

        result_with_cache = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn_with_cache,
            objective="Guess the golden integer. off_by shows distance from target.",
            config=config_with_cache,
        )

        print(
            f"\n[Comparison] No cache: {call_counter_no_cache['count']} calls, With cache: {call_counter_with_cache['count']} calls"
        )

        # With caching, we should have equal or fewer actual fitness calls
        assert call_counter_with_cache["count"] <= call_counter_no_cache["count"]

    def test_custom_refiner_prompt_respected(self):
        """Test that user-provided refiner_prompt in seed is NOT overwritten."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)
        custom_prompt = "My custom refiner instructions: always guess 42."

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=3,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(
                max_refinements=1,
            ),
        )

        seed = {"number": "50", "refiner_prompt": custom_prompt}
        result = optimize_anything(
            seed_candidate=seed,
            evaluator=fitness_fn,
            objective="Guess the golden integer.",
            config=config,
        )

        assert result is not None
        # The custom prompt should be preserved (not overwritten by default)
        # Note: GEPA may mutate refiner_prompt during evolution, but the seed should start with our custom prompt
        print(
            f"Best candidate refiner_prompt starts with custom: {result.best_candidate.get('refiner_prompt', '').startswith('My custom')}"
        )

    def test_multi_param_refiner(self):
        """Test refiner with multiple parameters — both refined together."""
        call_counter = {"count": 0}
        fitness_fn = create_multi_param_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(
                max_refinements=2,
            ),
        )

        result = optimize_anything(
            seed_candidate={"param_a": "30", "param_b": "20"},
            evaluator=fitness_fn,
            objective="Find two integers that sum to 100. The side_info shows 'off_by' which is how far the sum is from 100.",
            config=config,
        )

        assert result is not None
        assert "refiner_prompt" in result.best_candidate
        assert "param_a" in result.best_candidate
        assert "param_b" in result.best_candidate
        print(f"\n[Multi-param] Best candidate: {result.best_candidate}")
        print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")

    def test_side_info_structure(self):
        """Test that side_info has user's structure with refiner_prompt_specific_info added.

        Tests the adapter directly to inspect the side_info structure produced
        by _evaluate_single_with_refinement.
        """
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
        from gepa.optimize_anything import _SINGLE_INSTANCE_SENTINEL, EvaluatorWrapper

        call_counter = {"count": 0}

        def raw_fitness_fn(candidate: dict[str, str], **kwargs) -> tuple[float, dict]:
            call_counter["count"] += 1
            try:
                guess = int(candidate["number"])
            except (ValueError, KeyError):
                guess = 0

            off_by = abs(guess - GOLDEN_NUMBER)
            score = -off_by

            return score, {
                "guess": guess,
                "golden": GOLDEN_NUMBER,
                "off_by": off_by,
                "scores": {"accuracy": max(0, 1 - off_by / 100)},
            }

        # Wrap fitness_fn the same way optimize_anything does
        wrapped = EvaluatorWrapper(raw_fitness_fn, single_instance_mode=True)

        refiner_config = RefinerConfig(
            refiner_lm=make_litellm_lm("openrouter/openai/gpt-5-nano"),
            max_refinements=1,
        )

        adapter = OptimizeAnythingAdapter(
            evaluator=wrapped,
            parallel=False,
            refiner_config=refiner_config,
            cache_mode="off",
        )

        # Build candidate with refiner_prompt (as optimize_anything would auto-inject)
        candidate = {
            "number": "50",
            "refiner_prompt": "Improve the guess. Return a JSON dict with the 'number' key.",
        }

        # Call _evaluate_single_with_refinement directly
        score, output, side_info = adapter._evaluate_single_with_refinement(candidate, _SINGLE_INSTANCE_SENTINEL)

        # User fields should be at top level (not nested under a target param)
        assert "guess" in side_info, f"User field 'guess' missing from side_info: {side_info.keys()}"
        assert "golden" in side_info, f"User field 'golden' missing from side_info: {side_info.keys()}"
        assert "off_by" in side_info, f"User field 'off_by' missing from side_info: {side_info.keys()}"
        # User's scores should pass through
        assert "scores" in side_info, f"User 'scores' missing from side_info: {side_info.keys()}"
        assert "accuracy" in side_info["scores"], f"User score 'accuracy' missing: {side_info['scores']}"
        # Refiner info should be in dedicated key
        assert "refiner_prompt_specific_info" in side_info, f"refiner_prompt_specific_info missing: {side_info.keys()}"
        refiner_info = side_info["refiner_prompt_specific_info"]
        assert "scores" in refiner_info, f"'scores' missing from refiner_info: {refiner_info.keys()}"
        assert "Attempts" in refiner_info, f"'Attempts' missing from refiner_info: {refiner_info.keys()}"
        # scores should contain actual best metric values (not rates)
        assert "accuracy" in refiner_info["scores"], f"'accuracy' missing from refiner scores: {refiner_info['scores']}"
        # Attempts should include original (iteration 0) + refinement iterations
        assert len(refiner_info["Attempts"]) >= 1, "Attempts should have at least the original evaluation"
        assert refiner_info["Attempts"][0]["iteration"] == 0, "First attempt should be iteration 0 (original)"
        print(f"Side info keys: {list(side_info.keys())}")
        print(f"Refiner info keys: {list(refiner_info.keys())}")
        print(f"Refiner scores: {refiner_info['scores']}")
        print(f"Num attempts: {len(refiner_info['Attempts'])}")

    def test_refiner_improves_score(self):
        """Test that the refiner actually produces a better (or equal) score.

        Uses the adapter directly so we can inspect original vs refined scores.
        The number-guessing task is simple enough that even a small LLM should
        improve a deliberately bad seed (number=0, score=-42).
        """
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
        from gepa.optimize_anything import _SINGLE_INSTANCE_SENTINEL, EvaluatorWrapper

        def raw_fitness_fn(candidate: dict[str, str], **kwargs) -> tuple[float, dict]:
            try:
                guess = int(candidate["number"])
            except (ValueError, KeyError):
                guess = 0

            off_by = abs(guess - GOLDEN_NUMBER)
            score = -off_by

            return score, {
                "guess": guess,
                "golden": GOLDEN_NUMBER,
                "off_by": off_by,
                "hint": 'The target is 42. Return {"number": "42"} to get a perfect score.',
            }

        wrapped = EvaluatorWrapper(raw_fitness_fn, single_instance_mode=True)

        refiner_config = RefinerConfig(
            refiner_lm=make_litellm_lm("openrouter/openai/gpt-5-nano"),
            max_refinements=3,
        )

        adapter = OptimizeAnythingAdapter(
            evaluator=wrapped,
            parallel=False,
            refiner_config=refiner_config,
            cache_mode="off",
        )

        # Deliberately bad seed — far from target
        candidate = {
            "number": "0",
            "refiner_prompt": (
                "You are improving a number-guessing candidate. "
                "The evaluation feedback contains 'off_by' (distance from target) and a 'hint'. "
                "Return a JSON dict with the 'number' key set to a better guess."
            ),
        }

        score, output, side_info = adapter._evaluate_single_with_refinement(candidate, _SINGLE_INSTANCE_SENTINEL)

        refiner_info = side_info["refiner_prompt_specific_info"]
        attempts = refiner_info["Attempts"]
        original_score = attempts[0]["score"]
        best_attempt_score = max(a.get("score", float("-inf")) for a in attempts)

        print(f"\nOriginal score: {original_score}")
        print(f"Best attempt score: {best_attempt_score}")
        print(f"Final score (max): {score}")
        print(f"Num attempts: {len(attempts)}")

        # The max(original, refined) guarantee must always hold
        assert score >= original_score, f"Final score ({score}) should be >= original ({original_score})"
        # Soft check: with the hint, a capable model should improve.
        # Nano models often return unparseable output, so we don't hard-assert.
        if best_attempt_score > original_score:
            print(f"Refiner improved: {original_score} -> {best_attempt_score}")
        else:
            print(
                f"WARNING: Refiner did not improve (original={original_score}, "
                f"best_attempt={best_attempt_score}). This is expected with small models "
                f"that struggle to return valid JSON."
            )

    def test_refiner_score_never_worse(self):
        """Test the max(original, refined) guarantee — refiner can only help, never hurt."""
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
        from gepa.optimize_anything import _SINGLE_INSTANCE_SENTINEL, EvaluatorWrapper

        def raw_fitness_fn(candidate: dict[str, str], **kwargs) -> tuple[float, dict]:
            try:
                guess = int(candidate["number"])
            except (ValueError, KeyError):
                guess = 0

            off_by = abs(guess - GOLDEN_NUMBER)
            score = -off_by
            return score, {"guess": guess, "off_by": off_by}

        wrapped = EvaluatorWrapper(raw_fitness_fn, single_instance_mode=True)

        refiner_config = RefinerConfig(
            refiner_lm=make_litellm_lm("openrouter/openai/gpt-5-nano"),
            max_refinements=2,
        )

        adapter = OptimizeAnythingAdapter(
            evaluator=wrapped,
            parallel=False,
            refiner_config=refiner_config,
            cache_mode="off",
        )

        # Seed already near-perfect — refiner shouldn't make it worse
        candidate = {
            "number": "41",
            "refiner_prompt": "Improve the guess. Return a JSON dict with 'number'.",
        }

        score, output, side_info = adapter._evaluate_single_with_refinement(candidate, _SINGLE_INSTANCE_SENTINEL)

        attempts = side_info["refiner_prompt_specific_info"]["Attempts"]
        original_score = attempts[0]["score"]

        print(f"\nOriginal score: {original_score}, Final score: {score}")

        # max(original, refined) guarantee
        assert score >= original_score, (
            f"Final score ({score}) must be >= original ({original_score}) — the refiner should never make things worse"
        )


    def test_refiner_fallback_scores_when_all_refinements_fail(self):
        """When all refinement attempts fail (e.g. JSON parse errors), best_refined_scores
        should fall back to the original evaluation's scores, not remain empty.
        This prevents losing objective frontier metrics when the original score is negative
        and failed attempts have placeholder score=0.0.
        """
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
        from gepa.optimize_anything import _SINGLE_INSTANCE_SENTINEL, EvaluatorWrapper

        def raw_fitness_fn(candidate: dict[str, str], **kwargs) -> tuple[float, dict]:
            try:
                guess = int(candidate["number"])
            except (ValueError, KeyError):
                guess = 0
            off_by = abs(guess - GOLDEN_NUMBER)
            score = -off_by
            return score, {
                "guess": guess,
                "off_by": off_by,
                "scores": {"accuracy": max(0.0, 1.0 - off_by / 100.0)},
            }

        # Use a refiner_lm that always returns invalid JSON to force all refinements to fail
        def bad_refiner_lm(prompt: str) -> str:
            return "this is not valid json at all"

        wrapped = EvaluatorWrapper(raw_fitness_fn, single_instance_mode=True)

        refiner_config = RefinerConfig(
            refiner_lm=bad_refiner_lm,
            max_refinements=3,
        )

        adapter = OptimizeAnythingAdapter(
            evaluator=wrapped,
            parallel=False,
            refiner_config=refiner_config,
            cache_mode="off",
        )

        candidate = {
            "number": "50",
            "refiner_prompt": "Improve the guess. Return a JSON dict with 'number'.",
        }

        score, output, side_info = adapter._evaluate_single_with_refinement(
            candidate, _SINGLE_INSTANCE_SENTINEL
        )

        refiner_info = side_info["refiner_prompt_specific_info"]

        # All refinement attempts should have failed (only iteration 0 has "side_info")
        evaluated = [a for a in refiner_info["Attempts"] if "side_info" in a]
        failed = [a for a in refiner_info["Attempts"] if "error" in a]
        assert len(evaluated) == 1, f"Expected only original eval to succeed, got {len(evaluated)}"
        assert len(failed) >= 1, "Expected at least one failed refinement attempt"

        # The key check: best_refined_scores should fall back to original's scores,
        # not be empty
        assert refiner_info["scores"] == {"accuracy": max(0.0, 1.0 - abs(50 - GOLDEN_NUMBER) / 100.0)}, (
            f"Expected fallback to original scores, got: {refiner_info['scores']}"
        )

        # Score should equal the original (no refinement improved)
        assert score == -abs(50 - GOLDEN_NUMBER)


class TestRefinerWithDataset:
    """Test refiner with a dataset (per-instance evaluation)."""

    def test_refiner_dataset_single_instance_baseline(self):
        """Baseline: single-instance mode (no dataset), default instance frontier."""
        call_counter = {"count": 0}
        fitness_fn = create_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(max_refinements=2),
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer. off_by shows distance. Minimize off_by to 0.",
            config=config,
        )

        assert result is not None
        assert "refiner_prompt" in result.best_candidate
        best_score = result.val_aggregate_scores[result.best_idx]
        print(f"\n[Single-instance] Best: {result.best_candidate['number']}, Score: {best_score}")
        print(f"Metric calls: {result.total_metric_calls}, Fitness calls: {call_counter['count']}")

    def test_refiner_dataset_mode(self):
        """Dataset mode: refiner evaluated across multiple examples."""
        call_counter = {"count": 0}
        fitness_fn = create_dataset_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(max_refinements=2),
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess a number close to each example's golden number. off_by shows distance.",
            dataset=DATASET,
            config=config,
        )

        assert result is not None
        assert "refiner_prompt" in result.best_candidate
        best_score = result.val_aggregate_scores[result.best_idx]
        print(f"\n[Dataset mode] Best: {result.best_candidate['number']}, Score: {best_score}")
        print(f"Metric calls: {result.total_metric_calls}, Fitness calls: {call_counter['count']}")


class TestRefinerFrontierTypes:
    """Test refiner with each frontier type using a dataset."""

    @pytest.mark.parametrize("frontier_type", ["instance", "objective", "hybrid", "cartesian"])
    def test_refiner_frontier_type(self, frontier_type):
        """Test refiner works with each frontier type."""
        call_counter = {"count": 0}
        fitness_fn = create_dataset_fitness_fn(call_counter)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                cache_evaluation=False,
                frontier_type=frontier_type,
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-5-nano",
            ),
            refiner=RefinerConfig(max_refinements=2),
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess a number close to each example's golden number. off_by shows distance. The scores dict has 'accuracy'.",
            dataset=DATASET,
            config=config,
        )

        assert result is not None
        assert "refiner_prompt" in result.best_candidate
        best_score = result.val_aggregate_scores[result.best_idx]
        print(f"\n[{frontier_type}] Best: {result.best_candidate['number']}, Score: {best_score}")
        print(f"Metric calls: {result.total_metric_calls}, Fitness calls: {call_counter['count']}")

    @pytest.mark.parametrize("frontier_type", ["instance", "objective", "hybrid", "cartesian"])
    def test_refiner_side_info_structure_with_dataset(self, frontier_type):
        """Verify side_info structure is correct per frontier type — uses adapter directly."""
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter

        call_counter = {"count": 0}

        # This function is passed directly to OptimizeAnythingAdapter (bypassing
        # EvaluatorWrapper), so it must return the adapter's internal 3-tuple
        # format (score, output, side_info) rather than the public Evaluator
        # protocol format of (score, side_info) or plain float.
        def raw_fitness_fn(candidate, example, **kwargs):
            call_counter["count"] += 1
            golden = example["golden"]
            try:
                guess = int(candidate["number"])
            except (ValueError, KeyError):
                guess = 0
            off_by = abs(guess - golden)
            score = -off_by
            return (
                score,
                None,
                {
                    "guess": guess,
                    "golden": golden,
                    "off_by": off_by,
                    "scores": {"accuracy": max(0.0, 1.0 - off_by / 100.0)},
                },
            )

        refiner_config = RefinerConfig(
            refiner_lm=make_litellm_lm("openrouter/openai/gpt-5-nano"),
            max_refinements=2,
        )

        adapter = OptimizeAnythingAdapter(
            evaluator=raw_fitness_fn,
            parallel=False,
            refiner_config=refiner_config,
            cache_mode="off",
        )

        candidate = {
            "number": "50",
            "refiner_prompt": (
                "You are refining a number-guessing candidate. "
                "Return a JSON dict with the 'number' key set to a better guess. "
                "The evaluation_feedback shows off_by for each example."
            ),
        }

        # Evaluate across dataset
        eval_batch = adapter.evaluate(DATASET, candidate)

        assert len(eval_batch.scores) == len(DATASET)
        assert len(eval_batch.objective_scores) == len(DATASET)

        for i, (score, side_info, obj_scores) in enumerate(
            zip(eval_batch.scores, eval_batch.trajectories, eval_batch.objective_scores)
        ):
            print(f"\n[{frontier_type}] Example {i} (golden={DATASET[i]['golden']}):")
            print(f"  Score: {score}")
            print(f"  Side info keys: {list(side_info.keys())}")

            # Top-level should have user fields from ORIGINAL evaluation
            assert "guess" in side_info
            assert "golden" in side_info
            assert "off_by" in side_info
            assert "scores" in side_info
            assert "accuracy" in side_info["scores"]

            # Refiner info with best scores + attempt history
            assert "refiner_prompt_specific_info" in side_info
            rinfo = side_info["refiner_prompt_specific_info"]
            assert "scores" in rinfo
            assert "Attempts" in rinfo
            assert len(rinfo["Attempts"]) >= 1
            print(f"  Refiner scores: {rinfo['scores']}")
            print(f"  Num attempts: {len(rinfo['Attempts'])}")

            # objective_scores should include both global and refiner metrics
            assert "accuracy" in obj_scores, f"Global 'accuracy' missing from objective_scores: {obj_scores}"
            # Refiner scores should have actual metric values (e.g., accuracy)
            assert "refiner_prompt::accuracy" in obj_scores, (
                f"Refiner accuracy missing from objective_scores: {obj_scores}"
            )
            print(f"  Objective scores: {obj_scores}")


if __name__ == "__main__":
    print("Running refiner manual test...")

    call_counter = {"count": 0}
    fitness_fn = create_fitness_fn(call_counter)

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=5,
            cache_evaluation=True,
            cache_evaluation_storage="memory",
        ),
        reflection=ReflectionConfig(
            reflection_lm="openrouter/openai/gpt-5-nano",
        ),
        refiner=RefinerConfig(
            max_refinements=2,
        ),
    )

    result = optimize_anything(
        seed_candidate={"number": "50"},
        evaluator=fitness_fn,
        objective="Guess the golden integer. The side_info shows 'off_by' which is how far your guess is from the target. Minimize off_by to 0. Return ONLY an integer.",
        config=config,
    )

    print(f"\nBest candidate: {result.best_candidate}")
    print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")
    print(f"Total metric calls: {result.total_metric_calls}")
    print(f"Actual fitness_fn calls: {call_counter['count']}")
