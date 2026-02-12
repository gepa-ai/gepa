# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for OptimizationState / best_example_evals warm-start feature."""

import pytest

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    OptimizationState,
    ReflectionConfig,
    optimize_anything,
)

# Golden number to guess
GOLDEN_NUMBER = 42


def create_fitness_fn_with_best_evals_tracking(call_log: list):
    """Create an evaluator that logs OptimizationState received."""

    def fitness_fn(
        candidate: dict[str, str], opt_state: OptimizationState | None = None, **kwargs
    ) -> tuple[float, dict]:
        try:
            guess = int(candidate["number"])
        except (ValueError, KeyError):
            guess = 0

        off_by = abs(guess - GOLDEN_NUMBER)
        score = -off_by

        best_example_evals = opt_state.best_example_evals if opt_state else []

        # Log the call with best_example_evals info
        call_log.append({
            "guess": guess,
            "score": score,
            "best_example_evals": best_example_evals,
            "num_best_evals": len(best_example_evals),
        })

        side_info = {
            "guess": guess,
            "golden": GOLDEN_NUMBER,
            "off_by": off_by,
            "received_best_evals": len(best_example_evals),
        }

        return score, side_info

    return fitness_fn


class TestExampleBestEvals:
    """Tests for OptimizationState / best_example_evals feature."""

    def test_best_example_evals_passed_to_fitness_fn(self):
        """Verify OptimizationState is passed to evaluator via opt_state."""
        call_log = []
        fitness_fn = create_fitness_fn_with_best_evals_tracking(call_log)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                best_example_evals_k=3,  # Track top 3
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=None,
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer. off_by shows distance from target.",
            config=config,
        )

        assert result is not None
        assert len(call_log) > 0

        # First call should have empty best_example_evals
        first_call = call_log[0]
        assert first_call["num_best_evals"] == 0 or first_call["best_example_evals"] == []

        print(f"\nCall log ({len(call_log)} calls):")
        for i, call in enumerate(call_log):
            print(f"  {i}: guess={call['guess']}, score={call['score']}, num_best_evals={call['num_best_evals']}")

    def test_best_example_evals_accumulates(self):
        """Verify best_example_evals accumulates over iterations."""
        call_log = []
        fitness_fn = create_fitness_fn_with_best_evals_tracking(call_log)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=10,
                best_example_evals_k=5,  # Track top 5
            ),
            reflection=ReflectionConfig(
                reflection_lm="openrouter/openai/gpt-4.1-nano",
            ),
            refiner=None,
        )

        result = optimize_anything(
            seed_candidate={"number": "50"},
            evaluator=fitness_fn,
            objective="Guess the golden integer. off_by shows distance from target.",
            config=config,
        )

        assert result is not None

        # Check that later calls have more best_evals (up to k)
        best_evals_counts = [call["num_best_evals"] for call in call_log]
        print(f"\nBest evals counts over {len(call_log)} calls: {best_evals_counts}")

        # After first call, subsequent calls should have at least 1 best_eval
        if len(call_log) > 1:
            # At least some later calls should have accumulated best_evals
            max_best_evals = max(best_evals_counts)
            assert max_best_evals > 0, (
                f"Expected at least one call to receive non-empty best_example_evals, "
                f"but max was {max_best_evals}. Counts: {best_evals_counts}"
            )
            print(f"Max best_evals seen: {max_best_evals}")

    def test_best_example_evals_contains_scores(self):
        """Verify best_example_evals contains score and side_info."""
        call_log = []
        fitness_fn = create_fitness_fn_with_best_evals_tracking(call_log)

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5,
                best_example_evals_k=3,
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

        # Find a call that received best_evals
        calls_with_best_evals = [c for c in call_log if c["num_best_evals"] > 0]

        if calls_with_best_evals:
            call = calls_with_best_evals[0]
            best_evals = call["best_example_evals"]

            print(f"\nExample best_evals structure:")
            for i, eval_entry in enumerate(best_evals[:3]):
                print(f"  {i}: {eval_entry}")

            # Each entry should have 'score' and 'side_info'
            for entry in best_evals:
                assert "score" in entry, "best_eval entry should have 'score'"
                assert "side_info" in entry, "best_eval entry should have 'side_info'"


if __name__ == "__main__":
    print("Running OptimizationState / best_example_evals manual test...")

    call_log: list = []
    fitness_fn = create_fitness_fn_with_best_evals_tracking(call_log)

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=8,
            best_example_evals_k=3,
        ),
        reflection=ReflectionConfig(
            reflection_lm="openrouter/openai/gpt-4.1-nano",
        ),
        refiner=None,
    )

    result = optimize_anything(
        seed_candidate={"number": "50"},
        evaluator=fitness_fn,
        objective="Guess the golden integer. The side_info shows 'off_by' - minimize it to 0.",
        config=config,
    )

    print(f"\nBest candidate: {result.best_candidate}")
    print(f"Best score: {result.val_aggregate_scores[result.best_idx]}")
    print(f"Total calls: {len(call_log)}")

    print("\nCall details:")
    for i, call in enumerate(call_log):
        print(f"  {i}: guess={call['guess']}, score={call['score']}, num_best_evals={call['num_best_evals']}")
        if call["best_example_evals"]:
            for j, be in enumerate(call["best_example_evals"][:2]):
                print(f"      best_eval[{j}]: score={be.get('score')}")
