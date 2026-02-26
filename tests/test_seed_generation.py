"""Tests for seedless prompt-evolution mode (seed_candidate=None)."""

import pytest

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    _build_seed_generation_prompt,
    optimize_anything,
)

# ---------------------------------------------------------------------------
# _build_seed_generation_prompt
# ---------------------------------------------------------------------------


class TestBuildSeedGenerationPrompt:
    def test_objective_only(self):
        prompt = _build_seed_generation_prompt(objective="Maximize throughput.")
        assert "## Goal" in prompt
        assert "Maximize throughput." in prompt
        assert "## Domain Context" not in prompt
        assert "## Sample Inputs" not in prompt
        assert "``` blocks" in prompt

    def test_with_background(self):
        prompt = _build_seed_generation_prompt(
            objective="Write fast code.",
            background="Use CUDA. Target H100 GPUs.",
        )
        assert "## Goal" in prompt
        assert "Write fast code." in prompt
        assert "## Domain Context & Constraints" in prompt
        assert "Use CUDA. Target H100 GPUs." in prompt

    def test_with_dataset(self):
        dataset = [{"input": "a"}, {"input": "b"}, {"input": "c"}, {"input": "d"}]
        prompt = _build_seed_generation_prompt(
            objective="Solve problems.",
            dataset=dataset,
        )
        assert "## Sample Inputs" not in prompt
        assert "Example 1" not in prompt
        assert "Example 2" not in prompt
        assert "Example 3" not in prompt

    def test_with_all_sections(self):
        prompt = _build_seed_generation_prompt(
            objective="Optimize kernels.",
            background="Target A100 GPUs.",
            dataset=[{"problem": "matmul"}],
        )
        assert "## Goal" in prompt
        assert "## Domain Context & Constraints" in prompt
        assert "## Sample Inputs" not in prompt
        assert "## Output Format" in prompt

    def test_empty_dataset(self):
        prompt = _build_seed_generation_prompt(
            objective="Do stuff.",
            dataset=[],
        )
        assert "## Sample Inputs" not in prompt


# ---------------------------------------------------------------------------
# optimize_anything(seed_candidate=None, ...) — validation
# ---------------------------------------------------------------------------


class TestOptimizeAnythingSeedNoneValidation:
    def test_error_without_objective(self):
        """seed_candidate=None requires objective."""
        with pytest.raises(ValueError, match="'objective' is required"):
            optimize_anything(
                seed_candidate=None,
                evaluator=lambda c: 1.0,
                config=GEPAConfig(engine=EngineConfig(max_metric_calls=1)),
            )

    def test_error_with_empty_objective(self):
        """seed_candidate=None with whitespace-only objective should error."""
        with pytest.raises(ValueError, match="'objective' is required"):
            optimize_anything(
                seed_candidate=None,
                evaluator=lambda c: 1.0,
                objective="   ",
                config=GEPAConfig(engine=EngineConfig(max_metric_calls=1)),
            )

    def test_error_without_reflection_lm(self):
        """seed_candidate=None with reflection_lm=None should error."""
        with pytest.raises(ValueError, match="reflection_lm is required"):
            optimize_anything(
                seed_candidate=None,
                evaluator=lambda c: 1.0,
                objective="Test objective.",
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=1),
                    reflection=ReflectionConfig(reflection_lm=None),
                ),
            )


# ---------------------------------------------------------------------------
# optimize_anything(seed_candidate=None, ...) — integration
# ---------------------------------------------------------------------------


class TestOptimizeAnythingSeedNoneIntegration:
    def test_full_flow_single_instance(self):
        """Full flow: seed_candidate=None → prompt-evolution mode runs end-to-end."""
        calls = []

        def mock_reflection_lm(prompt):
            calls.append(prompt)
            return "```\ngenerated initial candidate\n```"

        def evaluator(candidate: str) -> float:
            # Score based on length as a simple metric
            return min(len(candidate) / 100.0, 1.0)

        result = optimize_anything(
            seed_candidate=None,
            evaluator=evaluator,
            objective="Generate a long candidate string.",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2),
                reflection=ReflectionConfig(
                    reflection_lm=mock_reflection_lm,
                    reflection_minibatch_size=1,
                ),
            ),
        )

        # The LLM should be called at least once to materialize candidate text
        # from the evolved internal prompt.
        assert len(calls) >= 1
        # At least one call should include the seed-generation prompt scaffold.
        assert any("## Goal" in call for call in calls)
        assert any("Generate a long candidate string." in call for call in calls)
        # Result should have a best_candidate (str because str_candidate_mode)
        assert isinstance(result.best_candidate, str)

    def test_full_flow_with_dataset(self):
        """seed_candidate=None with dataset does not inject sample inputs into seed prompt."""
        calls = []

        def mock_reflection_lm(prompt):
            calls.append(prompt)
            return "```\nSolve the math problem step by step.\n```"

        dataset = [
            {"input": "2+2", "answer": "4"},
            {"input": "3*5", "answer": "15"},
        ]

        def evaluator(candidate: str, example) -> float:
            return 0.5

        result = optimize_anything(
            seed_candidate=None,
            evaluator=evaluator,
            objective="Generate a prompt for math problems.",
            dataset=dataset,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=3),
                reflection=ReflectionConfig(
                    reflection_lm=mock_reflection_lm,
                    reflection_minibatch_size=1,
                ),
            ),
        )

        # Seed-generation prompt scaffold should not include sample-input section.
        seed_prompt_calls = [call for call in calls if isinstance(call, str) and "## Goal" in call]
        assert seed_prompt_calls
        assert all("## Sample Inputs" not in call for call in seed_prompt_calls)
        # Per-example seedless mode has no single canonical best artifact.
        assert result.best_candidate is None
        assert isinstance(result.best_prompt, str)
        assert "## Goal" in result.best_prompt
        # May be sparse depending on optimization budget / which examples were evaluated.
        assert result.best_candidate_per_example is not None
        assert set(result.best_candidate_per_example).issubset({0, 1})

    def test_per_example_mode_can_return_sparse_map_with_tight_budget(self):
        calls = []

        def mock_reflection_lm(prompt):
            calls.append(prompt)
            return "```\nSolve the math problem step by step.\n```"

        dataset = [
            {"input": "2+2", "answer": "4"},
            {"input": "3*5", "answer": "15"},
        ]

        def evaluator(candidate: str, example) -> float:
            return 0.5

        result = optimize_anything(
            seed_candidate=None,
            evaluator=evaluator,
            objective="Generate a prompt for math problems.",
            dataset=dataset,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=1),
                reflection=ReflectionConfig(
                    reflection_lm=mock_reflection_lm,
                    reflection_minibatch_size=1,
                ),
            ),
        )

        # Tight budgets may stop before per-example proposals are evaluated.
        assert result.best_candidate is None
        assert isinstance(result.best_candidate_per_example, dict)
