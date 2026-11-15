"""Tests for LLM-based seed candidate generation (seed_candidate=None)."""

from unittest.mock import MagicMock

import pytest

from gepa.optimize_anything import (
    _STR_CANDIDATE_KEY,
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    _build_seed_generation_prompt,
    _generate_seed_candidate,
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
        assert "## Sample Inputs" in prompt
        # Up to 3 examples
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "Example 3" in prompt
        # 4th example should NOT appear
        assert "Example 4" not in prompt

    def test_with_all_sections(self):
        prompt = _build_seed_generation_prompt(
            objective="Optimize kernels.",
            background="Target A100 GPUs.",
            dataset=[{"problem": "matmul"}],
        )
        assert "## Goal" in prompt
        assert "## Domain Context & Constraints" in prompt
        assert "## Sample Inputs" in prompt
        assert "## Output Format" in prompt

    def test_empty_dataset(self):
        prompt = _build_seed_generation_prompt(
            objective="Do stuff.",
            dataset=[],
        )
        # Empty dataset still shows the section but no examples
        assert "## Sample Inputs" in prompt


# ---------------------------------------------------------------------------
# _generate_seed_candidate
# ---------------------------------------------------------------------------


class TestGenerateSeedCandidate:
    def test_extracts_from_backtick_blocks(self):
        mock_lm = MagicMock(return_value="```\ngenerated candidate text\n```")
        result = _generate_seed_candidate(
            lm=mock_lm,
            objective="Test objective.",
        )
        assert result == {_STR_CANDIDATE_KEY: "generated candidate text"}
        mock_lm.assert_called_once()

    def test_extracts_with_language_specifier(self):
        mock_lm = MagicMock(return_value="```python\ndef solve():\n    return 42\n```")
        result = _generate_seed_candidate(
            lm=mock_lm,
            objective="Write code.",
        )
        assert result[_STR_CANDIDATE_KEY] == "def solve():\n    return 42"

    def test_passes_objective_and_background_to_prompt(self):
        mock_lm = MagicMock(return_value="```\nresult\n```")
        _generate_seed_candidate(
            lm=mock_lm,
            objective="My objective.",
            background="My background.",
        )
        prompt = mock_lm.call_args[0][0]
        assert "My objective." in prompt
        assert "My background." in prompt

    def test_passes_dataset_examples_to_prompt(self):
        mock_lm = MagicMock(return_value="```\nresult\n```")
        dataset = [{"input": "example1"}, {"input": "example2"}]
        _generate_seed_candidate(
            lm=mock_lm,
            objective="Solve.",
            dataset=dataset,
        )
        prompt = mock_lm.call_args[0][0]
        assert "example1" in prompt
        assert "example2" in prompt

    def test_logs_when_logger_provided(self):
        mock_lm = MagicMock(return_value="```\ncandidate\n```")
        mock_logger = MagicMock()
        _generate_seed_candidate(
            lm=mock_lm,
            objective="Goal.",
            logger=mock_logger,
        )
        assert mock_logger.log.call_count == 2
        # First call: "Generating initial seed candidate via LLM..."
        assert "Generating" in mock_logger.log.call_args_list[0][0][0]
        # Second call: "Generated seed candidate (N chars)"
        assert "Generated" in mock_logger.log.call_args_list[1][0][0]


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
        """Full flow: seed_candidate=None → LLM generates seed → optimization runs."""
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

        # The LLM was called at least once (for seed generation)
        assert len(calls) >= 1
        # First call should be the seed generation prompt
        assert "## Goal" in calls[0]
        assert "Generate a long candidate string." in calls[0]
        # Result should have a best_candidate (str because str_candidate_mode)
        assert isinstance(result.best_candidate, str)

    def test_full_flow_with_dataset(self):
        """seed_candidate=None with dataset includes examples in prompt."""
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

        # Seed generation prompt should include dataset examples
        assert "2+2" in calls[0]
        assert isinstance(result.best_candidate, str)
