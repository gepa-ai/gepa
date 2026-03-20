"""Tests for the max_candidate_tokens feature across the GEPA optimization pipeline."""

from unittest.mock import Mock

from gepa.core.token_budget import (
    _WARNING_THRESHOLD,
    FALLBACK_TOKEN_COUNTER_MODEL,
    build_candidate_token_context,
    check_candidate_token_limit,
    count_candidate_tokens,
)

# ---------------------------------------------------------------------------
# Unit tests for core/token_budget.py
# ---------------------------------------------------------------------------


_MODEL = FALLBACK_TOKEN_COUNTER_MODEL


class TestCountCandidateTokens:
    def test_single_component(self):
        candidate = {"prompt": "Hello world"}
        tokens = count_candidate_tokens(candidate, token_counter_model=_MODEL)
        assert tokens > 0

    def test_multi_component(self):
        candidate = {"a": "Hello", "b": "World of tokens"}
        tokens = count_candidate_tokens(candidate, token_counter_model=_MODEL)
        single_a = count_candidate_tokens({"a": "Hello"}, token_counter_model=_MODEL)
        assert tokens > single_a

    def test_empty_candidate(self):
        candidate = {"prompt": ""}
        tokens = count_candidate_tokens(candidate, token_counter_model=_MODEL)
        assert tokens >= 0

    def test_long_text_proportional(self):
        short = count_candidate_tokens({"p": "word " * 10}, token_counter_model=_MODEL)
        long = count_candidate_tokens({"p": "word " * 100}, token_counter_model=_MODEL)
        assert long > short


class TestCheckCandidateTokenLimit:
    def test_within_limit(self):
        candidate = {"p": "short"}
        token_count, exceeds, warn = check_candidate_token_limit(candidate, 10000, token_counter_model=_MODEL)
        assert not exceeds
        assert not warn
        assert token_count > 0

    def test_exceeds_limit(self):
        candidate = {"p": "word " * 500}
        token_count, exceeds, warn = check_candidate_token_limit(candidate, 5, token_counter_model=_MODEL)
        assert exceeds
        assert warn
        assert token_count > 5

    def test_warning_threshold(self):
        candidate = {"p": "x"}
        token_count = count_candidate_tokens(candidate, token_counter_model=_MODEL)
        # Set limit so that token_count > 80% of limit but <= limit
        limit = int(token_count / _WARNING_THRESHOLD) - 1
        if limit < token_count:
            limit = token_count
        _, exceeds, warn = check_candidate_token_limit(candidate, limit, token_counter_model=_MODEL)
        assert not exceeds
        assert warn


class TestBuildCandidateTokenContext:
    def test_contains_limit_info(self):
        candidate = {"p": "hello world"}
        context = build_candidate_token_context(candidate, 1000, token_counter_model=_MODEL)
        assert "1000" in context
        assert "Token Limit" in context
        assert "MUST" in context

    def test_contains_current_tokens(self):
        candidate = {"p": "hello world"}
        tokens = count_candidate_tokens(candidate, token_counter_model=_MODEL)
        context = build_candidate_token_context(candidate, 1000, token_counter_model=_MODEL)
        assert str(tokens) in context


# ---------------------------------------------------------------------------
# Integration tests: max_candidate_tokens flows through optimize()
# ---------------------------------------------------------------------------


class TestMaxCandidateTokensOptimize:
    """Test max_candidate_tokens parameter in gepa.optimize()."""

    def _make_fixtures(self):
        mock_data = [
            {
                "input": "test_input",
                "answer": "test_answer",
                "additional_context": {"context": "test_context"},
            }
        ]
        task_lm = Mock()
        task_lm.return_value = "test response"
        return mock_data, task_lm

    def test_max_candidate_tokens_none_no_effect(self):
        """max_candidate_tokens=None should not change existing behavior."""
        from gepa import optimize

        mock_data, task_lm = self._make_fixtures()
        reflection_calls = []

        def mock_reflection_lm(prompt):
            reflection_calls.append(prompt)
            return "```\nimproved instructions\n```"

        result = optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=2,
            reflection_minibatch_size=1,
            max_candidate_tokens=None,
        )
        assert result is not None

    def test_max_candidate_tokens_injects_context_into_reflection(self):
        """When max_candidate_tokens is set, the reflection prompt should contain token limit context."""
        from gepa import optimize

        mock_data, task_lm = self._make_fixtures()
        reflection_calls = []

        def mock_reflection_lm(prompt):
            reflection_calls.append(prompt)
            return "```\nimproved instructions\n```"

        optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=2,
            reflection_minibatch_size=1,
            max_candidate_tokens=4096,
        )

        assert len(reflection_calls) > 0
        assert any("Token Limit" in str(call) for call in reflection_calls)
        assert any("4096" in str(call) for call in reflection_calls)

    def test_max_candidate_tokens_rejects_oversized_candidates(self):
        """Candidates exceeding max_candidate_tokens should be rejected (not added to result)."""
        from gepa import optimize

        mock_data, task_lm = self._make_fixtures()

        def mock_reflection_lm(prompt):
            return "```\n" + ("This is a very long text. " * 200) + "\n```"

        result = optimize(
            seed_candidate={"instructions": "hi"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=4,
            reflection_minibatch_size=1,
            max_candidate_tokens=5,
        )

        # Only the seed candidate (idx 0) should survive since all proposals
        # will be rejected for exceeding the limit
        assert result.best_idx == 0


class TestMaxCandidateTokensOptimizeAnything:
    """Test max_candidate_tokens in EngineConfig for optimize_anything()."""

    def test_max_candidate_tokens_in_engine_config(self):
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

        reflection_calls = []

        def mock_reflection_lm(prompt):
            reflection_calls.append(prompt)
            return "```\nimproved\n```"

        def evaluator(candidate):
            return 0.5

        config = GEPAConfig(
            engine=EngineConfig(max_metric_calls=2, max_candidate_tokens=2048),
            reflection=ReflectionConfig(reflection_lm=mock_reflection_lm),
        )

        result = optimize_anything(
            seed_candidate="initial text",
            evaluator=evaluator,
            config=config,
        )

        assert result is not None
        if reflection_calls:
            assert any("Token Limit" in str(call) for call in reflection_calls)
