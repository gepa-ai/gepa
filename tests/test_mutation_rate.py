# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for mutation_rate constraint injection."""

from __future__ import annotations

from unittest.mock import MagicMock

from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer


def _make_proposer(
    mutation_rate: float = 1.0,
    reflection_prompt_template: str = "Current: <curr_param>\nFeedback: <side_info>",
) -> ReflectiveMutationProposer:
    """Create a minimal proposer with a mock LM that captures the prompt."""
    mock_lm = MagicMock(return_value="```\nnew candidate text\n```")
    return ReflectiveMutationProposer(
        logger=MagicMock(),
        trainset=[],
        adapter=MagicMock(propose_new_texts=None),
        candidate_selector=MagicMock(),
        module_selector=MagicMock(),
        batch_sampler=MagicMock(),
        perfect_score=None,
        skip_perfect_score=False,
        experiment_tracker=MagicMock(),
        reflection_lm=mock_lm,
        reflection_prompt_template=reflection_prompt_template,
        custom_candidate_proposer=None,
        mutation_rate=mutation_rate,
    )


def _get_prompt(proposer: ReflectiveMutationProposer) -> str:
    """Extract the prompt string passed to the mock LM."""
    call_args = proposer.reflection_lm.call_args
    return call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")


class TestMutationRateDefault:
    def test_default_no_constraint(self) -> None:
        """mutation_rate=1.0 (default): no constraint appended."""
        proposer = _make_proposer(mutation_rate=1.0)
        proposer.propose_new_texts(
            {"component": "original text"},
            {"component": [{"input": "test", "output": "result"}]},
            ["component"],
        )
        assert "Mutation Constraint" not in _get_prompt(proposer)


class TestMutationRateConstraintText:
    def test_retention_percentage_in_prompt(self) -> None:
        """mutation_rate=0.2 appends constraint with 80% retention."""
        proposer = _make_proposer(mutation_rate=0.2)
        proposer.propose_new_texts(
            {"component": "original text"},
            {"component": [{"input": "test", "output": "result"}]},
            ["component"],
        )
        prompt = _get_prompt(proposer)
        assert "## Mutation Constraint" in prompt
        assert "80%" in prompt

    def test_mutation_rate_0_3_retention_70(self) -> None:
        """mutation_rate=0.3 appends constraint with 70% retention."""
        proposer = _make_proposer(mutation_rate=0.3)
        proposer.propose_new_texts(
            {"component": "original text"},
            {"component": [{"input": "test", "output": "result"}]},
            ["component"],
        )
        prompt = _get_prompt(proposer)
        assert "70%" in prompt

    def test_freeze_at_0_skips_lm(self) -> None:
        """mutation_rate=0.0 returns original text without calling LM, with metadata."""
        proposer = _make_proposer(mutation_rate=0.0)
        new_texts, prompts, raw_outputs = proposer.propose_new_texts(
            {"component": "original text"},
            {"component": [{"input": "test", "output": "result"}]},
            ["component"],
        )
        assert new_texts["component"] == "original text"
        proposer.reflection_lm.assert_not_called()
        assert "frozen" in prompts["component"]
        assert raw_outputs["component"] == "original text"

    def test_constraint_uses_generic_language(self) -> None:
        """Constraint text says 'parameter value', not 'prompt'."""
        proposer = _make_proposer(mutation_rate=0.2)
        proposer.propose_new_texts(
            {"component": "original text"},
            {"component": [{"input": "test", "output": "result"}]},
            ["component"],
        )
        prompt = _get_prompt(proposer)
        assert "parameter value" in prompt.lower()

    def test_constraint_appended_after_template(self) -> None:
        """Constraint appears after the template content, not before."""
        proposer = _make_proposer(mutation_rate=0.2)
        proposer.propose_new_texts(
            {"component": "original text"},
            {"component": [{"input": "test", "output": "result"}]},
            ["component"],
        )
        prompt = _get_prompt(proposer)
        template_end = prompt.find("Feedback:")
        constraint_start = prompt.find("## Mutation Constraint")
        assert template_end < constraint_start
