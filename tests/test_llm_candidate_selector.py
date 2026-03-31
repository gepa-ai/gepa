# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

import pytest

from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.llm_candidate_selector import CandidateSelectionSignature, LLMCandidateSelector


@pytest.fixture
def mock_state():
    """Create a mock GEPAState with 3 candidates for testing."""
    seed_candidate = {"system_prompt": "You are a helpful assistant.", "task_instructions": "Answer questions."}
    base_valset_eval_output = ValsetEvaluation(
        outputs_by_val_id={0: "out1", 1: "out2", 2: "out3"},
        scores_by_val_id={0: 0.5, 1: 0.3, 2: 0.7},
        objective_scores_by_val_id=None,
    )
    state = GEPAState(seed_candidate, base_valset_eval_output, track_best_outputs=False)

    # Add two more candidates with different scores
    state.program_candidates.append(
        {"system_prompt": "You are an expert assistant.", "task_instructions": "Answer questions precisely."}
    )
    state.program_candidates.append(
        {"system_prompt": "You are a knowledgeable tutor.", "task_instructions": "Explain step by step."}
    )

    state.prog_candidate_val_subscores.append({0: 0.6, 1: 0.6, 2: 0.6})
    state.prog_candidate_val_subscores.append({0: 0.8, 1: 0.8, 2: 0.8})

    state.prog_candidate_objective_scores.append({})
    state.prog_candidate_objective_scores.append({})

    state.parent_program_for_candidate.append([0])
    state.parent_program_for_candidate.append([1])

    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)

    state.num_metric_calls_by_discovery.append(0)
    state.num_metric_calls_by_discovery.append(0)

    for i in range(len(state.pareto_front_valset)):
        state.program_at_pareto_front_valset[i] = {0, 1, 2}

    assert state.is_consistent()
    return state


# --- CandidateSelectionSignature tests ---


class TestCandidateSelectionSignature:
    def _make_candidates_info(self):
        return [
            {
                "real_idx": 0,
                "score": 0.5,
                "objective_scores": {},
                "texts": {"system_prompt": "You are helpful.", "task": "Answer."},
                "parents": [None],
            },
            {
                "real_idx": 2,
                "score": 0.8,
                "objective_scores": {"accuracy": 0.9, "fluency": 0.7},
                "texts": {"system_prompt": "You are expert.", "task": "Explain."},
                "parents": [1],
            },
        ]

    def test_prompt_renderer_best_mode(self):
        prompt = CandidateSelectionSignature.prompt_renderer(
            {"candidates": self._make_candidates_info(), "mode": "best", "num_candidates": 2}
        )
        assert "Candidate 0" in prompt
        assert "Candidate 1" in prompt
        assert "Aggregate Score: 0.5000" in prompt
        assert "Aggregate Score: 0.8000" in prompt
        assert "select the single best candidate" in prompt.lower()

    def test_prompt_renderer_diverse_mode(self):
        prompt = CandidateSelectionSignature.prompt_renderer(
            {"candidates": self._make_candidates_info(), "mode": "diverse", "num_candidates": 2}
        )
        assert "different approaches" in prompt
        assert "comma-separated" in prompt

    def test_prompt_includes_objective_scores(self):
        prompt = CandidateSelectionSignature.prompt_renderer(
            {"candidates": self._make_candidates_info(), "mode": "best", "num_candidates": 2}
        )
        assert "accuracy: 0.9000" in prompt
        assert "fluency: 0.7000" in prompt

    def test_prompt_includes_parent_info(self):
        prompt = CandidateSelectionSignature.prompt_renderer(
            {"candidates": self._make_candidates_info(), "mode": "best", "num_candidates": 2}
        )
        assert "derived from candidate(s) 1" in prompt

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            CandidateSelectionSignature.prompt_renderer(
                {"candidates": [], "mode": "invalid", "num_candidates": 0}
            )

    # --- Output extractor tests ---

    def test_output_extractor_single_digit(self):
        result = CandidateSelectionSignature.output_extractor("3")
        assert result["selected_index"] == "3"

    def test_output_extractor_comma_list(self):
        result = CandidateSelectionSignature.output_extractor("0, 2, 5")
        assert result["selected_indices"] == "0,2,5"

    def test_output_extractor_deduplicates(self):
        result = CandidateSelectionSignature.output_extractor("1, 1, 3")
        assert result["selected_indices"] == "1,3"

    def test_output_extractor_candidate_ref(self):
        """Parses 'Candidate N' patterns from explanatory text."""
        result = CandidateSelectionSignature.output_extractor(
            "Based on scores of 0.8000, Candidate 2 is the best choice."
        )
        assert result["selected_index"] == "2"

    def test_output_extractor_multiple_candidate_refs(self):
        result = CandidateSelectionSignature.output_extractor(
            "Candidate 0 and Candidate 3 represent different trade-offs."
        )
        assert result["selected_indices"] == "0,3"

    def test_output_extractor_no_numbers_raises(self):
        with pytest.raises(ValueError, match="No candidate numbers found"):
            CandidateSelectionSignature.output_extractor("no numbers here")

    def test_output_extractor_score_digits_not_extracted(self):
        """Bare integers from scores (e.g., '0.8000') are not picked up as candidate indices."""
        with pytest.raises(ValueError, match="No candidate numbers found"):
            CandidateSelectionSignature.output_extractor(
                "Based on the scores of 0.8000, the best approach is clearly superior"
            )


# --- LLMCandidateSelector tests ---


class TestLLMCandidateSelector:
    def test_protocol_compliance(self):
        """LLMCandidateSelector satisfies the CandidateSelector protocol."""
        lm = lambda prompt: "0"
        selector = LLMCandidateSelector(lm=lm, mode="best")
        assert isinstance(selector, CandidateSelector)

    def test_selects_index_from_llm(self, mock_state):
        """LLM returns display index 1, which maps to real program index based on score sorting."""
        lm = lambda prompt: "1"
        selector = LLMCandidateSelector(lm=lm, mode="best")
        selected = selector.select_candidate_idx(mock_state)
        # Candidates sorted by score desc: [2 (0.8), 1 (0.6), 0 (0.5)]
        # Display index 1 -> real index 1
        assert selected == 1

    def test_selects_top_candidate(self, mock_state):
        """LLM returns display index 0, which maps to the top-scoring candidate."""
        lm = lambda prompt: "0"
        selector = LLMCandidateSelector(lm=lm, mode="best")
        selected = selector.select_candidate_idx(mock_state)
        # Display index 0 -> real index 2 (highest score 0.8)
        assert selected == 2

    def test_single_candidate_skips_llm(self):
        """With only 1 candidate, LLM should not be called."""
        call_count = 0

        def counting_lm(prompt):
            nonlocal call_count
            call_count += 1
            return "0"

        seed_candidate = {"prompt": "test"}
        base_eval = ValsetEvaluation(
            outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None
        )
        state = GEPAState(seed_candidate, base_eval, track_best_outputs=False)

        selector = LLMCandidateSelector(lm=counting_lm, mode="best")
        result = selector.select_candidate_idx(state)
        assert result == 0
        assert call_count == 0

    def test_pre_filtering(self, mock_state):
        """With pre_filter_k=2, only top 2 candidates should be shown to LLM."""
        shown_candidates = []

        def inspecting_lm(prompt):
            shown_candidates.append(prompt)
            return "0"

        selector = LLMCandidateSelector(lm=inspecting_lm, mode="best", pre_filter_k=2)
        selector.select_candidate_idx(mock_state)

        prompt = shown_candidates[0]
        # Should have Candidate 0 and Candidate 1, but NOT Candidate 2
        assert "Candidate 0" in prompt
        assert "Candidate 1" in prompt
        assert "Candidate 2" not in prompt

    def test_fallback_on_api_error(self, mock_state):
        """When LLM raises an exception, falls back to fallback selector."""

        def failing_lm(prompt):
            raise RuntimeError("API error")

        selector = LLMCandidateSelector(lm=failing_lm, mode="best")
        # Should not raise, should fall back to CurrentBestCandidateSelector
        selected = selector.select_candidate_idx(mock_state)
        # CurrentBestCandidateSelector picks index 2 (highest score 0.8)
        assert selected == 2

    def test_fallback_on_bad_output(self, mock_state):
        """When LLM returns unparseable text, falls back."""
        lm = lambda prompt: "I cannot decide"
        selector = LLMCandidateSelector(lm=lm, mode="best")
        selected = selector.select_candidate_idx(mock_state)
        # Falls back to CurrentBestCandidateSelector -> index 2
        assert selected == 2

    def test_fallback_on_out_of_range_index(self, mock_state):
        """When LLM returns an out-of-range index, falls back."""
        lm = lambda prompt: "99"
        selector = LLMCandidateSelector(lm=lm, mode="best")
        selected = selector.select_candidate_idx(mock_state)
        # Falls back to CurrentBestCandidateSelector -> index 2
        assert selected == 2

    def test_custom_fallback(self, mock_state):
        """A custom fallback selector is used when LLM fails."""

        def failing_lm(prompt):
            raise RuntimeError("fail")

        class AlwaysZeroSelector:
            def select_candidate_idx(self, state):
                return 0

        selector = LLMCandidateSelector(lm=failing_lm, mode="best", fallback=AlwaysZeroSelector())
        assert selector.select_candidate_idx(mock_state) == 0

    def test_diverse_mode_samples_from_set(self, mock_state):
        """In diverse mode, LLM returns a set and selector samples from it."""
        lm = lambda prompt: "0, 2"
        rng = random.Random(42)
        selector = LLMCandidateSelector(lm=lm, mode="diverse", rng=rng)

        # Run multiple times to check sampling behavior
        results = set()
        for _ in range(50):
            results.add(selector.select_candidate_idx(mock_state))

        # Display indices 0 and 2 map to real indices based on score-sorted order
        # Candidates sorted by score desc: [2 (0.8), 1 (0.6), 0 (0.5)]
        # Display 0 -> real 2, display 2 -> real 0
        assert results.issubset({0, 2})
        assert len(results) == 2  # should have both with 50 samples

    def test_diverse_mode_empty_valid_indices_falls_back(self, mock_state):
        """If LLM returns indices all out of range in diverse mode, falls back."""
        lm = lambda prompt: "99, 100"
        selector = LLMCandidateSelector(lm=lm, mode="diverse")
        selected = selector.select_candidate_idx(mock_state)
        assert selected == 2  # fallback to ParetoCandidateSelector (default for diverse mode)

    def test_text_truncation(self, mock_state):
        """Long texts are truncated in the prompt."""
        long_text = "x" * 1000
        mock_state.program_candidates[2] = {"system_prompt": long_text, "task_instructions": "short"}

        shown_prompts = []

        def inspecting_lm(prompt):
            shown_prompts.append(prompt)
            return "0"

        selector = LLMCandidateSelector(lm=inspecting_lm, mode="best", max_text_chars=100)
        selector.select_candidate_idx(mock_state)

        prompt = shown_prompts[0]
        assert "...[truncated]" in prompt
        assert long_text not in prompt  # full text should NOT appear

    def test_deterministic_with_seed(self, mock_state):
        """Same rng seed produces same sampling sequence for pareto mode."""
        lm = lambda prompt: "0, 1, 2"

        results1 = []
        selector1 = LLMCandidateSelector(lm=lm, mode="diverse", rng=random.Random(42))
        for _ in range(10):
            results1.append(selector1.select_candidate_idx(mock_state))

        results2 = []
        selector2 = LLMCandidateSelector(lm=lm, mode="diverse", rng=random.Random(42))
        for _ in range(10):
            results2.append(selector2.select_candidate_idx(mock_state))

        assert results1 == results2

    def test_default_fallback_matches_mode(self):
        """Default fallback selector matches the mode: best->CurrentBest, diverse->Pareto."""
        lm = lambda prompt: "0"
        best_sel = LLMCandidateSelector(lm=lm, mode="best")
        diverse_sel = LLMCandidateSelector(lm=lm, mode="diverse")
        assert type(best_sel.fallback).__name__ == "CurrentBestCandidateSelector"
        assert type(diverse_sel.fallback).__name__ == "ParetoCandidateSelector"

    def test_only_two_modes(self):
        """Only 'best' and 'diverse' modes are supported."""
        lm = lambda prompt: "0"
        # Valid modes work
        LLMCandidateSelector(lm=lm, mode="best")
        LLMCandidateSelector(lm=lm, mode="diverse")
