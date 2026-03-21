# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for proposal_mode='edit' end-to-end through optimize_anything and api."""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from gepa.strategies.instruction_proposal import InstructionEditSignature, InstructionProposalSignature


# ---------------------------------------------------------------------------
# _apply_edits_with_retry — shared retry logic
# ---------------------------------------------------------------------------


class TestApplyEditsWithRetry:
    def test_successful_edit(self):
        """Exact match edit returns the edited instruction."""
        lm = MagicMock()  # not called — no retry needed
        result, raw = InstructionEditSignature._apply_edits_with_retry(
            original_instruction="Hello world. Foo bar.",
            lm_out="<<<<<<< SEARCH\nFoo bar\n=======\nBaz qux\n>>>>>>> REPLACE",
            lm=lm,
        )
        assert result["new_instruction"] == "Hello world. Baz qux."
        lm.assert_not_called()

    def test_retry_on_failed_match(self):
        """When a SEARCH block fails, the LM is retried with error feedback."""
        call_count = [0]

        def mock_lm(prompt):
            call_count[0] += 1
            # Retry call: return a corrected block
            return "<<<<<<< SEARCH\nworld\n=======\nuniverse\n>>>>>>> REPLACE"

        result, raw = InstructionEditSignature._apply_edits_with_retry(
            original_instruction="Hello world.",
            lm_out="<<<<<<< SEARCH\nno match\n=======\nreplacement\n>>>>>>> REPLACE",
            lm=mock_lm,
            max_retries=1,
        )
        # LM called once for the retry
        assert call_count[0] == 1
        assert result["new_instruction"] == "Hello universe."

    def test_fallback_to_backtick_extraction(self):
        """When no SEARCH/REPLACE blocks are found, falls back to backtick extraction."""
        lm = MagicMock()
        result, raw = InstructionEditSignature._apply_edits_with_retry(
            original_instruction="old instruction",
            lm_out="```\nnew instruction from backticks\n```",
            lm=lm,
        )
        assert result["new_instruction"] == "new instruction from backticks"
        lm.assert_not_called()

    def test_multiple_edits(self):
        """Multiple SEARCH/REPLACE blocks are applied in sequence."""
        lm = MagicMock()
        lm_out = (
            "<<<<<<< SEARCH\nAAA\n=======\nXXX\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nCCC\n=======\nZZZ\n>>>>>>> REPLACE"
        )
        result, raw = InstructionEditSignature._apply_edits_with_retry(
            original_instruction="AAA\nBBB\nCCC",
            lm_out=lm_out,
            lm=lm,
        )
        assert result["new_instruction"] == "XXX\nBBB\nZZZ"


# ---------------------------------------------------------------------------
# run_with_metadata — routes through edit pipeline
# ---------------------------------------------------------------------------


class TestRunWithMetadata:
    def test_edit_mode_run_with_metadata(self):
        """run_with_metadata uses SEARCH/REPLACE parsing, not just backtick extraction."""
        def mock_lm(prompt):
            return "<<<<<<< SEARCH\nold text\n=======\nnew text\n>>>>>>> REPLACE"

        result, prompt, raw = InstructionEditSignature.run_with_metadata(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": "Hello old text. Goodbye.",
                "dataset_with_feedback": [{"Input": "x", "Output": "y"}],
            },
        )
        assert result["new_instruction"] == "Hello new text. Goodbye."
        assert "SEARCH" not in result["new_instruction"]

    def test_run_with_metadata_fallback(self):
        """run_with_metadata falls back to backtick extraction when no edit blocks found."""
        def mock_lm(prompt):
            return "```\ncompletely rewritten instruction\n```"

        result, prompt, raw = InstructionEditSignature.run_with_metadata(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": "old instruction",
                "dataset_with_feedback": [{"Input": "x", "Output": "y"}],
            },
        )
        assert result["new_instruction"] == "completely rewritten instruction"

    def test_rewrite_mode_run_with_metadata_unchanged(self):
        """InstructionProposalSignature.run_with_metadata still does backtick extraction."""
        def mock_lm(prompt):
            return "```\nnew prompt text\n```"

        result, prompt, raw = InstructionProposalSignature.run_with_metadata(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": "old prompt",
                "dataset_with_feedback": [{"Input": "x", "Output": "y"}],
            },
        )
        assert result["new_instruction"] == "new prompt text"

    def test_run_with_metadata_multimodal(self):
        """run_with_metadata works with multimodal prompts (images in dataset)."""
        from gepa.image import Image

        # Create a mock image
        mock_img = MagicMock(spec=Image)
        mock_img.to_openai_content_part.return_value = {"type": "image_url", "image_url": {"url": "data:..."}}

        def mock_lm(prompt):
            # prompt is a list of message dicts (multimodal)
            assert isinstance(prompt, list)
            return "<<<<<<< SEARCH\nold rule\n=======\nnew rule\n>>>>>>> REPLACE"

        result, prompt, raw = InstructionEditSignature.run_with_metadata(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": "Follow old rule carefully.",
                "dataset_with_feedback": [{"Input": "x", "Output": "y", "image": mock_img}],
            },
        )
        # Should still parse SEARCH/REPLACE correctly
        assert result["new_instruction"] == "Follow new rule carefully."
        # Prompt should be multimodal (list)
        assert isinstance(prompt, list)


# ---------------------------------------------------------------------------
# run() uses shared method (no duplication)
# ---------------------------------------------------------------------------


class TestRunUsesSharedMethod:
    def test_run_exact_edit(self):
        """run() applies edits correctly via the shared method."""
        def mock_lm(prompt):
            return "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE"

        result = InstructionEditSignature.run(
            lm=mock_lm,
            input_dict={
                "current_instruction_doc": "Hello foo world.",
                "dataset_with_feedback": [{"Input": "x", "Output": "y"}],
            },
        )
        assert result["new_instruction"] == "Hello bar world."

    def test_run_and_run_with_metadata_produce_same_result(self):
        """run() and run_with_metadata() produce the same new_instruction."""
        call_count = [0]

        def mock_lm(prompt):
            call_count[0] += 1
            return "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"

        input_dict = {
            "current_instruction_doc": "Replace old text here.",
            "dataset_with_feedback": [{"Input": "x", "Output": "y"}],
        }

        result_run = InstructionEditSignature.run(lm=mock_lm, input_dict=input_dict)
        result_meta, _, _ = InstructionEditSignature.run_with_metadata(lm=mock_lm, input_dict=input_dict)

        assert result_run["new_instruction"] == result_meta["new_instruction"]
        assert result_run["new_instruction"] == "Replace new text here."


# ---------------------------------------------------------------------------
# _build_reflection_prompt_template — mode-dependent output format
# ---------------------------------------------------------------------------


class TestBuildReflectionPromptTemplate:
    def test_rewrite_mode_has_backtick_instructions(self):
        from gepa.optimize_anything import _build_reflection_prompt_template

        template = _build_reflection_prompt_template(
            objective="test", proposal_mode="rewrite"
        )
        assert "```" in template
        assert "SEARCH" not in template
        assert "REPLACE" not in template

    def test_edit_mode_has_search_replace_instructions(self):
        from gepa.optimize_anything import _build_reflection_prompt_template

        template = _build_reflection_prompt_template(
            objective="test", proposal_mode="edit"
        )
        assert "<<<<<<< SEARCH" in template
        assert ">>>>>>> REPLACE" in template
        assert "Do NOT rewrite the component from scratch" in template

    def test_default_is_rewrite(self):
        from gepa.optimize_anything import _build_reflection_prompt_template

        template = _build_reflection_prompt_template(objective="test")
        assert "```" in template
        assert "SEARCH" not in template


# ---------------------------------------------------------------------------
# api.py warning — edit mode with custom template
# ---------------------------------------------------------------------------


class TestApiEditModeWarning:
    def test_edit_mode_with_custom_template_warns(self):
        """gepa.optimize warns and falls back to rewrite when edit + custom template."""
        import gepa
        from gepa.core.adapter import EvaluationBatch

        class DummyAdapter:
            propose_new_texts = None

            def evaluate(self, batch, candidate, capture_traces=False):
                return EvaluationBatch(outputs=[0.5] * len(batch), scores=[0.5] * len(batch))

            def make_reflective_dataset(self, candidate, eval_batch, components):
                return {c: [] for c in components}

        with patch("gepa.api.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_state = MagicMock()
            mock_state.program_candidates = [{"p": "x"}]
            mock_state.parent_program_for_candidate = [[None]]
            mock_state.prog_candidate_val_subscores = [{}]
            mock_state.program_at_pareto_front_valset = {}
            mock_state.program_full_scores_val_set = [0.5]
            mock_state.num_metric_calls_by_discovery = [0]
            mock_state.total_num_evals = 1
            mock_state.num_full_ds_evals = 1
            mock_state.best_outputs_valset = None
            mock_state.objective_pareto_front = {}
            mock_state.program_at_pareto_front_objectives = {}
            mock_engine.run.return_value = mock_state
            mock_cls.return_value = mock_engine

            with pytest.warns(UserWarning, match="rewrite mode"):
                gepa.optimize(
                    seed_candidate={"p": "x"},
                    trainset=[{"q": "a"}],
                    adapter=DummyAdapter(),
                    reflection_lm=MagicMock(return_value="```\nx\n```"),
                    max_metric_calls=3,
                    proposal_mode="edit",
                    reflection_prompt_template="custom template <curr_param> <side_info>",
                )

    def test_edit_mode_without_custom_template_no_warning(self):
        """gepa.optimize with edit mode and no custom template does NOT warn."""
        import gepa
        from gepa.core.adapter import EvaluationBatch

        class DummyAdapter:
            propose_new_texts = None

            def evaluate(self, batch, candidate, capture_traces=False):
                return EvaluationBatch(outputs=[0.5] * len(batch), scores=[0.5] * len(batch))

            def make_reflective_dataset(self, candidate, eval_batch, components):
                return {c: [] for c in components}

        with patch("gepa.api.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_state = MagicMock()
            mock_state.program_candidates = [{"p": "x"}]
            mock_state.parent_program_for_candidate = [[None]]
            mock_state.prog_candidate_val_subscores = [{}]
            mock_state.program_at_pareto_front_valset = {}
            mock_state.program_full_scores_val_set = [0.5]
            mock_state.num_metric_calls_by_discovery = [0]
            mock_state.total_num_evals = 1
            mock_state.num_full_ds_evals = 1
            mock_state.best_outputs_valset = None
            mock_state.objective_pareto_front = {}
            mock_state.program_at_pareto_front_objectives = {}
            mock_engine.run.return_value = mock_state
            mock_cls.return_value = mock_engine

            # Should not emit UserWarning about rewrite mode
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                gepa.optimize(
                    seed_candidate={"p": "x"},
                    trainset=[{"q": "a"}],
                    adapter=DummyAdapter(),
                    reflection_lm=MagicMock(return_value="```\nx\n```"),
                    max_metric_calls=3,
                    proposal_mode="edit",
                )
