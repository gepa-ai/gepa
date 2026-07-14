# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for per-parent rejected-proposal memory (issue #379).

GEPA previously discarded all information about a rejected reflective
mutation the moment it was rejected. If the same parent was re-selected
later, the reflective proposer started from a blank slate and could
regenerate the exact same broken mutation indefinitely, burning the
evaluation budget on one dead end.

These tests cover the acceptance criteria from the issue:
  * GEPAState retains rejected proposals per parent.
  * The reflective proposer surfaces them in the reflection prompt under a
    clearly labeled section -- but only once a rejection has actually
    happened for that parent, never on the very first attempt.
  * The surfaced history is bounded (last 3 per parent) so hot parents
    don't bloat the prompt indefinitely.
  * Rejection memory survives a save/load round-trip, and a run_dir saved
    before this feature existed still loads cleanly.
"""

import os
import pickle
import tempfile

from gepa.core.state import GEPAState, RejectionRecord, ValsetEvaluation
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

SEED = "BASELINE_SOLUTION"
BROKEN_MUTATION = "import nonexistent_pkg\nBASELINE_SOLUTION"


def _make_seed_state() -> GEPAState:
    seed = {"instructions": "be helpful"}
    ve = ValsetEvaluation(outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None)
    return GEPAState(seed, ve)


def _evaluate(candidate, opt_state):
    """Deterministic stand-in for 'this code throws ImportError and scores 0'."""
    score = 1.0 if candidate == SEED else 0.0
    return score, {"score": score}


def _make_fake_reflection_lm(captured_prompts: list):
    """A zero-memory 'reflection LM' that always proposes the same broken edit."""

    def fake_reflection_lm(prompt):
        captured_prompts.append(prompt)
        return f"```\n{BROKEN_MUTATION}\n```"

    return fake_reflection_lm


class TestGEPAStateRejectionMemory:
    def test_fresh_state_has_empty_rejection_memory(self):
        state = _make_seed_state()
        assert state.rejected_proposals_by_parent == {}

    def test_save_load_round_trip_preserves_rejection_memory(self):
        state = _make_seed_state()
        state.rejected_proposals_by_parent.setdefault(0, []).append(
            RejectionRecord(
                proposed_text_diff={"instructions": "broken"},
                summarized_failure="nope",
                minibatch_score=0.0,
                iteration=1,
            )
        )

        with tempfile.TemporaryDirectory() as run_dir:
            state.save(run_dir)
            loaded = GEPAState.load(run_dir)

        assert loaded.rejected_proposals_by_parent[0][0].summarized_failure == "nope"
        assert loaded.rejected_proposals_by_parent[0][0].proposed_text_diff == {"instructions": "broken"}

    def test_loading_old_run_dir_without_field_defaults_to_empty(self):
        """A run_dir saved before issue #379's fix should still load cleanly."""
        state = _make_seed_state()

        with tempfile.TemporaryDirectory() as run_dir:
            state.save(run_dir)
            path = os.path.join(run_dir, "gepa_state.bin")
            with open(path, "rb") as f:
                data = pickle.load(f)
            del data["rejected_proposals_by_parent"]
            data["validation_schema_version"] = 5
            with open(path, "wb") as f:
                pickle.dump(data, f)

            loaded = GEPAState.load(run_dir)

        assert loaded.rejected_proposals_by_parent == {}
        assert loaded.validation_schema_version == GEPAState._VALIDATION_SCHEMA_VERSION


class TestReflectionPromptThreadsRejectionHistory:
    def test_no_history_on_first_attempt(self):
        """The very first proposal for a parent has nothing to recall yet."""
        captured_prompts: list = []
        optimize_anything(
            seed_candidate=SEED,
            evaluator=_evaluate,
            objective="test: no history on first attempt",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2, display_progress_bar=False),
                reflection=ReflectionConfig(reflection_lm=_make_fake_reflection_lm(captured_prompts)),
            ),
        )

        assert len(captured_prompts) >= 1
        assert "Earlier attempted edits" not in captured_prompts[0]

    def test_rejection_history_surfaces_on_next_attempt_with_same_parent(self):
        """Core repro for issue #379: same parent re-selected -> history must appear."""
        captured_prompts: list = []
        optimize_anything(
            seed_candidate=SEED,
            evaluator=_evaluate,
            objective="test: rejection history surfaces",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=10, display_progress_bar=False),
                reflection=ReflectionConfig(reflection_lm=_make_fake_reflection_lm(captured_prompts)),
            ),
        )

        assert len(captured_prompts) >= 2
        assert "Earlier attempted edits" not in captured_prompts[0]
        for later_prompt in captured_prompts[1:]:
            assert "Earlier attempted edits" in later_prompt
            assert "New subsample score 0.0 not better than old score 1.0" in later_prompt

    def test_rejection_history_is_bounded_to_last_three(self):
        """Issue #379 requirement: bound the list so hot parents don't bloat the prompt."""
        captured_prompts: list = []
        optimize_anything(
            seed_candidate=SEED,
            evaluator=_evaluate,
            objective="test: rejection history is bounded",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=20, display_progress_bar=False),
                reflection=ReflectionConfig(reflection_lm=_make_fake_reflection_lm(captured_prompts)),
            ),
        )

        last_prompt = captured_prompts[-1]
        # By the last iteration there have been far more than 3 rejections
        # for this parent; only the most recent 3 should be surfaced.
        assert last_prompt.count("#### iteration") == 3