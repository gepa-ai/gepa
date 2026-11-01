# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import copy
import json
from unittest.mock import Mock

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.proposer import PlaybookProposer


def test_appends_without_rewriting_or_mutating_inputs():
    candidate = {"system": "Keep this spacing.  \n", "tools": "Use tools carefully."}
    examples = {"system": [{"Input": "2 + 2", "Feedback": "Check arithmetic."}]}
    original = copy.deepcopy((candidate, examples))
    lm = Mock(return_value=json.dumps({"system": ["Verify the final answer."]}))

    proposed = PlaybookProposer(lm)(candidate, examples, ["system"])

    assert proposed == {"system": candidate["system"] + "\n\n- Verify the final answer."}
    assert (candidate, examples) == original
    prompt = lm.call_args.args[0]
    assert "Check arithmetic." in prompt
    assert "Use tools carefully." in prompt
    assert "Do not rewrite or remove existing content" in prompt


def test_addition_limit_applies_across_components():
    lm = Mock(return_value='{"system": ["Check units."], "tools": ["Validate output."]}')
    proposer = PlaybookProposer(lm, max_additions_per_step=2)
    candidate = {"system": "Be precise.", "tools": "Use tools."}
    examples = {name: [{"Feedback": "Needs improvement."}] for name in candidate}

    assert proposer(candidate, examples, list(candidate)) == {
        "system": "Be precise.\n\n- Check units.",
        "tools": "Use tools.\n\n- Validate output.",
    }
    lm.return_value = '{"system": ["One", "Two"], "tools": ["Three"]}'
    with pytest.raises(ValueError, match="max_additions_per_step"):
        proposer(candidate, examples, list(candidate))


@pytest.mark.parametrize("avoid_redundancy", [True, False])
def test_duplicate_bullets_follow_redundancy_option(avoid_redundancy):
    lm = Mock(return_value='{"system": ["Keep units.", "Check signs.", "Check signs."]}')
    candidate = {"system": "Be precise.\n- Keep units."}
    proposer = PlaybookProposer(lm, avoid_redundancy=avoid_redundancy)

    proposed = proposer(candidate, {"system": [{"Feedback": "Sign error."}]}, ["system"])

    additions = "- Check signs." if avoid_redundancy else "- Keep units.\n- Check signs.\n- Check signs."
    assert proposed == {"system": candidate["system"] + "\n\n" + additions}
    assert ("Avoid rules that duplicate" in lm.call_args.args[0]) == avoid_redundancy


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        '{"unselected": ["Do something else."]}',
        '{"system": "Replace the whole prompt."}',
        '{"system": [1]}',
        '{"system": ["   "]}',
    ],
)
def test_invalid_additions_cannot_change_candidate(response):
    candidate = {"system": "Preserve this."}
    proposer = PlaybookProposer(Mock(return_value=response))

    with pytest.raises(ValueError):
        proposer(candidate, {"system": [{"Feedback": "Try again."}]}, ["system"])

    assert candidate == {"system": "Preserve this."}


@pytest.mark.parametrize("components, examples", [([], {}), (["system"], {"system": []})])
def test_no_reflective_examples_does_not_call_lm(components, examples):
    lm = Mock()

    assert PlaybookProposer(lm)({"system": "Keep this."}, examples, components) == {}

    lm.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_addition_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        PlaybookProposer(Mock(), max_additions_per_step=limit)


@pytest.mark.parametrize("response", ["{}", '{"system": []}', '{"system": ["Keep this."]}'])
def test_no_additions_preserves_text_exactly(response):
    candidate = {"system": "Keep this.\n"}
    proposer = PlaybookProposer(Mock(return_value=response))

    assert proposer(candidate, {"system": [{"Feedback": "No change needed."}]}, ["system"]) == candidate


def test_empty_component_can_start_a_playbook():
    proposer = PlaybookProposer(Mock(return_value='{"system": ["Check units."]}'))

    assert proposer({"system": ""}, {"system": [{"Feedback": "Wrong units."}]}, ["system"]) == {
        "system": "- Check units."
    }


def test_optimizer_accepts_appended_candidate_and_preserves_other_components(tmp_path):
    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            score = float("Check units." in candidate["system"])
            return EvaluationBatch(
                outputs=[candidate["system"] for _ in batch],
                scores=[score for _ in batch],
                trajectories=[{"Feedback": "Check units."} for _ in batch] if capture_traces else None,
            )

        def make_reflective_dataset(self, candidate, eval_batch, components):
            return dict.fromkeys(components, eval_batch.trajectories)

    lm = Mock(return_value='{"system": ["Check units."]}')
    seed = {"system": "Solve the problem.", "tools": "Leave tool instructions intact."}

    result = gepa.optimize(
        seed_candidate=seed,
        trainset=[1, 2],
        valset=[1, 2],
        adapter=Adapter(),
        custom_candidate_proposer=PlaybookProposer(lm),
        max_metric_calls=10,
        stop_callbacks=lambda state: len(state.program_candidates) >= 2,
        run_dir=str(tmp_path),
    )

    assert result.candidates == [seed, {**seed, "system": "Solve the problem.\n\n- Check units."}]
    assert result.val_aggregate_scores == [0.0, 1.0]
    assert result.best_candidate == result.candidates[1]
    lm.assert_called_once()
