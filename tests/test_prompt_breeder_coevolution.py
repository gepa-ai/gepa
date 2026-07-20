# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from pathlib import Path
from types import SimpleNamespace

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.prompt_breeder import (
    PromptBreederConfig,
    PromptBreederReflectionLM,
)


class QueueLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.total_cost = 2.5

    def __call__(self, prompt):
        self.calls.append(prompt)
        if not self.replies:
            raise AssertionError("QueueLM received more calls than expected")
        return self.replies.pop(0)


def reflective_feedback():
    return {
        "system_prompt": [
            {
                "Inputs": "He walked to the bank.",
                "Generated Outputs": "financial institution",
                "Feedback": "The context is insufficient; acknowledge both senses or ask for clarification.",
            }
        ]
    }


def breeder_config(operator: str, **overrides):
    return PromptBreederConfig(
        mutation_prompts=("parent mutation rule",),
        thinking_styles=("first principles",),
        operator_weights={"zero_order": 0.0, "hypermutation": 0.0, "lineage": 0.0, operator: 1.0},
        exploration_rate=0.0,
        seed=11,
        **overrides,
    )


def proposal_from(reflection_proposal, child, before=0.4, after=0.7):
    return CandidateProposal(
        candidate=child,
        parent_program_ids=[0],
        subsample_scores_before=[before],
        subsample_scores_after=[after],
        metadata=dict(reflection_proposal.metadata),
    )


class FakeState:
    def __init__(self, scores, program_candidates=None):
        self.scores = scores
        self.program_candidates = program_candidates or []

    def get_program_average_val_subset(self, idx):
        return self.scores[idx], 1


class ImprovementAdapter:
    propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        improved = "improved" in candidate["system_prompt"]
        score = 1.0 if improved else 0.0
        trajectories = (
            [
                {
                    "data": item,
                    "full_assistant_response": "ambiguous",
                    "feedback": "Mention ambiguity explicitly and ask for clarification when needed.",
                }
                for item in batch
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(
            outputs=[{"candidate": candidate["system_prompt"]} for _ in batch],
            scores=[score for _ in batch],
            trajectories=trajectories,
            objective_scores=None,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {
            component: [
                {
                    "Inputs": "He walked to the bank.",
                    "Generated Outputs": "financial institution",
                    "Feedback": "Call out both senses or ask a clarification question.",
                }
            ]
            for component in components_to_update
        }


def test_config_rejects_zero_total_operator_weight():
    with pytest.raises(ValueError, match="positive weight"):
        PromptBreederConfig(operator_weights={"zero_order": 0, "hypermutation": 0, "lineage": 0})


def test_operator_selection_is_deterministic_with_fixed_seed():
    config = PromptBreederConfig(
        operator_weights={"zero_order": 0.2, "hypermutation": 0.3, "lineage": 0.5},
        exploration_rate=1.0,
        seed=7,
    )
    left = PromptBreederReflectionLM(QueueLM([]), config)
    right = PromptBreederReflectionLM(QueueLM([]), config)

    assert [left._choose_operator() for _ in range(8)] == [right._choose_operator() for _ in range(8)]


def test_zero_order_mutation_uses_current_reflection_genome():
    lm = QueueLM(["```\nnew task instruction\n```"])
    breeder = PromptBreederReflectionLM(lm, breeder_config("zero_order"))

    reflected, successor = breeder.reflect(
        {"system_prompt": "old task instruction"},
        reflective_feedback(),
        ["system_prompt"],
    )

    trial = reflected.metadata["promptbreeder:trial_genomes"]["system_prompt"]
    assert reflected.new_texts == {"system_prompt": "new task instruction"}
    assert reflected.metadata["promptbreeder:operators"]["system_prompt"] == "zero_order"
    assert trial["mutation_prompt"] == "parent mutation rule"
    assert trial["thinking_style"] == "first principles"
    assert len(lm.calls) == 1
    assert successor is not breeder


def test_hypermutation_evolves_reflection_genes_before_task_prompt():
    lm = QueueLM(
        [
            "<mutation_prompt>classify ambiguity, test alternatives, then revise</mutation_prompt>"
            "<thinking_style>contrastive linguistic analysis</thinking_style>",
            "```\nnew task instruction\n```",
        ]
    )
    breeder = PromptBreederReflectionLM(lm, breeder_config("hypermutation"))

    reflected, successor = breeder.reflect(
        {"system_prompt": "old task instruction"},
        reflective_feedback(),
        ["system_prompt"],
    )

    trial = reflected.metadata["promptbreeder:trial_genomes"]["system_prompt"]
    assert len(lm.calls) == 2
    assert trial["mutation_prompt"] == "classify ambiguity, test alternatives, then revise"
    assert trial["thinking_style"] == "contrastive linguistic analysis"
    assert reflected.new_texts == {"system_prompt": "new task instruction"}
    assert successor is not breeder


def test_lineage_mutation_uses_accepted_ancestors():
    lm = QueueLM(
        [
            "<mutation_prompt>mutator one</mutation_prompt><thinking_style>style one</thinking_style>",
            "```\nchild one\n```",
            "<mutation_prompt>combined mutator</mutation_prompt><thinking_style>combined style</thinking_style>",
            "```\nchild two\n```",
        ]
    )
    breeder = PromptBreederReflectionLM(lm, breeder_config("lineage"))
    parent = {"system_prompt": "seed task"}

    first_reflection, breeder = breeder.reflect(parent, reflective_feedback(), ["system_prompt"])
    first_child = {"system_prompt": "child one"}
    breeder.on_proposal_accepted(
        proposal_from(first_reflection, first_child),
        1,
        FakeState({0: 0.4, 1: 0.7}, program_candidates=[parent, first_child]),
    )

    second_reflection, breeder = breeder.reflect(first_child, reflective_feedback(), ["system_prompt"])
    second_trial = second_reflection.metadata["promptbreeder:trial_genomes"]["system_prompt"]

    assert len(second_trial["parent_genome_ids"]) == 2
    assert second_trial["mutation_prompt"] == "combined mutator"
    assert second_trial["thinking_style"] == "combined style"
    assert "Recombine two successful reflection-process genomes" in lm.calls[2]


def test_rejected_child_does_not_promote_trial_genome():
    lm = QueueLM(["```\nrejected task instruction\n```"])
    breeder = PromptBreederReflectionLM(lm, breeder_config("zero_order"))
    reflected, breeder = breeder.reflect(
        {"system_prompt": "parent task instruction"}, reflective_feedback(), ["system_prompt"]
    )
    proposal = proposal_from(reflected, {"system_prompt": reflected.new_texts["system_prompt"]}, before=0.7, after=0.5)

    breeder.on_proposal_rejected(proposal, FakeState({0: 0.7}), "score regression")
    diagnostics = breeder.diagnostics()

    assert diagnostics["num_genomes"] == 1
    assert proposal.metadata["promptbreeder:status"] == "rejected"
    assert diagnostics["operator_stats"]["zero_order"]["rejects"] == 1


def test_accepted_child_assigns_trial_genome_to_new_candidate_index_and_credits_mutator():
    lm = QueueLM(["```\nchild task instruction\n```"])
    breeder = PromptBreederReflectionLM(lm, breeder_config("zero_order"))
    parent = {"system_prompt": "parent task instruction"}
    reflected, breeder = breeder.reflect(parent, reflective_feedback(), ["system_prompt"])
    child = {"system_prompt": reflected.new_texts["system_prompt"]}
    proposal = proposal_from(reflected, child)

    breeder.on_proposal_accepted(
        proposal,
        1,
        FakeState({0: 0.45, 1: 0.80}, program_candidates=[parent, child]),
    )
    diagnostics = breeder.diagnostics()
    state = breeder.get_state()

    assert proposal.metadata["promptbreeder:status"] == "accepted"
    assert diagnostics["num_genomes"] == 2
    assert state["candidate_index_genomes"][1]["system_prompt"].startswith("pb-")
    accepted = next(g for g in diagnostics["elite_genomes"] if g["generation"] == 1)
    assert accepted["creation_reward"] == pytest.approx(0.35)
    seed = next(g for g in diagnostics["elite_genomes"] if g["generation"] == 0)
    assert seed["offspring_accepts"] == 1
    assert diagnostics["operator_stats"]["zero_order"]["accepts"] == 1


def test_rejected_mutators_receive_rejection_stats_without_replacing_parent_genome():
    lm = QueueLM(["```\nrejected task instruction\n```"])
    breeder = PromptBreederReflectionLM(lm, breeder_config("zero_order"))
    parent = {"system_prompt": "parent task instruction"}
    reflected, breeder = breeder.reflect(parent, reflective_feedback(), ["system_prompt"])
    proposal = proposal_from(reflected, {"system_prompt": reflected.new_texts["system_prompt"]}, before=0.7, after=0.5)

    parent_key = breeder._candidate_key(parent)
    parent_mapping_before = breeder.get_state()["candidate_genomes"][parent_key].copy()
    breeder.on_proposal_rejected(proposal, FakeState({0: 0.7}), "score regression")
    diagnostics = breeder.diagnostics()
    parent_mapping_after = breeder.get_state()["candidate_genomes"][parent_key]

    seed = diagnostics["elite_genomes"][0]
    assert parent_mapping_after == parent_mapping_before
    assert seed["offspring_attempts"] == 1
    assert seed["offspring_accepts"] == 0
    assert seed["offspring_rejects"] == 1
    assert seed["offspring_reward"] == pytest.approx(-0.2)
    assert diagnostics["operator_stats"]["zero_order"]["attempts"] == 1
    assert diagnostics["operator_stats"]["zero_order"]["rejects"] == 1


def test_merge_created_candidate_inherits_fittest_parent_genome():
    lm = QueueLM(["```\nchild\n```"])
    breeder = PromptBreederReflectionLM(lm, breeder_config("zero_order"))
    seed = {"system_prompt": "seed"}
    reflection, breeder = breeder.reflect(seed, reflective_feedback(), ["system_prompt"])
    child = {"system_prompt": "child"}
    breeder.on_proposal_accepted(
        proposal_from(reflection, child),
        1,
        FakeState({0: 0.2, 1: 0.9}, program_candidates=[seed, child]),
    )

    merged = {"system_prompt": "merged task"}
    breeder.on_candidate_imported(merged, [seed, child], new_candidate_idx=2)
    state = breeder.get_state()
    child_gid = state["candidate_index_genomes"][1]["system_prompt"]
    merged_gid = state["candidate_index_genomes"][2]["system_prompt"]

    assert merged_gid == child_gid


def test_missing_reflective_feedback_skips_without_unnecessary_lm_calls():
    breeder = PromptBreederReflectionLM(QueueLM([]), breeder_config("zero_order"))
    reflected, _ = breeder.reflect({"system_prompt": "old task instruction"}, {}, ["system_prompt"])

    assert reflected.new_texts == {}


def test_reflection_cost_delegates_to_underlying_lm():
    assert PromptBreederReflectionLM(QueueLM([]), breeder_config("zero_order")).total_cost == 2.5


def test_breeder_state_round_trips_through_gepa_state_checkpoint(tmp_path):
    lm = QueueLM(["```\nchild\n```"])
    breeder = PromptBreederReflectionLM(lm, breeder_config("zero_order"))
    parent = {"system_prompt": "seed"}
    reflection, breeder = breeder.reflect(parent, reflective_feedback(), ["system_prompt"])
    child = {"system_prompt": "child"}
    breeder.on_proposal_accepted(
        proposal_from(reflection, child),
        1,
        FakeState({0: 0.3, 1: 0.6}, program_candidates=[parent, child]),
    )

    state = GEPAState(
        parent,
        ValsetEvaluation(outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None),
    )
    state.num_full_ds_evals = 1
    state.total_num_evals = 1
    state.proposer_state = {"reflection_strategy": breeder.get_state()}
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    state.save(str(run_dir))
    loaded = GEPAState.load(str(run_dir))
    restored = PromptBreederReflectionLM(QueueLM([]), breeder_config("zero_order"))
    restored.set_state(loaded.proposer_state["reflection_strategy"])

    assert restored.diagnostics() == breeder.diagnostics()


def test_old_checkpoints_without_breeder_state_remain_loadable(tmp_path):
    legacy_dir = tmp_path / "legacy_run"
    legacy_dir.mkdir()
    legacy_resource_path = Path(__file__).parent / "legacy_test_state.bin"
    (legacy_dir / "gepa_state.bin").write_bytes(legacy_resource_path.read_bytes())

    loaded = GEPAState.load(str(legacy_dir))

    assert loaded.proposer_state == {}


def test_prompt_breeder_disabled_behavior_remains_unchanged(tmp_path):
    run_dir = tmp_path / "plain_run"
    result = gepa.optimize(
        seed_candidate={"system_prompt": "base prompt"},
        trainset=[{"input": "x"}],
        valset=[{"input": "x"}],
        adapter=ImprovementAdapter(),
        reflection_lm=lambda prompt: "```\nimproved prompt\n```",
        max_metric_calls=4,
        reflection_minibatch_size=1,
        run_dir=str(run_dir),
        seed=0,
    )

    loaded = GEPAState.load(str(run_dir))
    assert result.best_candidate["system_prompt"] == "improved prompt"
    assert loaded.proposer_state == {}


def test_public_imports_and_both_entry_points_work(tmp_path):
    assert gepa.PromptBreederConfig is PromptBreederConfig
    assert callable(gepa.make_prompt_breeder_strategy)

    optimize_run = tmp_path / "optimize_run"
    optimize_result = gepa.optimize(
        seed_candidate={"system_prompt": "base prompt"},
        trainset=[{"input": "x"}],
        valset=[{"input": "x"}],
        adapter=ImprovementAdapter(),
        reflection_lm=lambda prompt: "```\nimproved prompt\n```",
        prompt_breeder_config=gepa.PromptBreederConfig(
            mutation_prompts=("mutate",),
            thinking_styles=("style",),
            operator_weights={"zero_order": 1.0, "hypermutation": 0.0, "lineage": 0.0},
            exploration_rate=0.0,
            seed=0,
        ),
        max_metric_calls=4,
        reflection_minibatch_size=1,
        run_dir=str(optimize_run),
        seed=0,
    )

    oa_result = optimize_anything(
        seed_candidate="base candidate",
        evaluator=lambda candidate: (1.0 if "improved" in candidate else 0.0, {"feedback": "add improved"}),
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=5, seed=0),
            reflection=ReflectionConfig(
                reflection_lm=lambda prompt: "```\nimproved candidate\n```",
                prompt_breeder_config=PromptBreederConfig(
                    mutation_prompts=("mutate",),
                    thinking_styles=("style",),
                    operator_weights={"zero_order": 1.0, "hypermutation": 0.0, "lineage": 0.0},
                    exploration_rate=0.0,
                    seed=0,
                ),
            ),
        ),
    )

    assert optimize_result.best_candidate["system_prompt"] == "improved prompt"
    assert oa_result.best_candidate == "improved candidate"


def test_prompt_breeder_configuration_conflicts_are_validated():
    with pytest.raises(ValueError, match="Cannot provide both prompt_breeder_config and reflection_strategy"):
        gepa.optimize(
            seed_candidate={"system_prompt": "base prompt"},
            trainset=[{"input": "x"}],
            valset=[{"input": "x"}],
            adapter=ImprovementAdapter(),
            reflection_lm=lambda prompt: "```\nimproved prompt\n```",
            reflection_strategy=SimpleNamespace(total_cost=0.0, reflect=lambda *args, **kwargs: None),
            prompt_breeder_config=PromptBreederConfig(),
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )

    with pytest.raises(ValueError, match="custom_candidate_proposer"):
        optimize_anything(
            seed_candidate="base candidate",
            evaluator=lambda candidate: (0.0, {"feedback": "f"}),
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=3),
                reflection=ReflectionConfig(
                    reflection_lm=lambda prompt: "```\nimproved candidate\n```",
                    prompt_breeder_config=PromptBreederConfig(),
                    custom_candidate_proposer=lambda c, d, comps: {"candidate": "x"},
                ),
            ),
        )
