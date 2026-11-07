import random

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.logging.logger import StdOutLogger
from gepa.proposer.merge import (
    MergeProposer,
    find_common_ancestor_pair,
    sample_and_attempt_merge_programs_by_common_predictors,
)


@pytest.mark.parametrize("ancestor_score", [-3.0, -0.5, 0.0, 1.0])
def test_merge_combines_complementary_descendants_with_finite_rewards(ancestor_score):
    candidates = [
        {"planner": "old plan", "solver": "old solution"},
        {"planner": "new plan", "solver": "old solution"},
        {"planner": "old plan", "solver": "new solution"},
    ]

    result = sample_and_attempt_merge_programs_by_common_predictors(
        agg_scores=[ancestor_score, ancestor_score + 1, ancestor_score + 2],
        rng=random.Random(0),
        merge_candidates=[1, 2],
        merges_performed=([], []),
        program_candidates=candidates,
        parent_program_for_candidate=[[None], [0], [0]],
    )

    assert result == ({"planner": "new plan", "solver": "new solution"}, 1, 2, 0)


@pytest.mark.parametrize("ancestor_scores", [(-2.0, -1.0), (-1.0, 1.0), (-1.0, 2.0)])
def test_negative_weights_keep_all_eligible_common_ancestors_reachable(ancestor_scores):
    candidates = [
        {"planner": "initial plan", "solver": "old solution"},
        {"planner": "old plan", "solver": "old solution"},
        {"planner": "new plan", "solver": "old solution"},
        {"planner": "old plan", "solver": "new solution"},
    ]
    rng = random.Random(42)
    observed_ancestors = set()

    for _ in range(64):
        result = find_common_ancestor_pair(
            rng=rng,
            parent_list=[[None], [0], [1], [1]],
            program_indexes=[2, 3],
            merges_performed=([], []),
            agg_scores=[*ancestor_scores, 3.0, 3.0],
            program_candidates=candidates,
        )
        assert result is not None
        assert result[:2] == (2, 3)
        observed_ancestors.add(result[2])

    assert observed_ancestors == {0, 1}


@pytest.mark.parametrize("ancestor_score", [-3.0, 0.0, 1.0])
def test_merge_proposer_evaluates_complementary_pareto_candidates(ancestor_score):
    candidates = [
        {"planner": "old plan", "solver": "old solution"},
        {"planner": "new plan", "solver": "old solution"},
        {"planner": "old plan", "solver": "new solution"},
    ]

    def evaluation(scores):
        return ValsetEvaluation(
            outputs_by_val_id={0: "output-0", 1: "output-1"},
            scores_by_val_id=dict(enumerate(scores)),
            objective_scores_by_val_id=None,
        )

    state = GEPAState(candidates[0], evaluation([ancestor_score, ancestor_score]))
    for idx, scores in enumerate(
        [[ancestor_score + 2, ancestor_score + 1], [ancestor_score + 1, ancestor_score + 2]],
        start=1,
    ):
        state.update_state_with_new_program(
            [0], candidates[idx], evaluation(scores), None, 2, iteration_id=f"child-{idx}"
        )
    assert state.get_pareto_front_mapping() == {0: {1}, 1: {2}}
    state.total_num_evals = 0
    state.full_program_trace.append({})
    evaluated_candidates = []

    def evaluate(batch, candidate):
        evaluated_candidates.append(dict(candidate))
        return list(batch), [ancestor_score + 3] * len(batch), None

    proposer = MergeProposer(
        logger=StdOutLogger(),
        valset=ListDataLoader(["example-0", "example-1"]),
        evaluator=evaluate,
        use_merge=True,
        max_merge_invocations=1,
        val_overlap_floor=2,
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.schedule_if_needed()

    proposal = proposer.propose(state)

    merged = {"planner": "new plan", "solver": "new solution"}
    assert proposal is not None
    assert proposal.candidate == merged
    assert proposal.parent_program_ids == [1, 2]
    assert evaluated_candidates == [merged]
    assert state.total_num_evals == len(proposal.subsample_indices)
