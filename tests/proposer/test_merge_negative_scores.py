import random

import pytest

from gepa.proposer.merge import find_common_ancestor_pair, sample_and_attempt_merge_programs_by_common_predictors


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
