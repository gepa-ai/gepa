import random
import sys

import pytest

from gepa.proposer.merge import find_common_ancestor_pair, sample_and_attempt_merge_programs_by_common_predictors


@pytest.mark.parametrize("shared_parents", [False, True])
def test_merge_combines_complementary_descendants_of_a_deep_lineage(shared_parents):
    depth = sys.getrecursionlimit() + 50
    parents = [[None]]
    for idx in range(1, depth):
        parents.append([idx - 1, max(0, idx - 2)] if shared_parents else [idx - 1])
    parents.extend([[depth - 1], [depth - 1]])
    candidates = [{"planner": "original plan", "solver": "original solution"} for _ in range(depth)]
    candidates.extend(
        [
            {"planner": "better plan", "solver": "original solution"},
            {"planner": "original plan", "solver": "better solution"},
        ]
    )
    # Only the root is a viable merge base: intermediate ancestors outscore
    # both tips and must still be excluded by the existing filtering policy.
    scores = [0.1] + [0.9] * (depth - 1) + [0.5, 0.5]

    result = sample_and_attempt_merge_programs_by_common_predictors(
        agg_scores=scores,
        rng=random.Random(42),
        merge_candidates=[depth, depth + 1],
        merges_performed=([], []),
        program_candidates=candidates,
        parent_program_for_candidate=parents,
    )

    assert result == ({"planner": "better plan", "solver": "better solution"}, depth, depth + 1, 0)


def test_deep_lineage_still_excludes_ancestor_descendant_pairs():
    depth = sys.getrecursionlimit() + 50
    parents = [[None]] + [[idx - 1] for idx in range(1, depth)]

    result = find_common_ancestor_pair(
        rng=random.Random(42),
        parent_list=parents,
        program_indexes=[depth - 2, depth - 1],
        merges_performed=([], []),
        agg_scores=[0.5] * depth,
        program_candidates=[{"prompt": str(idx)} for idx in range(depth)],
        max_attempts=1,
    )

    assert result is None


def test_disconnected_deep_lineages_have_no_merge_base():
    depth = sys.getrecursionlimit() + 50
    parents = [[None]] + [[idx - 1] for idx in range(1, depth)]
    parents += [[None]] + [[idx - 1] for idx in range(depth + 1, 2 * depth)]

    result = find_common_ancestor_pair(
        rng=random.Random(42),
        parent_list=parents,
        program_indexes=[depth - 1, 2 * depth - 1],
        merges_performed=([], []),
        agg_scores=[0.5] * (2 * depth),
        program_candidates=[{"prompt": str(idx)} for idx in range(2 * depth)],
        max_attempts=1,
    )

    assert result is None
