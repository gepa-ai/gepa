from gepa.proposer.merge import (
    does_triplet_have_desirable_predictors,
    filter_ancestors,
    find_common_ancestor_pair,
    sample_and_attempt_merge_programs_by_common_predictors,
)


def test_does_triplet_have_desirable_predictors_true_when_descendants_diverge():
    program_candidates = [
        {"pred": "A"},  # ancestor (0)
        {"pred": "A"},  # id1 (1) matches ancestor
        {"pred": "B"},  # id2 (2) differs
    ]

    assert does_triplet_have_desirable_predictors(program_candidates, 0, 1, 2)


def test_does_triplet_have_desirable_predictors_false_when_descendants_identical():
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "A"},
    ]

    assert not does_triplet_have_desirable_predictors(program_candidates, 0, 1, 2)


def test_filter_ancestors_skips_previously_merged_triplets():
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "B"},
    ]
    agg_scores = [0.1, 0.5, 0.6]
    merges_performed = ([(1, 2, 0)], [])

    result = filter_ancestors(
        1,
        2,
        {0},
        merges_performed,
        agg_scores,
        program_candidates,
    )

    assert result == []


def test_filter_ancestors_skips_when_ancestor_outscores_descendants():
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "B"},
    ]
    agg_scores = [0.9, 0.5, 0.6]  # ancestor outperforms descendants
    merges_performed = ([], [])

    result = filter_ancestors(
        1,
        2,
        {0},
        merges_performed,
        agg_scores,
        program_candidates,
    )

    assert result == []


def test_filter_ancestors_returns_viable_common_ancestor():
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "B"},
    ]
    agg_scores = [0.1, 0.6, 0.7]
    merges_performed = ([], [])

    result = filter_ancestors(
        1,
        2,
        {0},
        merges_performed,
        agg_scores,
        program_candidates,
    )

    assert result == [0]


def test_find_common_ancestor_pair_returns_expected_triplet(rng):
    rng.seed(0)
    parent_list = [
        [],     # program 0 (root)
        [0],    # program 1 -> parent 0
        [0],    # program 2 -> parent 0
    ]
    program_indexes = [1, 2]
    agg_scores = [0.1, 0.6, 0.7]
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "B"},
    ]

    result = find_common_ancestor_pair(
        rng,
        parent_list,
        program_indexes,
        merges_performed=([], []),
        agg_scores=agg_scores,
        program_candidates=program_candidates,
        max_attempts=3,
    )

    assert result == (1, 2, 0)


def test_find_common_ancestor_pair_returns_none_when_already_merged(rng):
    rng.seed(0)
    parent_list = [
        [],
        [0],
        [0],
    ]
    program_indexes = [1, 2]
    agg_scores = [0.1, 0.6, 0.7]
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "B"},
    ]

    result = find_common_ancestor_pair(
        rng,
        parent_list,
        program_indexes,
        merges_performed=([(1, 2, 0)], []),
        agg_scores=agg_scores,
        program_candidates=program_candidates,
        max_attempts=3,
    )

    assert result is None


def test_sample_and_attempt_merge_creates_combined_program_and_records_triplet(rng):
    rng.seed(0)
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "B"},
    ]
    agg_scores = [0.1, 0.6, 0.7]
    merges_performed = ([], [])
    parent_program_for_candidate = [
        [],    # 0
        [0],   # 1
        [0],   # 2
    ]

    result = sample_and_attempt_merge_programs_by_common_predictors(
        agg_scores=agg_scores,
        rng=rng,
        merge_candidates=[1, 2],
        merges_performed=merges_performed,
        program_candidates=program_candidates,
        parent_program_for_candidate=parent_program_for_candidate,
    )

    assert result is not None
    new_program, id1, id2, ancestor = result
    assert (id1, id2, ancestor) == (1, 2, 0)
    assert new_program == {"pred": "B"}
    assert merges_performed[1] == [(1, 2, (2,))]

    # Attempting the same merge again should return None because the triplet was recorded
    second_attempt = sample_and_attempt_merge_programs_by_common_predictors(
        agg_scores=agg_scores,
        rng=rng,
        merge_candidates=[1, 2],
        merges_performed=merges_performed,
        program_candidates=program_candidates,
        parent_program_for_candidate=parent_program_for_candidate,
    )
    assert second_attempt is None


def test_sample_and_attempt_merge_respects_val_support_overlap_gate(rng):
    rng.seed(0)
    program_candidates = [
        {"pred": "A"},
        {"pred": "A"},
        {"pred": "B"},
    ]
    agg_scores = [0.1, 0.6, 0.7]
    merges_performed = ([], [])
    parent_program_for_candidate = [
        [],
        [0],
        [0],
    ]

    result = sample_and_attempt_merge_programs_by_common_predictors(
        agg_scores=agg_scores,
        rng=rng,
        merge_candidates=[1, 2],
        merges_performed=merges_performed,
        program_candidates=program_candidates,
        parent_program_for_candidate=parent_program_for_candidate,
        has_val_support_overlap=lambda _id1, _id2: False,
        max_attempts=3,
    )

    assert result is None
    assert merges_performed[1] == []
