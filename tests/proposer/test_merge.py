import random
from types import SimpleNamespace

from gepa.proposer.merge import (
    MergeProposer,
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
        [],  # program 0 (root)
        [0],  # program 1 -> parent 0
        [0],  # program 2 -> parent 0
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
        [],  # 0
        [0],  # 1
        [0],  # 2
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


class _StubLogger:
    def log(self, _msg):
        pass


class _StubValset:
    def fetch(self, ids):
        return [{"id": idx} for idx in ids]


def _make_state(prog_val_scores, evaluator=None):
    state = SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_at_pareto_front_valset={0: {1, 2}},
        program_full_scores_val_set=[0.1, 0.6, 0.7],
        program_candidates=[{"pred": "base"}, {"pred": "p1"}, {"pred": "p2"}],
        parent_program_for_candidate=[[None], [0], [0]],
        prog_candidate_val_subscores=prog_val_scores,
        total_num_evals=0,
        evaluation_cache=None,  # No cache for tests
    )
    # Add the get_pareto_front_mapping method to match GEPAState interface
    state.get_pareto_front_mapping = lambda: {
        val_id: set(front) for val_id, front in state.program_at_pareto_front_valset.items()
    }
    # Add increment_evals method to match GEPAState interface
    state.increment_evals = lambda count: setattr(state, "total_num_evals", state.total_num_evals + count)

    # Add cached_evaluate method to match GEPAState interface (no caching for stubs)
    def cached_evaluate(candidate, example_ids, fetcher, eval_fn):
        _, scores, _ = eval_fn(fetcher(example_ids), candidate)
        return scores, len(example_ids)

    state.cached_evaluate = cached_evaluate

    # Add cached_evaluate_full method to match GEPAState interface (no caching for stubs)
    def cached_evaluate_full(candidate, example_ids, fetcher, eval_fn):
        outputs, scores, obj_scores = eval_fn(fetcher(example_ids), candidate)
        outputs_by_id = dict(zip(example_ids, outputs, strict=False))
        scores_by_id = dict(zip(example_ids, scores, strict=False))
        objective_by_id = dict(zip(example_ids, obj_scores, strict=False)) if obj_scores else None
        return outputs_by_id, scores_by_id, objective_by_id, len(example_ids)

    state.cached_evaluate_full = cached_evaluate_full
    return state


def test_merge_proposer_skips_pairs_below_overlap_floor(monkeypatch):
    proposer = MergeProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.0 for _ in batch], None),
        use_merge=True,
        max_merge_invocations=5,
        val_overlap_floor=2,
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.merges_due = 1

    state = _make_state(
        [
            {0: 0.1, 1: 0.2},
            {0: 0.4, 1: 0.5},
            {1: 0.55},  # intersection with program 1 has size 1 (< floor)
        ]
    )

    monkeypatch.setattr(
        "gepa.proposer.merge.find_dominator_programs",
        lambda *_args, **_kwargs: [1, 2],
    )

    calls: list[bool] = []

    def fake_sample(
        agg_scores,
        rng,
        merge_candidates,
        merges_performed,
        program_candidates,
        parent_program_for_candidate,
        has_val_support_overlap=None,
        **_kwargs,
    ):
        id1, id2 = merge_candidates[:2]
        allowed = has_val_support_overlap(id1, id2) if has_val_support_overlap else True
        calls.append(allowed)
        if not allowed:
            return None
        merges_performed[1].append((id1, id2, ("desc",)))
        return ({"pred": "merged"}, id1, id2, 0)

    monkeypatch.setattr(
        "gepa.proposer.merge.sample_and_attempt_merge_programs_by_common_predictors",
        fake_sample,
    )

    result = proposer.propose(state)

    assert result is None
    assert calls == [False]
    assert proposer.merges_performed[0] == []
    assert proposer.merges_performed[1] == []


def test_merge_proposer_allows_pairs_meeting_overlap_floor(monkeypatch):
    proposer = MergeProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.9 for _ in batch], None),
        use_merge=True,
        max_merge_invocations=5,
        val_overlap_floor=2,
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.merges_due = 1

    state = _make_state(
        [
            {0: 0.1, 1: 0.2},
            {0: 0.4, 1: 0.5},
            {0: 0.45, 1: 0.55},  # perfect overlap with program 1
        ]
    )

    monkeypatch.setattr(
        "gepa.proposer.merge.find_dominator_programs",
        lambda *_args, **_kwargs: [1, 2],
    )

    calls: list[bool] = []

    def fake_sample(
        agg_scores,
        rng,
        merge_candidates,
        merges_performed,
        program_candidates,
        parent_program_for_candidate,
        has_val_support_overlap=None,
        **_kwargs,
    ):
        id1, id2 = merge_candidates[:2]
        allowed = has_val_support_overlap(id1, id2) if has_val_support_overlap else True
        calls.append(allowed)
        if not allowed:
            return None
        merges_performed[1].append((id1, id2, ("desc",)))
        return ({"pred": "merged"}, id1, id2, 0)

    monkeypatch.setattr(
        "gepa.proposer.merge.sample_and_attempt_merge_programs_by_common_predictors",
        fake_sample,
    )

    monkeypatch.setattr(
        MergeProposer,
        "select_eval_subsample_for_merged_program",
        lambda self, scores1, scores2, num_subsample_ids=5: [0, 1],
    )

    result = proposer.propose(state)

    assert result is not None
    assert calls == [True]
    assert result.parent_program_ids == [1, 2]
    assert result.subsample_indices == [0, 1]
    assert proposer.merges_performed[0] == [(1, 2, 0)]
    assert proposer.merges_performed[1] == [(1, 2, ("desc",))]
    assert state.total_num_evals == 2
    assert state.full_program_trace[-1]["merged"] is True
