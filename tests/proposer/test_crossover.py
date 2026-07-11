import random
from types import SimpleNamespace

from gepa.proposer.crossover import CrossoverProposer, CrossoverSignature


class _StubLogger:
    def log(self, _msg):
        pass


class _StubValset:
    def fetch(self, ids):
        return [{"id": idx} for idx in ids]


def _make_state(program_candidates, prog_val_scores, frontier, tracked_scores):
    state = SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_at_pareto_front_valset=frontier,
        program_candidates=program_candidates,
        prog_candidate_val_subscores=prog_val_scores,
        per_program_tracked_scores=tracked_scores,
        total_num_evals=0,
        evaluation_cache=None,
    )
    state.get_pareto_front_mapping = lambda: {
        val_id: set(front) for val_id, front in state.program_at_pareto_front_valset.items()
    }
    state.increment_evals = lambda count: setattr(state, "total_num_evals", state.total_num_evals + count)

    def cached_evaluate_full(candidate, example_ids, fetcher, eval_fn):
        outputs, scores, obj_scores = eval_fn(fetcher(example_ids), candidate)
        outputs_by_id = dict(zip(example_ids, outputs, strict=False))
        scores_by_id = dict(zip(example_ids, scores, strict=False))
        objective_by_id = dict(zip(example_ids, obj_scores, strict=False)) if obj_scores else None
        return outputs_by_id, scores_by_id, objective_by_id, len(example_ids)

    state.cached_evaluate_full = cached_evaluate_full
    return state


def _complementary_state():
    # Programs 1 and 2 win on disjoint validation instances and differ in text.
    return _make_state(
        program_candidates=[{"pred": "base"}, {"pred": "variant-A"}, {"pred": "variant-B"}],
        prog_val_scores=[
            {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
            {0: 0.9, 1: 0.9, 2: 0.1, 3: 0.1},  # program 1 wins on {0, 1}
            {0: 0.1, 1: 0.1, 2: 0.9, 3: 0.9},  # program 2 wins on {2, 3}
        ],
        frontier={0: {1}, 1: {1}, 2: {2}, 3: {2}},
        tracked_scores=[0.5, 0.5, 0.5],
    )


def test_crossover_output_extractor_reads_fenced_block():
    assert CrossoverSignature.output_extractor("```\nnew text\n```") == {"new_instruction": "new text"}


def test_crossover_synthesizes_from_complementary_pair():
    reflection_calls = []

    def fake_reflection_lm(prompt):
        reflection_calls.append(prompt)
        return "```\nsynthesized instruction\n```"

    proposer = CrossoverProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.95 for _ in batch], None),
        reflection_lm=fake_reflection_lm,
        use_crossover=True,
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.crossovers_due = 1

    state = _complementary_state()
    proposal = proposer.propose(state)

    assert proposal is not None
    assert proposal.tag == "crossover"
    assert proposal.parent_program_ids == [1, 2]
    assert proposal.candidate["pred"] == "synthesized instruction"
    # The synthesis prompt should include both parent texts.
    assert len(reflection_calls) == 1
    assert "variant-A" in reflection_calls[0]
    assert "variant-B" in reflection_calls[0]
    # Subsample was drawn from the contested instances and counted as evals.
    assert proposal.subsample_indices
    assert set(proposal.subsample_indices).issubset({0, 1, 2, 3})
    assert state.total_num_evals == len(proposal.subsample_indices)


def test_crossover_skips_when_not_scheduled():
    proposer = CrossoverProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.95 for _ in batch], None),
        reflection_lm=lambda _p: "```\nx\n```",
        use_crossover=True,
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = False  # nothing new last iteration
    proposer.crossovers_due = 1

    assert proposer.propose(_complementary_state()) is None


def test_crossover_skips_when_parents_share_component_text():
    called = []

    proposer = CrossoverProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.95 for _ in batch], None),
        reflection_lm=lambda p: called.append(p) or "```\nx\n```",
        use_crossover=True,
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.crossovers_due = 1

    # Complementary win-sets but identical component text -> nothing to synthesize.
    state = _make_state(
        program_candidates=[{"pred": "base"}, {"pred": "same"}, {"pred": "same"}],
        prog_val_scores=[
            {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
            {0: 0.9, 1: 0.9, 2: 0.1, 3: 0.1},
            {0: 0.1, 1: 0.1, 2: 0.9, 3: 0.9},
        ],
        frontier={0: {1}, 1: {1}, 2: {2}, 3: {2}},
        tracked_scores=[0.5, 0.5, 0.5],
    )

    assert proposer.propose(state) is None
    assert called == []  # no LM call when there is no divergent component


def test_crossover_selects_valid_but_lopsided_pair():
    # Program 1 uniquely wins 2 instances, program 2 uniquely wins 8 (diff = 6).
    # Both clear min_win_asymmetry=2, so this genuinely complementary pair must be
    # selected even though the win-sets are very unbalanced.
    n = 10
    prog1 = {k: (0.9 if k < 2 else 0.1) for k in range(n)}
    prog2 = {k: (0.1 if k < 2 else 0.9) for k in range(n)}
    base = dict.fromkeys(range(n), 0.05)
    frontier = {k: ({1} if k < 2 else {2}) for k in range(n)}

    proposer = CrossoverProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.95 for _ in batch], None),
        reflection_lm=lambda _p: "```\nsynth\n```",
        use_crossover=True,
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.crossovers_due = 1

    state = _make_state(
        program_candidates=[{"pred": "base"}, {"pred": "variant-A"}, {"pred": "variant-B"}],
        prog_val_scores=[base, prog1, prog2],
        frontier=frontier,
        tracked_scores=[0.05, 0.5, 0.5],
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.parent_program_ids == [1, 2]


def test_crossover_retries_other_component_on_multicomponent_pair():
    # Parents differ in BOTH components; consecutive crossover schedules should try
    # each divergent component once, then retire the pair (rather than blacklisting
    # the whole pair after the first attempt).
    state = _make_state(
        program_candidates=[
            {"a": "base-a", "b": "base-b"},
            {"a": "A1", "b": "B1"},
            {"a": "A2", "b": "B2"},
        ],
        prog_val_scores=[
            {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
            {0: 0.9, 1: 0.9, 2: 0.1, 3: 0.1},
            {0: 0.1, 1: 0.1, 2: 0.9, 3: 0.9},
        ],
        frontier={0: {1}, 1: {1}, 2: {2}, 3: {2}},
        tracked_scores=[0.5, 0.5, 0.5],
    )

    proposer = CrossoverProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.95 for _ in batch], None),
        reflection_lm=lambda _p: "```\nsynth\n```",
        use_crossover=True,
        rng=random.Random(0),
    )

    used = []
    for _ in range(2):
        proposer.last_iter_found_new_program = True
        proposer.crossovers_due = 1
        proposal = proposer.propose(state)
        assert proposal is not None
        used.append(proposal.metadata["component"])

    assert set(used) == {"a", "b"}  # both divergent components were tried

    # A third schedule finds nothing new and retires the pair.
    proposer.last_iter_found_new_program = True
    proposer.crossovers_due = 1
    assert proposer.propose(state) is None
    assert (1, 2) in proposer._exhausted_pairs


def test_crossover_skips_when_no_complementary_pair():
    proposer = CrossoverProposer(
        logger=_StubLogger(),
        valset=_StubValset(),
        evaluator=lambda batch, prog: (batch, [0.95 for _ in batch], None),
        reflection_lm=lambda _p: "```\nx\n```",
        use_crossover=True,
        min_win_asymmetry=3,  # require 3 unique wins each; only 2 available
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.crossovers_due = 1

    assert proposer.propose(_complementary_state()) is None
