# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for batch-based parallel proposals."""

from itertools import pairwise
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import create_mocked_lms_context

from gepa.core.adapter import EvaluationBatch, default_batch_evaluate
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa.strategies.proposal_sampling import (
    IndependentSampling,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
)
from gepa.strategies.proposal_selection import (
    AllImprovements,
    BestImprovement,
    TopKImprovements,
)


@pytest.fixture
def mock_state():
    """Create a GEPAState with 2 candidates."""
    seed_candidate = {"system_prompt": "test0"}
    base_eval = ValsetEvaluation(
        outputs_by_val_id={0: "out0", 1: "out1"},
        scores_by_val_id={0: 0.5, 1: 0.5},
        objective_scores_by_val_id=None,
    )
    state = GEPAState(seed_candidate, base_eval, track_best_outputs=False)

    # Add a second candidate
    state.program_candidates.append({"system_prompt": "test1"})
    state.prog_candidate_val_subscores.append({0: 0.7, 1: 0.7})
    state.prog_candidate_objective_scores.append({})
    state.parent_program_for_candidate.append([0])
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.num_metric_calls_by_discovery.append(0)

    for i in range(len(state.pareto_front_valset)):
        state.program_at_pareto_front_valset[i] = {0, 1}

    assert state.is_consistent()
    return state


@pytest.fixture
def mock_candidate_selector():
    selector = MagicMock()
    selector.select_candidate_idx = MagicMock(return_value=0)
    return selector


@pytest.fixture
def mock_batch_sampler():
    sampler = MagicMock()
    call_count = 0

    def next_ids(trainset, state):
        nonlocal call_count
        call_count += 1
        return [call_count - 1]

    sampler.next_minibatch_ids = MagicMock(side_effect=next_ids)
    return sampler


@pytest.fixture
def mock_trainset():
    trainset = MagicMock()
    trainset.fetch = MagicMock(side_effect=lambda ids: [f"example_{i}" for i in ids])
    trainset.all_ids = MagicMock(return_value=[0, 1, 2, 3])
    trainset.__len__ = MagicMock(return_value=4)
    return trainset


class TestSamplingStrategies:
    def test_single_mutation(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = SingleMutationSampling()
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 1
        assert tasks[0].parent_idx == 0

    def test_same_parent_sampling(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = SameParentSampling(n=3)
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 3
        # All tasks should have the same parent
        assert all(t.parent_idx == tasks[0].parent_idx for t in tasks)
        # But different minibatches
        ids = [tuple(t.minibatch_ids) for t in tasks]
        assert len(set(ids)) == 3

    def test_independent_sampling(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = IndependentSampling(n=4)
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 4
        # Selector called 4 times
        assert mock_candidate_selector.select_candidate_idx.call_count == 4

    def test_pxn_sampling(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = PxNSampling(p=2, n=3)
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 6  # 2 * 3
        # Selector called 2 times (once per parent)
        assert mock_candidate_selector.select_candidate_idx.call_count == 2


class TestSelectionStrategies:
    def _make_proposal(self, before: float, after: float) -> CandidateProposal:
        return CandidateProposal(
            candidate={"prompt": "test"},
            parent_program_ids=[0],
            subsample_indices=[0],
            subsample_scores_before=[before],
            subsample_scores_after=[after],
            eval_before=SubsampleEvaluation(scores=[before], outputs=["out"]),
            eval_after=SubsampleEvaluation(scores=[after], outputs=["out"]),
            tag="test",
        )

    def test_all_improvements(self, mock_state):
        strategy = AllImprovements()
        criterion = StrictImprovementAcceptance()
        proposals = [
            self._make_proposal(0.5, 0.8),  # improvement
            self._make_proposal(0.5, 0.3),  # regression
            self._make_proposal(0.5, 0.6),  # improvement
        ]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 2

    def test_best_improvement(self, mock_state):
        strategy = BestImprovement()
        criterion = StrictImprovementAcceptance()
        proposals = [
            self._make_proposal(0.5, 0.8),
            self._make_proposal(0.5, 0.9),
            self._make_proposal(0.5, 0.6),
        ]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 1
        assert accepted[0].subsample_scores_after == [0.9]

    def test_best_improvement_none_pass(self, mock_state):
        strategy = BestImprovement()
        criterion = StrictImprovementAcceptance()
        proposals = [self._make_proposal(0.5, 0.3)]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 0

    def test_top_k_improvements(self, mock_state):
        strategy = TopKImprovements(k=2)
        criterion = StrictImprovementAcceptance()
        proposals = [
            self._make_proposal(0.5, 0.9),
            self._make_proposal(0.5, 0.6),
            self._make_proposal(0.5, 0.8),
            self._make_proposal(0.5, 0.3),
        ]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 2
        assert accepted[0].subsample_scores_after == [0.9]
        assert accepted[1].subsample_scores_after == [0.8]


class TestDefaultBatchEvaluate:
    def test_sequential_fallback(self):
        adapter = MagicMock()
        eval_result = EvaluationBatch(outputs=["out"], scores=[0.5])
        adapter.evaluate = MagicMock(return_value=eval_result)

        items = [
            ({"prompt": "a"}, ["ex1"]),
            ({"prompt": "b"}, ["ex2"]),
        ]
        results = default_batch_evaluate(adapter, items)
        assert len(results) == 2
        assert adapter.evaluate.call_count == 2


class _EventLog:
    """Records every callback event GEPA emits, in order."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __getattr__(self, name):
        if name.startswith("on_"):

            def _record(event, _name=name):
                self.events.append((_name, dict(event)))

            return _record
        raise AttributeError(name)

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def of(self, name: str) -> list[dict]:
        return [e for n, e in self.events if n == name]


_AIME_CACHE_DIR = Path(__file__).parent / "test_aime_prompt_optimization"
_AIME_SEED_PROMPT = (
    "You are a helpful assistant. You are given a question and you need to answer it. "
    "The answer should be given at the end of your response in exactly the format '### <final answer>'"
)


@pytest.fixture(scope="module")
def aime_lms():
    """Replay LMs backed by the recorded AIME run (fails on any unseen LM input)."""
    yield from create_mocked_lms_context(_AIME_CACHE_DIR)


def _run_recorded_optimization(aime_lms, sampling_strategy=None, selection_strategy=None):
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    task_lm, reflection_lm = aime_lms
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    log = _EventLog()
    result = gepa.optimize(
        seed_candidate={"system_prompt": _AIME_SEED_PROMPT},
        trainset=trainset[:10],
        valset=valset[:10],
        adapter=DefaultAdapter(model=task_lm),
        max_metric_calls=30,
        reflection_lm=reflection_lm,
        callbacks=[log],
        display_progress_bar=False,
        sampling_strategy=sampling_strategy,
        selection_strategy=selection_strategy,
    )
    return result, log


@pytest.fixture(scope="module")
def default_run(aime_lms):
    """The default path (sampling/selection strategies left unset)."""
    return _run_recorded_optimization(aime_lms)


@pytest.fixture(scope="module")
def explicit_run(aime_lms):
    """Same run with the documented defaults passed explicitly."""
    return _run_recorded_optimization(
        aime_lms,
        sampling_strategy=SingleMutationSampling(),
        selection_strategy=AllImprovements(),
    )


class TestDefaultStrategiesRetainBehavior:
    """Pin the default path against a fully cached, pre-#369 GEPA run.

    The replay fixture hard-fails on any LM input that is not in the recorded
    cache, so these tests only pass if the default
    SingleMutationSampling + AllImprovements path reproduces the recorded run's
    exact LM request stream. On top of that we pin the final result, the budget
    ledger, and the callback-event stream.
    """

    def test_default_path_reproduces_recorded_run(self, default_run):
        result, _ = default_run
        golden = (_AIME_CACHE_DIR / "optimized_prompt.txt").read_text()
        assert result.best_candidate["system_prompt"] == golden

    def test_budget_ledger_is_consistent(self, default_run):
        result, log = default_run
        budget = log.of("on_budget_updated")
        assert budget, "expected at least one on_budget_updated event"
        assert all(e["metric_calls_delta"] > 0 for e in budget)
        used = [e["metric_calls_used"] for e in budget]
        assert used == sorted(used), "metric_calls_used must be monotonic"
        # Every event's absolute counter must agree with the previous one plus
        # its own delta (no unaccounted increments once the hook is live).
        for prev, cur in pairwise(budget):
            assert cur["metric_calls_used"] == prev["metric_calls_used"] + cur["metric_calls_delta"]
        # The only increment that predates hook registration is the seed's
        # full-valset evaluation (10 examples in the recorded run).
        assert used[0] - budget[0]["metric_calls_delta"] == 10
        assert used[-1] == result.total_metric_calls

    def test_explicit_defaults_identical_to_implicit_defaults(self, default_run, explicit_run):
        d_result, d_log = default_run
        e_result, e_log = explicit_run
        assert e_result.best_candidate == d_result.best_candidate
        assert e_result.total_metric_calls == d_result.total_metric_calls
        assert e_log.names() == d_log.names()

    def test_proposal_event_ordering(self, default_run):
        _, log = default_run
        names = log.names()
        for earlier, later in [
            ("on_candidate_selected", "on_minibatch_sampled"),
            ("on_minibatch_sampled", "on_reflective_dataset_built"),
            ("on_reflective_dataset_built", "on_proposal_start"),
            ("on_proposal_start", "on_proposal_end"),
        ]:
            assert earlier in names and later in names, f"missing {earlier}/{later}"
            assert names.index(earlier) < names.index(later)

    def test_budget_event_granularity_contract(self, default_run):
        """Each completed proposal emits separate parent-eval and child-eval
        budget events (finer granularity than pre-#369; totals unchanged), and
        each *accepted* candidate's full-valset evaluation emits one more (the
        seed's valset eval precedes budget-hook registration, so it produces an
        on_valset_evaluated event but no budget event). Pin the composition so
        any future change to budget-event granularity is made consciously."""
        _, log = default_run
        n_proposals = len(log.of("on_proposal_end"))
        n_budget = len(log.of("on_budget_updated"))
        n_accepted = len(log.of("on_candidate_accepted"))
        assert n_budget == 2 * n_proposals + n_accepted, (
            f"budget-event composition changed: {n_budget} events for "
            f"{n_proposals} proposals + {n_accepted} accepted candidates"
        )

    def test_child_evaluation_events_are_emitted(self, default_run):
        """Parity with the pre-#369 sequential path: every completed proposal
        produces an evaluation start/end pair for the parent minibatch eval AND
        one for the new candidate's minibatch eval (child pairs carry
        candidate_idx=None because the child is not in the pool yet)."""
        _, log = default_run
        starts = log.of("on_evaluation_start")
        ends = log.of("on_evaluation_end")
        n_proposals = len(log.of("on_proposal_end"))
        assert len(starts) == len(ends) == 2 * n_proposals
        child_ends = [e for e in ends if e["candidate_idx"] is None]
        assert len(child_ends) == n_proposals
