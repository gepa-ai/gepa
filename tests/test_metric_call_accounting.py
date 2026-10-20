# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Regression for #139: count repeated metric work across every validation path."""

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.engine import GEPAEngine
from gepa.core.state import VALSET_CACHE_SPLIT, EvaluationCache, GEPAState, ValsetEvaluation
from gepa.strategies.proposal_sampling import IndependentSampling
from gepa.strategies.proposal_selection import AllImprovements


class CountingAdapter:
    def __init__(self, repetitions=3):
        self.repetitions = repetitions
        self.calls = 0
        self.propose_new_texts = self.propose

    def evaluate(self, batch, candidate, capture_traces=False):
        outputs, scores = [], []
        for example in batch:
            observations = []
            for _ in range(self.repetitions):
                self.calls += 1
                observations.append(len(candidate["p"]) / 100)
            scores.append(sum(observations) / len(observations))
            outputs.append(example)
        return EvaluationBatch(
            outputs,
            scores,
            [object() for _ in batch] if capture_traces else None,
            num_metric_calls=len(batch) * self.repetitions,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {component: [{"score": score} for score in eval_batch.scores] for component in components_to_update}

    def propose(self, candidate, reflective_dataset, components_to_update, **kwargs):
        return {component: candidate[component] + " x" for component in components_to_update}


def _run(adapter, tmp_path, *, budget=1, cache=False, traced=False, valset=None, multi=False):
    return gepa.optimize(
        seed_candidate={"p": "s"},
        trainset=[0, 1, 2, 3],
        valset=valset,
        adapter=adapter,
        max_metric_calls=budget,
        reflection_minibatch_size=2,
        candidate_selection_strategy="current_best",
        cache_evaluation=cache,
        write_agent_state=traced,
        run_dir=str(tmp_path),
        sampling_strategy=IndependentSampling(2) if multi else None,
        selection_strategy=AllImprovements() if multi else None,
    )


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("traced", [False, True])
def test_seed_budget_counts_actual_work(cache, traced, tmp_path):
    adapter = CountingAdapter()
    result = _run(adapter, tmp_path, cache=cache, traced=traced)
    assert adapter.calls == 12
    assert result.total_metric_calls == adapter.calls
    assert len(result.candidates) == 1


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("shared_valset", [False, True])
def test_full_run_budget_matches_all_instrumented_calls(cache, traced, shared_valset, tmp_path):
    adapter = CountingAdapter()
    result = _run(adapter, tmp_path, budget=65, cache=cache, traced=traced, valset=None if shared_valset else [7, 8, 9])
    assert len(result.candidates) > 1
    assert result.total_metric_calls == adapter.calls


def test_batched_validation_keeps_each_candidates_explicit_call_count(tmp_path):
    adapter = CountingAdapter()
    result = _run(adapter, tmp_path, budget=65, multi=True, valset=[7, 8, 9])
    assert len(result.candidates) > 2
    assert result.total_metric_calls == adapter.calls


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_cached_evaluation_uses_explicit_counts_for_misses_and_zero_for_hits(cache_enabled, legacy):
    cache = EvaluationCache() if cache_enabled else None
    state = GEPAState({"p": "s"}, ValsetEvaluation({0: "o"}, {0: 0.0}), evaluation_cache=cache)
    adapter = CountingAdapter()

    def evaluate(batch, candidate):
        evaluation = adapter.evaluate(batch, candidate)
        return (evaluation.outputs, evaluation.scores, evaluation.objective_scores) if legacy else evaluation

    _, _, _, initial = state.cached_evaluate_full(
        {"p": "x"}, [0, 1], lambda ids: ids, evaluate, split=VALSET_CACHE_SPLIT
    )
    assert initial == (2 if legacy else adapter.calls)
    before = adapter.calls
    _, _, _, additional = state.cached_evaluate_full(
        {"p": "x"}, [0, 1, 2], lambda ids: ids, evaluate, split=VALSET_CACHE_SPLIT
    )
    assert additional == ((1 if cache_enabled else 3) if legacy else adapter.calls - before)
    before = adapter.calls
    _, _, _, same = state.cached_evaluate_full(
        {"p": "x"}, [0, 1, 2], lambda ids: ids, evaluate, split=VALSET_CACHE_SPLIT
    )
    assert same == (0 if cache_enabled else (3 if legacy else adapter.calls - before))


def test_merge_factory_passes_metric_metadata_through_to_the_cache(monkeypatch, tmp_path):
    adapter = CountingAdapter()
    seen = []

    def intercepted_run(engine):
        assert engine.merge_proposer is not None
        state = GEPAState({"p": "s"}, ValsetEvaluation({0: "o"}, {0: 0.0}), evaluation_cache=EvaluationCache())
        for evaluator in [engine.evaluator, engine.merge_proposer.evaluator]:
            before = adapter.calls
            _, _, _, count = state.cached_evaluate_full(
                {"p": str(before)}, [0, 1], lambda ids: ids, evaluator, split=VALSET_CACHE_SPLIT
            )
            seen.append(count)
            assert count == adapter.calls - before
        raise StopIteration

    monkeypatch.setattr(GEPAEngine, "run", intercepted_run)
    with pytest.raises(StopIteration):
        gepa.optimize(
            seed_candidate={"p": "s"},
            trainset=[0, 1],
            adapter=adapter,
            use_merge=True,
            max_metric_calls=10,
            run_dir=str(tmp_path),
        )
    assert seen == [6, 6]


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("traced", [False, True])
def test_wrapper_preserves_the_underlying_proposer_and_complete_run_accounting(cache, traced, tmp_path):
    from gepa.adapters.multi_rollout import MultiRolloutAdapter

    original = CountingAdapter(repetitions=2)
    adapter = MultiRolloutAdapter(original, num_rollouts=3)
    result = _run(adapter, tmp_path, budget=100, cache=cache, traced=traced, valset=[7, 8, 9])
    assert len(result.candidates) > 1
    assert result.total_metric_calls == original.calls
    assert result.val_aggregate_scores[-1] > result.val_aggregate_scores[0]


def test_explicit_zero_metric_calls_is_not_replaced_by_example_count():
    state = GEPAState({"p": "s"}, ValsetEvaluation({0: "o"}, {0: 0.0}))

    def evaluate(batch, candidate):
        return EvaluationBatch(batch, [0.0] * len(batch), num_metric_calls=0)

    assert state.cached_evaluate_full({"p": "x"}, [0, 1], lambda ids: ids, evaluate, split=VALSET_CACHE_SPLIT)[3] == 0
