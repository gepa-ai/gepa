# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for reusing parent rollouts instead of re-evaluating them (#357)."""

from itertools import count
from unittest.mock import MagicMock

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import TRAINSET_CACHE_SPLIT, EvaluationCache, GEPAState, ValsetEvaluation
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer


class _FixedProposalLM:
    def __init__(self, text: str | dict[str, str] = "improved"):
        self.text = text

    def reflect(self, candidate, reflective_dataset, components_to_update):
        new_texts = dict(self.text) if isinstance(self.text, dict) else dict.fromkeys(components_to_update, self.text)
        return ReflectionProposal(new_texts=new_texts), self


def _make_state(example_count: int = 1, cache: EvaluationCache | None = None) -> GEPAState:
    base_eval = ValsetEvaluation(
        outputs_by_val_id={i: f"seed-out-{i}" for i in range(example_count)},
        scores_by_val_id=dict.fromkeys(range(example_count), 0.5),
        objective_scores_by_val_id=None,
    )
    state = GEPAState({"c": "seed"}, base_eval, track_best_outputs=False, evaluation_cache=cache)
    # Matches the engine's seed evaluation cost.
    state.total_num_evals = example_count
    # The engine appends a per-iteration trace entry before calling propose().
    state.full_program_trace.append({"i": 0})
    return state


def _make_adapter(calls: list[list[tuple[dict[str, str], list]]]) -> MagicMock:
    def batch_evaluate(items, capture_traces=True):
        calls.append([(dict(candidate), list(batch)) for candidate, batch in items])
        evaluations = []
        for candidate, batch in items:
            evaluations.append(
                EvaluationBatch(
                    outputs=[f"out-{candidate['c']}-{i}" for i in range(len(batch))],
                    scores=[0.8] * len(batch),
                    trajectories=[{"candidate": candidate["c"], "index": i} for i in range(len(batch))]
                    if capture_traces
                    else None,
                    num_metric_calls=len(batch),
                )
            )
        return evaluations

    adapter = MagicMock()
    adapter.propose_new_texts = None
    adapter.batch_evaluate = MagicMock(side_effect=batch_evaluate)
    adapter.make_reflective_dataset = MagicMock(return_value={"c": [{"feedback": "f"}]})
    return adapter


def _make_proposer(
    adapter: MagicMock,
    trainset: list,
    minibatch_ids: list[int],
    reflection_strategy: _FixedProposalLM | None = None,
) -> ReflectiveMutationProposer:
    candidate_selector = MagicMock()
    candidate_selector.select_candidate_idx.return_value = 0
    batch_sampler = MagicMock()
    batch_sampler.next_minibatch_ids.return_value = minibatch_ids
    module_selector = MagicMock(return_value=["c"])
    return ReflectiveMutationProposer(
        logger=MagicMock(),
        trainset=trainset,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector,
        batch_sampler=batch_sampler,
        perfect_score=None,
        skip_perfect_score=False,
        experiment_tracker=MagicMock(),
        reflection_strategy=reflection_strategy or _FixedProposalLM(),
    )


def test_cached_parent_rollout_with_trajectory_is_reused():
    cache: EvaluationCache = EvaluationCache()
    seed = {"c": "seed"}
    cache.put(
        seed,
        0,
        "cached-out",
        0.9,
        split=TRAINSET_CACHE_SPLIT,
        trajectory={"cached": True},
    )
    state = _make_state(cache=cache)
    calls: list[list[tuple[dict[str, str], list]]] = []
    adapter = _make_adapter(calls)
    proposer = _make_proposer(adapter, trainset=[{"q": 0}], minibatch_ids=[0])

    proposals = proposer.propose(state)

    # Only the child is evaluated; the cached parent rollout is reused.
    assert len(calls) == 1
    assert calls[0] == [({"c": "improved"}, [{"q": 0}])]
    assert len(proposals) == 1
    assert proposals[0].subsample_scores_before == [0.9]
    assert proposals[0].eval_before.trajectories == [{"cached": True}]
    # The cache hit must not consume metric-call budget.
    assert state.total_num_evals == 2  # seed + one child eval


def test_cached_entry_without_trajectory_is_re_evaluated():
    cache: EvaluationCache = EvaluationCache()
    seed = {"c": "seed"}
    # A valset/score-only cache entry cannot feed reflection.
    cache.put(seed, 0, "score-only", 0.9, split=TRAINSET_CACHE_SPLIT)
    state = _make_state(cache=cache)
    calls: list[list[tuple[dict[str, str], list]]] = []
    adapter = _make_adapter(calls)
    proposer = _make_proposer(adapter, trainset=[{"q": 0}], minibatch_ids=[0])

    proposals = proposer.propose(state)

    assert len(calls) == 2
    assert calls[0] == [({"c": "seed"}, [{"q": 0}])]
    assert proposals[0].subsample_scores_before == [0.8]
    assert proposals[0].eval_before.trajectories == [{"candidate": "seed", "index": 0}]
    assert state.total_num_evals == 3  # seed + parent + child


def test_unproductive_iteration_does_not_charge_cached_rollouts():
    """A cache hit is not work, so it must not consume metric-call budget."""
    cache: EvaluationCache = EvaluationCache()
    seed = {"c": "seed"}
    cache.put(seed, 0, "cached-out", 0.9, split=TRAINSET_CACHE_SPLIT, trajectory={"cached": True})
    state = _make_state(cache=cache)
    calls: list[list[tuple[dict[str, str], list]]] = []
    adapter = _make_adapter(calls)
    proposer = _make_proposer(
        adapter,
        trainset=[{"q": 0}],
        minibatch_ids=[0],
        reflection_strategy=_FixedProposalLM({}),
    )

    assert proposer.propose(state) == []
    assert calls == []
    assert state.total_num_evals == 1


def test_end_to_end_accepted_child_is_not_re_evaluated_as_parent(tmp_path):
    """A candidate evaluated as a child must be served from cache when it later
    becomes the parent, instead of paying for the same traced rollout twice."""
    calls: list[tuple[str, tuple[int, ...]]] = []

    class CountingAdapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            text = candidate["c"]
            ids = tuple(example["id"] for example in batch)
            if capture_traces:
                calls.append((text, ids))
            version = 0 if text == "seed" else int(text.removeprefix("v"))
            return EvaluationBatch(
                outputs=[text] * len(batch),
                scores=[float(version)] * len(batch),
                trajectories=[{"text": text, "id": i} for i in ids] if capture_traces else None,
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {name: [{"feedback": "improve"}] for name in components_to_update}

    counter = count(1)
    trainset = [{"id": 0}, {"id": 1}]
    gepa.optimize(
        seed_candidate={"c": "seed"},
        trainset=trainset,
        valset=trainset,
        adapter=CountingAdapter(),
        reflection_lm=lambda _prompt: f"v{next(counter)}",
        reflection_minibatch_size=2,
        max_metric_calls=20,
        cache_evaluation=True,
        skip_perfect_score=False,
        run_dir=str(tmp_path),
        display_progress_bar=False,
    )

    assert len(calls) >= 4  # at least two proposal iterations
    assert len(calls) == len(set(calls)), f"a traced rollout was evaluated twice: {calls}"


def test_partial_cache_hit_only_evaluates_missing_examples():
    cache: EvaluationCache = EvaluationCache()
    seed = {"c": "seed"}
    cache.put(
        seed,
        0,
        "cached-out-0",
        0.9,
        split=TRAINSET_CACHE_SPLIT,
        trajectory={"cached": 0},
    )
    state = _make_state(example_count=2, cache=cache)
    calls: list[list[tuple[dict[str, str], list]]] = []
    adapter = _make_adapter(calls)
    proposer = _make_proposer(adapter, trainset=[{"q": 0}, {"q": 1}], minibatch_ids=[0, 1])

    proposals = proposer.propose(state)

    # The parent stage fetches only example 1; the child stage evaluates both.
    assert calls[0] == [({"c": "seed"}, [{"q": 1}])]
    assert calls[1] == [({"c": "improved"}, [{"q": 0}, {"q": 1}])]
    assert proposals[0].subsample_scores_before == [0.9, 0.8]
    assert proposals[0].eval_before.trajectories == [
        {"cached": 0},
        {"candidate": "seed", "index": 0},
    ]
