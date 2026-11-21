"""Regression tests for cached evaluation budget accounting (#316)."""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any

import pytest

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import (
    OptimizeAnythingAdapter,
)
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.core.callbacks import BudgetUpdatedEvent
from gepa.core.data_loader import ListDataLoader
from gepa.core.engine import GEPAEngine
from gepa.core.state import EvaluationCache, GEPAState, ValsetEvaluation, initialize_gepa_state
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import StdOutLogger
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import CurrentBestCandidateSelector
from gepa.strategies.component_selector import AllReflectionComponentSelector
from gepa.utils.stop_condition import MaxMetricCallsStopper


def _deterministic_evaluator_factory():
    real_calls = {"count": 0}

    def evaluator(candidate, example=None, **_kwargs):
        real_calls["count"] += 1
        digest = hashlib.sha256(f"{candidate}|{example}".encode()).digest()
        return digest[0] / 255.0, {"digest": digest[0]}

    return evaluator, real_calls


def _stub_reflection_lm():
    proposals = {"count": 0}

    def reflection_lm(_prompt: str) -> str:
        proposals["count"] += 1
        return f"```\ncandidate version {proposals['count']}\n```"

    return reflection_lm, proposals


def test_optimize_anything_cache_hits_not_charged_by_default():
    """Cache hits must not consume metric_calls budget when caching is enabled."""
    evaluator, real_calls = _deterministic_evaluator_factory()
    reflection_lm, _ = _stub_reflection_lm()
    data = [f"example {i}" for i in range(8)]

    result = optimize_anything(
        seed_candidate="candidate version 0",
        evaluator=evaluator,
        dataset=data,
        valset=data,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=40,
                cache_evaluation=True,
                cache_evaluation_storage="memory",
                parallel=False,
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
        ),
    )

    assert result.total_computed_metric_calls == real_calls["count"]
    assert result.total_metric_calls == real_calls["count"]
    assert result.total_cached_metric_calls is not None and result.total_cached_metric_calls > 0


def test_optimize_anything_opt_in_counts_cached_evals_toward_budget():
    """Opt-in flag restores counting cache hits toward the stopping budget."""
    evaluator, real_calls = _deterministic_evaluator_factory()
    reflection_lm, _ = _stub_reflection_lm()
    data = [f"example {i}" for i in range(8)]

    result = optimize_anything(
        seed_candidate="candidate version 0",
        evaluator=evaluator,
        dataset=data,
        valset=data,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=40,
                cache_evaluation=True,
                cache_evaluation_storage="memory",
                parallel=False,
                count_cached_evals_toward_budget=True,
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
        ),
    )

    assert result.total_computed_metric_calls == real_calls["count"]
    assert result.total_cached_metric_calls is not None and result.total_cached_metric_calls > 0
    assert result.total_metric_calls == result.total_computed_metric_calls + result.total_cached_metric_calls


def test_optimize_anything_batch_evaluator_cache_and_dedup_accounting():
    """batch_evaluator tracks misses vs cache hits + within-batch duplicates accurately."""
    real_calls = {"count": 0}

    def batch_evaluator(pairs):
        real_calls["count"] += len(pairs)
        return [(1.0, {"cand": c, "ex": e}) for c, e in pairs]

    adapter = OptimizeAnythingAdapter(
        evaluator=lambda c, e: (1.0, {}),
        batch_evaluator=batch_evaluator,
        parallel=False,
        cache_mode="memory",
    )

    # First call: 1 duplicate within-batch + 1 unique miss -> 1 computed, 1 cached
    b1 = adapter.batch_evaluate([({"p": "a"}, ["ex1"]), ({"p": "a"}, ["ex1"])])
    assert real_calls["count"] == 1
    assert b1[0].num_metric_calls == 1
    assert b1[0].num_cached_evals == 0
    assert b1[1].num_metric_calls == 0
    assert b1[1].num_cached_evals == 1

    # Second call: both hit cache -> 0 computed, 2 cached
    b2 = adapter.batch_evaluate([({"p": "a"}, ["ex1"]), ({"p": "a"}, ["ex1"])])
    assert real_calls["count"] == 1
    assert b2[0].num_metric_calls == 0
    assert b2[0].num_cached_evals == 1
    assert b2[1].num_metric_calls == 0
    assert b2[1].num_cached_evals == 1


def test_budget_updated_event_reports_computed_and_cached_deltas():
    """on_budget_updated receives both computed and cached totals/deltas."""
    evaluator, real_calls = _deterministic_evaluator_factory()
    reflection_lm, _ = _stub_reflection_lm()
    events: list[BudgetUpdatedEvent] = []

    class Recorder:
        def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
            events.append(event)

    data = [f"example {i}" for i in range(4)]
    result = optimize_anything(
        seed_candidate="candidate version 0",
        evaluator=evaluator,
        dataset=data,
        valset=data,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=25,
                cache_evaluation=True,
                cache_evaluation_storage="memory",
                parallel=False,
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
            callbacks=[Recorder()],
        ),
    )

    assert events
    last = events[-1]
    assert last["metric_calls_used"] == result.total_metric_calls == real_calls["count"]
    assert last["metric_calls_computed"] == result.total_computed_metric_calls == real_calls["count"]
    assert last["metric_calls_cached"] == result.total_cached_metric_calls
    assert sum(e["metric_calls_delta"] for e in events) + len(data) == result.total_metric_calls
    assert sum(e["metric_calls_cached_delta"] for e in events) == result.total_cached_metric_calls


class _CountingAdapter(GEPAAdapter[int, dict[str, Any], str]):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, batch: list[int], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        self.calls += len(batch)
        outputs = [f"{candidate['p']}:{x}" for x in batch]
        scores = [float(len(candidate["p"]) + x) for x in batch]
        trajectories = [{"x": x} for x in batch] if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {comp: [] for comp in components_to_update}


def test_state_level_evaluation_cache_excludes_hits_from_budget(tmp_path):
    """State-level EvaluationCache hits in valset and merge do not charge total_num_evals."""
    adapter = _CountingAdapter()
    loader = ListDataLoader([1, 2, 3])
    cache: EvaluationCache[str, int] = EvaluationCache()

    seed = {"p": "seed"}
    # Pre-warm cache for candidate {"p": "new"} on trainset split (shared loader)
    warm_cand = {"p": "new"}
    cache.put_batch(warm_cand, [0, 1, 2], ["w0", "w1", "w2"], [10.0, 11.0, 12.0], split="train")

    state = initialize_gepa_state(
        run_dir=None,
        logger=StdOutLogger(),
        seed_candidate=seed,
        seed_valset_evaluation=ValsetEvaluation(
            outputs_by_val_id={0: "s0", 1: "s1", 2: "s2"},
            scores_by_val_id={0: 1.0, 1: 2.0, 2: 3.0},
        ),
        evaluation_cache=cache,
    )
    assert state.total_num_evals == 3
    assert state.total_computed_evals == 3
    assert state.total_cached_evals == 0

    # Test GEPAEngine valset evaluation hitting state cache completely
    proposer = ReflectiveMutationProposer(
        logger=StdOutLogger(),
        trainset=loader,
        adapter=adapter,
        candidate_selector=CurrentBestCandidateSelector(),
        module_selector=AllReflectionComponentSelector(),
        batch_sampler=EpochShuffledBatchSampler(minibatch_size=1),
        perfect_score=100.0,
        skip_perfect_score=False,
        experiment_tracker=ExperimentTracker(),
        reflection_lm=lambda _prompt: "unused",
    )
    engine = GEPAEngine(
        adapter=adapter,
        run_dir=None,
        valset=loader,
        seed_candidate=seed,
        perfect_score=100.0,
        seed=0,
        reflective_proposer=proposer,
        merge_proposer=None,
        frontier_type="instance",
        logger=StdOutLogger(),
        experiment_tracker=ExperimentTracker(),
        evaluation_cache=cache,
    )
    val_eval, num_actual = engine._evaluate_programs_on_valset([warm_cand], state)[0]
    assert num_actual == 0
    assert val_eval.num_cached_evals == 3
    assert adapter.calls == 0

    state.full_program_trace.append({"i": 0, "iteration_id": "test0000"})
    engine._add_evaluated_program(warm_cand, state, [0], val_eval, num_actual)
    assert state.total_num_evals == 3
    assert state.total_computed_evals == 3
    assert state.total_cached_evals == 3

    # Test MergeProposer hitting state cache on 2 of 3 examples
    merged = {"p": "merged"}
    cache.put_batch(merged, [0, 1], ["m0", "m1"], [20.0, 21.0], split="train")
    state.prog_candidate_val_subscores[0] = {0: 1.0, 1: 2.0, 2: 3.0}
    state.prog_candidate_val_subscores[1] = {0: 10.0, 1: 11.0, 2: 12.0}
    merge_proposer = MergeProposer(
        logger=StdOutLogger(),
        valset=loader,
        evaluator=lambda batch, prog: (
            [f"{prog['p']}:{x}" for x in batch],
            [5.0 for _ in batch],
            None,
        ),
        use_merge=True,
        max_merge_invocations=5,
        rng=__import__("random").Random(0),
        val_overlap_floor=1,
    )
    merge_proposer.valset_cache_split = "train"
    # Direct test of state.cached_evaluate_full + increment_evals accounting on merge path
    _, _, _, merge_actual = state.cached_evaluate_full(
        merged, [0, 1, 2], loader.fetch, merge_proposer.evaluator, split="train"
    )
    assert merge_actual == 1
    state.increment_evals(merge_actual, cached=3 - merge_actual)
    assert state.total_num_evals == 4
    assert state.total_computed_evals == 4
    assert state.total_cached_evals == 5


def test_legacy_state_pickle_loads_with_safe_eval_counter_defaults(tmp_path):
    """Loading a gepa_state.bin without new counter attributes falls back cleanly."""
    state = GEPAState(
        {"p": "seed"},
        ValsetEvaluation(outputs_by_val_id={0: "o"}, scores_by_val_id={0: 1.0}),
    )
    state.total_num_evals = 7
    data = dict(state.__dict__)
    for key in ("total_computed_evals", "total_cached_evals", "count_cached_evals_toward_budget"):
        data.pop(key, None)

    with open(os.path.join(tmp_path, "gepa_state.bin"), "wb") as f:
        pickle.dump(data, f)

    loaded = GEPAState.load(str(tmp_path))
    assert loaded.total_num_evals == 7
    assert loaded.total_computed_evals == 7
    assert loaded.total_cached_evals == 0
    assert loaded.count_cached_evals_toward_budget is False

    loaded.increment_evals(2, cached=3)
    assert loaded.total_num_evals == 9
    assert loaded.total_computed_evals == 9
    assert loaded.total_cached_evals == 3


def test_cached_eval_only_iteration_still_triggers_no_progress_guard_on_exception():
    """Exception after a cache-only parent eval must re-raise rather than loop forever."""

    class _CachedHitAdapter(_CountingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.seen: set[tuple[str, int]] = set()

        def evaluate(self, batch: list[int], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
            outputs: list[str] = []
            scores: list[float] = []
            trajectories = [] if capture_traces else None
            cached = 0
            for x in batch:
                key = (candidate["p"], x)
                if key in self.seen:
                    cached += 1
                else:
                    self.seen.add(key)
                    self.calls += 1
                outputs.append(f"{candidate['p']}:{x}")
                scores.append(float(len(candidate["p"]) + x))
                if trajectories is not None:
                    trajectories.append({"x": x})
            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
                num_metric_calls=len(batch) - cached,
                num_cached_evals=cached,
            )

    def failing_module_selector(_state, _trajectories, _scores, _parent_idx, _parent_candidate):
        raise RuntimeError("selector failure after cache hit")

    adapter = _CachedHitAdapter()
    loader = ListDataLoader([0])
    seed = {"p": "seed"}
    proposer = ReflectiveMutationProposer(
        logger=StdOutLogger(),
        trainset=loader,
        adapter=adapter,
        candidate_selector=CurrentBestCandidateSelector(),
        module_selector=failing_module_selector,  # type: ignore[arg-type]
        batch_sampler=EpochShuffledBatchSampler(minibatch_size=1),
        perfect_score=100.0,
        skip_perfect_score=False,
        experiment_tracker=ExperimentTracker(),
        reflection_lm=lambda _prompt: "unused",
    )
    engine = GEPAEngine(
        adapter=adapter,
        run_dir=None,
        valset=loader,
        seed_candidate=seed,
        perfect_score=100.0,
        seed=0,
        reflective_proposer=proposer,
        merge_proposer=None,
        frontier_type="instance",
        logger=StdOutLogger(),
        experiment_tracker=ExperimentTracker(),
        raise_on_exception=False,
        stop_callback=MaxMetricCallsStopper(5),
        count_cached_evals_toward_budget=False,
    )

    with pytest.raises(RuntimeError, match="selector failure after cache hit"):
        engine.run()

    assert adapter.calls == 1
