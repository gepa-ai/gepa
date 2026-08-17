# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests that a separate valset does not share evaluation-cache keys with the trainset.

Data loaders identify examples by position, so a trainset and a distinct valset both number their
examples from zero. The evaluation cache is keyed by candidate and example id, the valset path reads
it and the minibatch path writes it, so without a namespace a valset lookup can be answered by the
rollout of the unrelated trainset example sitting at the same position.
"""

import collections
import contextlib
import json

import pytest

import gepa
from gepa.core.engine import GEPAEngine
from gepa.core.state import (
    TRAINSET_CACHE_SPLIT,
    VALSET_CACHE_SPLIT,
    CachedEvaluation,
    EvaluationCache,
    GEPAState,
    ValsetEvaluation,
    _candidate_hash,
)
from gepa.gepa_launcher import EngineConfig, GEPAConfig, MergeConfig, ReflectionConfig, optimize_anything

TRAIN = [{"content": f"train-{n}"} for n in range(12)]
VAL = [{"content": f"val-{n}"} for n in range(6)]


def _candidate_tag(candidate: dict[str, str]) -> str:
    return json.loads(next(iter(candidate.values())))["text"]


def _run(dataset, valset, produced_by):
    """Run a short optimization, recording which example produced each valset slot's score."""
    slot_of = {example["content"]: slot for slot, example in enumerate(valset or [])}

    def evaluator(candidate: str, example: dict[str, str]):
        tag = json.loads(candidate)["text"]
        content = example["content"]
        if content in slot_of:
            produced_by[tag][slot_of[content]] = content
        # Monotonically improving candidates, so children are accepted and trigger valset evals.
        score = 0.1 * (int(tag.split("-")[-1]) + 1 if tag != "seed" else 0)
        return score, {"ran": content}

    counter = iter(range(1000))
    optimize_anything(
        seed_candidate=json.dumps({"text": "seed"}),
        evaluator=evaluator,
        dataset=dataset,
        valset=valset,
        objective="valset cache split regression",
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=200,
                max_candidate_proposals=5,
                display_progress_bar=False,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=lambda _prompt: json.dumps({"text": f"cand-{next(counter)}"}),
                reflection_minibatch_size=4,
            ),
        ),
    )


def _record_cache_hits(monkeypatch, produced_by):
    """Attribute every cache hit to the example whose rollout was stored under that key."""
    get_batch = EvaluationCache.get_batch

    def instrumented(self, candidate, example_ids, *args, **kwargs):
        cached, uncached = get_batch(self, candidate, example_ids, *args, **kwargs)
        for slot, entry in cached.items():
            parts = entry.output if isinstance(entry.output, tuple) else (entry.output,)
            ran = next((part["ran"] for part in parts if isinstance(part, dict) and "ran" in part), "?")
            produced_by[_candidate_tag(candidate)][slot] = ran
        return cached, uncached

    monkeypatch.setattr(EvaluationCache, "get_batch", instrumented)


def test_valset_slots_are_never_filled_by_trainset_rollouts(monkeypatch):
    """Every valset score must come from the valset example that owns that slot."""
    produced_by = collections.defaultdict(dict)
    _record_cache_hits(monkeypatch, produced_by)

    _run(TRAIN, VAL, produced_by)

    assert produced_by, "no candidate was ever evaluated against the valset"
    leaked = {
        tag: {slot: content for slot, content in slots.items() if not content.startswith("val-")}
        for tag, slots in produced_by.items()
    }
    offenders = {tag: slots for tag, slots in leaked.items() if slots}
    assert not offenders, f"valset slots served by trainset rollouts: {offenders}"

    for tag, slots in produced_by.items():
        for slot, content in slots.items():
            assert content == VAL[slot]["content"], f"{tag} slot {slot} was produced by {content}"


def test_shared_valset_still_reuses_trainset_rollouts(monkeypatch):
    """When the valset is the trainset the ids agree, so caching must still be shared."""
    hits: list[int] = []
    get_batch = EvaluationCache.get_batch

    def instrumented(self, candidate, example_ids, *args, **kwargs):
        cached, uncached = get_batch(self, candidate, example_ids, *args, **kwargs)
        hits.append(len(cached))
        return cached, uncached

    monkeypatch.setattr(EvaluationCache, "get_batch", instrumented)
    _run(TRAIN, None, collections.defaultdict(dict))

    assert sum(hits) > 0, "valset evaluation stopped reusing minibatch rollouts of the same examples"


def test_cache_split_isolates_identical_example_ids():
    """Unit-level: the same id in two splits must not collide."""
    cache: EvaluationCache = EvaluationCache()
    candidate = {"text": "c"}

    cache.put(candidate, 0, "train rollout", 1.0, split=TRAINSET_CACHE_SPLIT)
    assert cache.get(candidate, 0, split=VALSET_CACHE_SPLIT) is None

    cache.put(candidate, 0, "val rollout", 0.0, split=VALSET_CACHE_SPLIT)
    assert cache.get(candidate, 0, split=TRAINSET_CACHE_SPLIT).output == "train rollout"
    assert cache.get(candidate, 0, split=VALSET_CACHE_SPLIT).output == "val rollout"


def test_drop_unsplit_entries_removes_unknown_splits():
    """A 3-tuple with an unknown split string is as unusable as a 2-tuple."""
    cache: EvaluationCache = EvaluationCache()
    candidate = {"text": "c"}
    cache.put(candidate, 0, "ok", 1.0, split=TRAINSET_CACHE_SPLIT)
    cache._cache[(_candidate_hash(candidate), "holdout", 0)] = CachedEvaluation("x", 0.0, None)

    assert cache.drop_unsplit_entries() == 1
    assert cache.get(candidate, 0, split=TRAINSET_CACHE_SPLIT).output == "ok"


def test_every_key_carries_its_split():
    """Including the trainset one: an unnamespaced key can never be served again."""
    cache: EvaluationCache = EvaluationCache()
    cache.put({"text": "c"}, 7, "rollout", 1.0, split=TRAINSET_CACHE_SPLIT)

    (key,) = cache._cache
    assert key[1] == TRAINSET_CACHE_SPLIT, f"trainset key is not namespaced: {key}"


def test_caches_written_before_splits_are_dropped_on_resume(tmp_path):
    """A pre-split cache is already contaminated, so resume must not carry it forward."""
    cache: EvaluationCache = EvaluationCache()
    candidate = {"text": "c"}
    cache.put(candidate, 0, "train rollout", 1.0, split=TRAINSET_CACHE_SPLIT)
    # What earlier versions wrote: no namespace, so a valset rollout landed on the trainset key.
    cache._cache[(_candidate_hash(candidate), 0)] = CachedEvaluation("val rollout", 0.0, None)

    state = _seeded_state(cache)
    state.save(str(tmp_path))
    reloaded = GEPAState.load(str(tmp_path))

    assert reloaded.evaluation_cache is not None
    assert all(EvaluationCache._is_split_key(key) for key in reloaded.evaluation_cache._cache)
    assert reloaded.evaluation_cache.get(candidate, 0, split=TRAINSET_CACHE_SPLIT).output == "train rollout"


def test_pure_pre_split_cache_is_emptied_on_resume(tmp_path):
    """A cache that is only 2-tuple keys has nothing safe to keep."""
    cache: EvaluationCache = EvaluationCache()
    candidate = {"text": "c"}
    cache._cache[(_candidate_hash(candidate), 0)] = CachedEvaluation("legacy rollout", 0.5, None)

    state = _seeded_state(cache)
    state.save(str(tmp_path))
    reloaded = GEPAState.load(str(tmp_path))

    assert reloaded.evaluation_cache is not None
    assert reloaded.evaluation_cache._cache == {}
    assert reloaded.evaluation_cache.get(candidate, 0, split=TRAINSET_CACHE_SPLIT) is None


def _seeded_state(cache: EvaluationCache) -> GEPAState:
    return GEPAState(
        {"text": "c"},
        ValsetEvaluation(outputs_by_val_id={0: "o"}, scores_by_val_id={0: 1.0}),
        evaluation_cache=cache,
    )


class _StubAdapter:
    """Never actually evaluated: the engine is intercepted before it runs."""

    def evaluate(self, batch, candidate, capture_traces=False):
        raise AssertionError("the engine should have been intercepted before evaluating")

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {name: [] for name in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        return dict.fromkeys(components_to_update, "x")


def _splits_used_by_optimize(monkeypatch, valset):
    """Return (engine split, merge split) that ``gepa.optimize`` wires up."""
    captured = {}

    def fake_run(self):
        captured["engine"] = self.valset_cache_split
        captured["merge"] = self.merge_proposer.valset_cache_split
        raise _Intercepted

    monkeypatch.setattr(GEPAEngine, "run", fake_run)
    with contextlib.suppress(_Intercepted):
        gepa.optimize(
            seed_candidate={"p": "seed"},
            trainset=TRAIN,
            valset=valset,
            adapter=_StubAdapter(),
            max_metric_calls=10,
            use_merge=True,
            cache_evaluation=True,
            display_progress_bar=False,
        )
    return captured["engine"], captured["merge"]


class _Intercepted(Exception):
    pass


def _splits_used_by_optimize_anything(monkeypatch, valset):
    """Return (engine split, merge split) that ``optimize_anything`` wires up."""
    captured = {}

    def fake_run(self):
        captured["engine"] = self.valset_cache_split
        captured["merge"] = None if self.merge_proposer is None else self.merge_proposer.valset_cache_split
        raise _Intercepted

    monkeypatch.setattr(GEPAEngine, "run", fake_run)
    with contextlib.suppress(_Intercepted):
        optimize_anything(
            seed_candidate=json.dumps({"text": "seed"}),
            evaluator=lambda candidate, example: (0.0, {}),
            dataset=TRAIN,
            valset=valset,
            objective="valset cache split wiring",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=10, display_progress_bar=False, cache_evaluation=True),
                reflection=ReflectionConfig(reflection_lm=lambda _prompt: json.dumps({"text": "x"})),
                merge=MergeConfig(max_merge_invocations=1),
            ),
        )
    return captured["engine"], captured["merge"]


@pytest.mark.parametrize(
    ("valset", "expected"),
    [
        (None, TRAINSET_CACHE_SPLIT),  # one loader for both: ids agree, keep sharing rollouts
        (TRAIN, TRAINSET_CACHE_SPLIT),  # same dataset passed twice: still one id space
        (VAL, VALSET_CACHE_SPLIT),  # distinct valset: must not read trainset rollouts
        (list(TRAIN), VALSET_CACHE_SPLIT),  # equal copy: different object, isolate
    ],
)
def test_optimize_gives_engine_and_merge_the_same_split(monkeypatch, valset, expected):
    """A split the merge proposer disagrees with silently re-runs cached valset examples."""
    engine_split, merge_split = _splits_used_by_optimize(monkeypatch, valset)

    assert engine_split == expected
    assert merge_split == engine_split, "merge proposer and engine disagree on the valset namespace"


@pytest.mark.parametrize(
    ("valset", "expected"),
    [
        (None, TRAINSET_CACHE_SPLIT),
        (TRAIN, TRAINSET_CACHE_SPLIT),
        (VAL, VALSET_CACHE_SPLIT),
        (list(TRAIN), VALSET_CACHE_SPLIT),
    ],
)
def test_optimize_anything_gives_engine_and_merge_the_same_split(monkeypatch, valset, expected):
    engine_split, merge_split = _splits_used_by_optimize_anything(monkeypatch, valset)

    assert engine_split == expected
    assert merge_split == engine_split, "merge proposer and engine disagree on the valset namespace"
