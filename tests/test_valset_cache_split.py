# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests that a separate valset does not share evaluation-cache keys with the trainset.

Data loaders identify examples by position, so a trainset and a distinct valset both number their
examples from zero. The evaluation cache is keyed by candidate and example id, the valset path reads
it and the minibatch path writes it, so without a namespace a valset lookup can be answered by the
rollout of the unrelated trainset example sitting at the same position.
"""

import collections
import json

from gepa.core.state import TRAINSET_CACHE_SPLIT, VALSET_CACHE_SPLIT, EvaluationCache
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

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


def test_default_split_keeps_the_legacy_key_shape():
    """Caches persisted before splits existed stay readable on resume."""
    cache: EvaluationCache = EvaluationCache()
    cache.put({"text": "c"}, 7, "rollout", 1.0)

    (key,) = cache._cache
    assert len(key) == 2, f"default split changed the persisted key shape: {key}"
