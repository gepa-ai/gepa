# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for ComBEEReflectionLM (#307 re-hosted on the ReflectionLM seam, #329 Phase 3).

The characterization tests in this file were first validated against the
ORIGINAL PR #307 implementation (`_propose_new_texts_combee`), then against
this port — with a fixed RNG seed and fake LM, the two produce byte-identical
LM-call sequences and final texts across all scenarios, so these tests pin
behavior preserved from the original contribution by @nuglifeleoji.
"""

import json
import random
import re
from collections import defaultdict
from typing import ClassVar

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.proposer.reflective_mutation.combee import (
    DEFAULT_AGGREGATION_PROMPT_TEMPLATE,
    ComBEEReflectionLM,
)
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionLM

RECORDS9 = [{"Inputs": f"q{i}", "Generated Outputs": f"a{i}", "Feedback": f"fb{i}"} for i in range(9)]
RECORDS3 = RECORDS9[:3]

AGG_MARKER = "synthesize these proposed instruction updates"
DEFAULT_TEMPLATE_MARKER = "I provided an assistant with the following instructions to perform a task"


class ScriptedLM:
    """Fake reflection LM: returns scripted replies in call order, logs prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        return f"```\n{self.replies[len(self.prompts) - 1]}\n```"


class CollectingLogger:
    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def _combee(replies, *, seed=0, p=2, template=None, agg_template=None, logger=None):
    lm = ScriptedLM(replies)
    combee = ComBEEReflectionLM(
        lm,
        reflection_prompt_template=template,
        aggregation_prompt_template=agg_template,
        duplication_factor=p,
        rng=random.Random(seed),
        logger=logger,
    )
    return combee, lm


# ---------------------------------------------------------------------------
# Characterization (validated against the original PR #307 implementation)
# ---------------------------------------------------------------------------


def test_map_reduce_structure_n9():
    """n=9 -> k=3 map calls with the standard template + 1 reduce call with the
    aggregation template; the reduce sees every intermediate proposal."""
    combee, lm = _combee([f"map-{i}" for i in range(3)] + ["final-synthesis"])
    proposal, next_lm = combee.reflect({"comp": "SEED INSTRUCTION"}, {"comp": RECORDS9}, ["comp"])

    assert next_lm is combee
    assert len(lm.prompts) == 4
    assert proposal.new_texts == {"comp": "final-synthesis"}
    map_prompts, reduce_prompt = lm.prompts[:3], lm.prompts[3]
    assert all(DEFAULT_TEMPLATE_MARKER in p for p in map_prompts)
    assert all(AGG_MARKER not in p for p in map_prompts)
    assert AGG_MARKER in reduce_prompt
    assert all(f"map-{i}" in reduce_prompt for i in range(3))
    assert "SEED INSTRUCTION" in reduce_prompt
    # Raw feedback records never leak into the reduce prompt.
    assert all(f"fb{i}" not in reduce_prompt for i in range(9))


def test_augmented_shuffle_duplicates_each_record_p_times():
    combee, lm = _combee(["m0", "m1", "m2", "fin"], p=2)
    combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    for i in range(9):
        assert sum(p.count(f"fb{i}") for p in lm.prompts[:3]) == 2

    combee, lm = _combee(["m0", "m1", "m2", "fin"], p=1)
    combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    for i in range(9):
        assert sum(p.count(f"fb{i}") for p in lm.prompts[:3]) == 1


def test_degenerate_n3_falls_back_to_single_call_with_log():
    logger = CollectingLogger()
    combee, lm = _combee(["naive-out"], logger=logger)
    proposal, _ = combee.reflect({"comp": "INSTR3"}, {"comp": RECORDS3}, ["comp"])

    assert len(lm.prompts) == 1
    assert DEFAULT_TEMPLATE_MARKER in lm.prompts[0]
    assert AGG_MARKER not in lm.prompts[0]
    assert proposal.new_texts == {"comp": "naive-out"}
    # Fallback does not duplicate records.
    assert all(lm.prompts[0].count(f"fb{i}") == 1 for i in range(3))
    # The old implementation fell back SILENTLY; the port logs it (review fix).
    assert any("falling back to standard single-call reflection" in m for m in logger.messages)
    # Diagnostics recorded even on the fallback path.
    assert proposal.prompts["comp"]
    assert proposal.raw_lm_outputs["comp"]
    assert proposal.metadata["combee:comp:mode"] == "fallback_single_call"


def test_multi_component_mixed_sizes():
    combee, lm = _combee(["c1m0", "c1m1", "c1m2", "c1final", "c2naive"])
    proposal, _ = combee.reflect({"c1": "I1", "c2": "I2"}, {"c1": RECORDS9, "c2": RECORDS3}, ["c1", "c2"])
    assert len(lm.prompts) == 5
    assert proposal.new_texts == {"c1": "c1final", "c2": "c2naive"}
    assert proposal.metadata["combee:c1:mode"] == "map_reduce"
    assert proposal.metadata["combee:c1:num_lm_calls"] == 4
    assert proposal.metadata["combee:total_lm_calls"] == 5


def test_component_missing_from_dataset_or_candidate_is_skipped():
    logger = CollectingLogger()
    combee, _lm = _combee(["m0", "m1", "m2", "fin"], logger=logger)
    proposal, _ = combee.reflect({"c1": "I1"}, {"c1": RECORDS9, "orphan": RECORDS9}, ["c1", "ghost", "orphan"])
    assert set(proposal.new_texts) == {"c1"}
    assert any("'ghost' is not in reflective dataset" in m for m in logger.messages)
    assert any("'orphan' is missing from candidate" in m for m in logger.messages)


def test_custom_aggregation_template_used_for_reduce_only():
    combee, lm = _combee(["m0", "m1", "m2", "fin"], agg_template="MY AGGREGATOR: <curr_param> || <side_info>")
    combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert "MY AGGREGATOR" in lm.prompts[3]
    assert AGG_MARKER not in lm.prompts[3]
    assert all("MY AGGREGATOR" not in p for p in lm.prompts[:3])


def test_custom_level1_template_used_for_maps_only():
    combee, lm = _combee(["m0", "m1", "m2", "fin"], template="LEVEL1 TEMPLATE <curr_param> :: <side_info>")
    combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert all("LEVEL1 TEMPLATE" in p for p in lm.prompts[:3])
    assert "LEVEL1 TEMPLATE" not in lm.prompts[3]


def test_per_component_template_dict_with_warn_once():
    logger = CollectingLogger()
    replies = ["a_m0", "a_m1", "a_m2", "a_fin", "b_m0", "b_m1", "b_m2", "b_fin"] * 2
    combee, lm = _combee(replies, template={"a": "TEMPLATE-A <curr_param> <side_info>"}, logger=logger)
    combee.reflect({"a": "IA", "b": "IB"}, {"a": RECORDS9, "b": RECORDS9}, ["a", "b"])
    combee.reflect({"a": "IA", "b": "IB"}, {"a": RECORDS9, "b": RECORDS9}, ["a", "b"])
    a_maps = lm.prompts[:3]
    assert all("TEMPLATE-A" in p for p in a_maps)
    warnings = [m for m in logger.messages if "No reflection_prompt_template found for parameter 'b'" in m]
    assert len(warnings) == 1  # warn once, mirroring StatelessReflectionLM


def test_deterministic_and_seed_sensitive():
    replies = ["m0", "m1", "m2", "fin"]
    combee_a, lm_a = _combee(replies, seed=42)
    combee_a.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    combee_b, lm_b = _combee(replies, seed=42)
    combee_b.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    combee_c, lm_c = _combee(replies, seed=7)
    combee_c.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert lm_a.prompts == lm_b.prompts
    assert lm_a.prompts != lm_c.prompts


# ---------------------------------------------------------------------------
# Seam integration and new-in-port behavior
# ---------------------------------------------------------------------------


def test_protocol_conformance():
    combee, _ = _combee(["x"])
    assert isinstance(combee, ReflectionLM)


def test_metadata_records_level1_intermediates():
    combee, _ = _combee(["m0", "m1", "m2", "fin"])
    proposal, _ = combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert proposal.metadata["combee:comp:k"] == 3
    assert len(proposal.metadata["combee:comp:level1_prompts"]) == 3
    assert [o.strip("`\n") for o in proposal.metadata["combee:comp:level1_outputs"]] == ["m0", "m1", "m2"]
    assert proposal.metadata["combee:comp:num_lm_calls"] == 4


def test_plain_callable_tracks_tokens_but_not_provider_cost():
    """TrackingLM supplies token diagnostics, but its zero cost is not a
    provider-spend signal suitable for a max-reflection-cost stopper."""
    combee, _ = _combee(["m0", "m1", "m2", "fin"])
    assert hasattr(combee, "total_cost")
    assert not combee.supports_cost_tracking()
    combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert combee.total_tokens_in > 0
    assert combee.total_tokens_out > 0

    from gepa.lm import TrackingLM

    assert not ComBEEReflectionLM(TrackingLM(lambda prompt: "update")).supports_cost_tracking()


def test_str_model_name_resolved_via_gepa_lm():
    """A model-name string constructs a gepa.lm.LM with cost tracking wired."""
    from gepa.lm import LM

    combee = ComBEEReflectionLM("openai/gpt-4.1-mini")
    assert isinstance(combee.lm, LM)
    assert combee.total_cost == 0.0
    assert combee.total_tokens_in == 0


def test_model_name_binds_public_lm_kwargs_unless_constructor_overrides(monkeypatch):
    import gepa.lm

    _KwargRecordingLM.instances.clear()
    monkeypatch.setattr(gepa.lm, "LM", _KwargRecordingLM)

    combee = ComBEEReflectionLM("test/model")
    assert combee.lm.kwargs == {}
    combee.bind_lm_kwargs({"temperature": 0.25, "max_tokens": 77})
    assert combee.lm.kwargs == {"temperature": 0.25, "max_tokens": 77}

    explicit = ComBEEReflectionLM("test/model", lm_kwargs={"temperature": 0.9})
    explicit.bind_lm_kwargs({"temperature": 0.25, "max_tokens": 77})
    assert explicit.lm.kwargs == {"temperature": 0.9}


def test_strategy_template_binding_uses_public_value_unless_explicit():
    public_template = "PUBLIC TEMPLATE <curr_param> :: <side_info>"
    explicit_template = "EXPLICIT TEMPLATE <curr_param> :: <side_info>"

    combee, lm = _combee(["m0", "m1", "m2", "final"])
    combee.bind_reflection_prompt_template(public_template)
    combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert all("PUBLIC TEMPLATE" in prompt for prompt in lm.prompts[:3])

    explicit, explicit_lm = _combee(["m0", "m1", "m2", "final"], template=explicit_template)
    explicit.bind_reflection_prompt_template(public_template)
    explicit.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert all("EXPLICIT TEMPLATE" in prompt for prompt in explicit_lm.prompts[:3])


def test_duplication_factor_validation():
    with pytest.raises(ValueError, match="duplication_factor"):
        ComBEEReflectionLM(lambda p: "x", duplication_factor=0)


def test_aggregation_template_placeholders_validated():
    with pytest.raises(Exception):  # noqa: B017 — validate_prompt_template's error type
        ComBEEReflectionLM(lambda p: "x", aggregation_prompt_template="no placeholders here")


def test_default_aggregation_template_has_placeholders():
    assert "<curr_param>" in DEFAULT_AGGREGATION_PROMPT_TEMPLATE
    assert "<side_info>" in DEFAULT_AGGREGATION_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# End-to-end: gepa.optimize and optimize_anything drive ComBEE
# ---------------------------------------------------------------------------

TRAIN = [{"k": i} for i in range(10)]


def _level_of(candidate):
    m = re.search(r"LEVEL=(\d+)", candidate["system_prompt"])
    return int(m.group(1)) if m else 0


class SyntheticAdapter:
    """score(candidate, example k) = 1.0 iff LEVEL >= k; one reflective record
    per example, so reflection_minibatch_size controls ComBEE's n."""

    propose_new_texts = None

    def __init__(self):
        self.metric_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        level = _level_of(candidate)
        self.metric_calls += len(batch)
        return EvaluationBatch(
            outputs=[f"out-{ex['k']}" for ex in batch],
            scores=[1.0 if level >= ex["k"] else 0.0 for ex in batch],
            trajectories=(
                [{"k": ex["k"], "feedback": f"level={level} needed k={ex['k']}"} for ex in batch]
                if capture_traces
                else None
            ),
            objective_scores=None,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(self, candidate, eval_batch, components):
        return {c: [{"Feedback": t["feedback"]} for t in (eval_batch.trajectories or [])] for c in components}


class ImprovingLM:
    """Reflection LM whose proposals strictly improve: LEVEL = max seen + 1."""

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.calls += 1
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.prompts.append(text)
        levels = [int(x) for x in re.findall(r"LEVEL=(\d+)", text)]
        return f"```\nLEVEL={max(levels, default=0) + 1}\n```"


class _KwargRecordingLM:
    """Cost-tracking fake used to assert model-name strategy wiring without
    making provider calls."""

    instances: ClassVar[list["_KwargRecordingLM"]] = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.prompts: list[str] = []
        self.instances.append(self)

    def __call__(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.prompts.append(text)
        levels = [int(x) for x in re.findall(r"LEVEL=(\d+)", text)]
        return f"```\nLEVEL={max(levels, default=0) + 1}\n```"


class _ProposalSpy:
    """Selection strategy spy: records proposals, then defers to default filtering."""

    def __init__(self):
        self.proposals = []

    def select(self, proposals, state, acceptance_criterion):
        self.proposals.extend(proposals)
        return [p for p in proposals if acceptance_criterion.should_accept(p, state)]


def test_e2e_gepa_optimize_with_combee(tmp_path):
    lm = ImprovingLM()
    combee = ComBEEReflectionLM(lm, rng=random.Random(0))
    spy = _ProposalSpy()
    log = defaultdict(list)

    class Cb:
        def on_proposal_end(self, event):
            log["proposal_end"].append(dict(event))

    result = gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=SyntheticAdapter(),
        reflection_strategy=combee,
        reflection_minibatch_size=9,  # n=9 -> k=3 -> 4 LM calls per proposal
        selection_strategy=spy,
        max_metric_calls=120,
        callbacks=[Cb()],
        run_dir=str(tmp_path / "combee-e2e"),
        seed=0,
        display_progress_bar=False,
    )

    n_proposals = len(log["proposal_end"])
    assert n_proposals > 0
    # Exactly k+1 = 4 reflection calls per proposal — ComBEE's cost contract.
    assert lm.calls == 4 * n_proposals
    # ComBEE diagnostics flow into both CandidateProposal.metadata and the
    # proposal-end callback.
    assert spy.proposals
    md = spy.proposals[0].metadata
    assert md["combee:system_prompt:mode"] == "map_reduce"
    assert md["combee:system_prompt:num_lm_calls"] == 4
    assert md["combee:total_lm_calls"] == 4
    assert log["proposal_end"][0]["metadata"]["combee:system_prompt:mode"] == "map_reduce"
    assert log["proposal_end"][0]["metadata"]["combee:total_lm_calls"] == 4
    # The optimization actually improved over the seed.
    assert result.val_aggregate_scores[result.best_idx] > result.val_aggregate_scores[0]


def test_e2e_optimize_anything_with_combee(tmp_path):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    lm = ImprovingLM()
    combee = ComBEEReflectionLM(lm, rng=random.Random(0))

    def evaluator(candidate, example=None):
        level = _level_of({"system_prompt": candidate})
        return (1.0 if level >= example else 0.0), {"feedback": f"level={level} needed k={example}"}

    result = optimize_anything(
        seed_candidate="LEVEL=0",
        evaluator=evaluator,
        dataset=list(range(10)),
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=120,
                run_dir=str(tmp_path / "combee-oa"),
                parallel=False,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(reflection_strategy=combee, reflection_minibatch_size=9),
        ),
    )
    assert lm.calls > 0
    assert lm.calls % 4 == 0  # every proposal used k+1 = 4 calls
    assert result.best_candidate is not None


def test_e2e_combee_with_pxn_sampling(tmp_path):
    """ComBEE x PxNSampling: multi-task iterations route through ComBEE's
    two-wave ``reflect_many`` implementation, so every attempted proposal
    costs exactly k+1 reflection calls and carries independent diagnostics."""
    from gepa.strategies.proposal_sampling import PxNSampling

    def run(seed, tag):
        lm = ImprovingLM()
        adapter = SyntheticAdapter()
        spy = _ProposalSpy()
        events = defaultdict(list)

        class Cb:
            def on_proposal_end(self, event):
                events["proposal_end"].append(dict(event))

            def on_budget_updated(self, event):
                events["budget"].append(dict(event))

        result = gepa.optimize(
            seed_candidate={"system_prompt": "LEVEL=0"},
            trainset=TRAIN,
            valset=TRAIN,
            adapter=adapter,
            reflection_strategy=ComBEEReflectionLM(lm, rng=random.Random(0)),
            reflection_minibatch_size=9,  # n=9 -> k=3 -> 4 calls per task
            sampling_strategy=PxNSampling(p=2, n=2),  # 4 tasks per iteration
            selection_strategy=spy,
            max_metric_calls=350,
            callbacks=[Cb()],
            run_dir=str(tmp_path / f"combee-pxn-{tag}"),
            seed=seed,
            display_progress_bar=False,
        )
        return result, lm, adapter, spy, events

    result, lm, adapter, spy, events = run(0, "a")

    n_proposals = len(events["proposal_end"])
    assert n_proposals > 0
    # Cost contract holds per attempted proposal, exactly as in single-task mode.
    assert lm.calls == 4 * n_proposals

    # PxN actually fanned out: some iteration attempted more than one proposal.
    per_iter = defaultdict(int)
    for e in events["proposal_end"]:
        per_iter[e["iteration"]] += 1
    assert max(per_iter.values()) > 1, f"PxN never produced a multi-proposal iteration: {dict(per_iter)}"

    # Every evaluated proposal carries its own, non-aliased ComBEE diagnostics.
    assert spy.proposals
    metadata_ids = {id(p.metadata) for p in spy.proposals}
    assert len(metadata_ids) == len(spy.proposals), "proposal metadata dicts are aliased"
    for p in spy.proposals:
        assert p.metadata["combee:system_prompt:num_lm_calls"] == 4
        assert p.metadata["combee:system_prompt:mode"] == "map_reduce"
        assert p.metadata["combee:total_lm_calls"] == 4
    proposal_ids = [p.metadata["proposal_id"] for p in spy.proposals]
    assert len(set(proposal_ids)) == len(proposal_ids), f"proposal_ids not unique: {proposal_ids}"

    # Budget integrity under the combination.
    assert result.total_metric_calls == adapter.metric_calls
    budget = events["budget"]
    assert budget[-1]["metric_calls_used"] == adapter.metric_calls

    # The run actually improved over the seed.
    assert result.val_aggregate_scores[result.best_idx] > result.val_aggregate_scores[0]

    # Determinism: identical config reproduces the run exactly (ComBEE's
    # internal shuffle RNG advances per task in deterministic task order).
    result2, lm2, _adapter2, _spy2, events2 = run(0, "b")
    assert result2.best_candidate == result.best_candidate
    assert lm2.calls == lm.calls
    assert len(events2["proposal_end"]) == n_proposals

    # Resume compatibility: re-running with the SAME run_dir loads the finished
    # state and stops on the exhausted budget without any new reflection calls.
    result3, lm3, _adapter3, _spy3, _events3 = run(0, "a")
    assert lm3.calls == 0
    assert result3.best_candidate == result.best_candidate


# ---------------------------------------------------------------------------
# reflect_many: two-wave batching
# ---------------------------------------------------------------------------


class BatchScriptedLM(ScriptedLM):
    """ScriptedLM that also exposes batch_complete, recording wave sizes."""

    def __init__(self, replies):
        super().__init__(replies)
        self.batch_sizes: list[int] = []

    def batch_complete(self, messages_list):
        self.batch_sizes.append(len(messages_list))
        out = []
        for messages in messages_list:
            self.prompts.append(str(messages[0]["content"]) if isinstance(messages, list) else str(messages))
            out.append(f"```\n{self.replies[len(self.prompts) - 1]}\n```")
        return out


def test_reflect_many_batches_in_two_waves():
    """All map/fallback calls across all jobs go out as ONE batched round,
    all reduce calls as a second — with per-job results and metadata intact."""
    from gepa.proposer.reflective_mutation.reflection_lm import BatchReflectionLM

    lm = BatchScriptedLM(["j0m0", "j0m1", "j0m2", "j1naive", "j0final"])
    combee = ComBEEReflectionLM(lm, rng=random.Random(0))
    assert isinstance(combee, BatchReflectionLM)

    jobs = [
        ({"comp": "I0"}, {"comp": RECORDS9}, ["comp"]),  # k=3 map/reduce
        ({"comp": "I1"}, {"comp": RECORDS3}, ["comp"]),  # k=1 fallback
    ]
    results = combee.reflect_many(jobs)
    assert len(results) == 2
    p0, p1 = results[0][0], results[1][0]
    assert all(next_lm is combee for _, next_lm in results)

    # Wave structure: 4 wave-1 calls (3 maps + 1 fallback), 1 wave-2 call.
    assert lm.batch_sizes == [4]  # wave 2 has one prompt -> plain call path
    assert len(lm.prompts) == 5
    assert p0.new_texts == {"comp": "j0final"}
    assert p0.metadata["combee:comp:mode"] == "map_reduce"
    assert p0.metadata["combee:comp:num_lm_calls"] == 4
    assert p0.metadata["combee:total_lm_calls"] == 4
    assert p1.new_texts == {"comp": "j1naive"}
    assert p1.metadata["combee:comp:mode"] == "fallback_single_call"
    assert p1.metadata["combee:total_lm_calls"] == 1
    # The reduce prompt (last) uses the aggregation template and sees j0's maps.
    assert AGG_MARKER in lm.prompts[-1]
    assert all(f"j0m{i}" in lm.prompts[-1] for i in range(3))


class ContentKeyedLM:
    """Deterministic LM whose reply depends only on the prompt content — makes
    results order-independent, so sequential and batched paths are comparable."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        text = prompt if isinstance(prompt, str) else str(prompt)
        return f"```\nR{hash(text) % 99991}\n```"


def test_reflect_many_equivalent_to_sequential_reflect():
    """reflect_many(jobs) must produce the same proposals as sequential
    reflect() calls in job order (same seed): identical texts, prompts, raw
    outputs, and metadata — only the global call ORDER differs."""
    jobs = [
        ({"a": "IA", "b": "IB"}, {"a": RECORDS9, "b": RECORDS3}, ["a", "b"]),
        ({"c": "IC"}, {"c": RECORDS9}, ["c"]),
    ]

    batched = ComBEEReflectionLM(ContentKeyedLM(), rng=random.Random(5))
    batched_results = [prop for prop, _ in batched.reflect_many([tuple(j) for j in jobs])]

    sequential = ComBEEReflectionLM(ContentKeyedLM(), rng=random.Random(5))
    sequential_results = []
    for job in jobs:
        prop, _ = sequential.reflect(*job)
        sequential_results.append(prop)

    for bp, sp in zip(batched_results, sequential_results, strict=True):
        assert bp.new_texts == sp.new_texts
        assert bp.prompts == sp.prompts
        assert bp.raw_lm_outputs == sp.raw_lm_outputs
        assert bp.metadata == sp.metadata


def test_e2e_pxn_with_batch_capable_lm(tmp_path):
    """ComBEE x PxN with a batch-capable LM: each iteration's reflection goes
    out as two waves (wave1 = 3 maps per job, wave2 = 1 reduce per job), and
    the per-proposal cost contract is unchanged."""
    from gepa.strategies.proposal_sampling import PxNSampling

    class BatchImprovingLM(ImprovingLM):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def batch_complete(self, messages_list):
            self.batch_sizes.append(len(messages_list))
            return [self(messages[0]["content"]) for messages in messages_list]

    lm = BatchImprovingLM()
    events = defaultdict(list)

    class Cb:
        def on_proposal_end(self, event):
            events["proposal_end"].append(dict(event))

    result = gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=SyntheticAdapter(),
        reflection_strategy=ComBEEReflectionLM(lm, rng=random.Random(0)),
        reflection_minibatch_size=9,
        sampling_strategy=PxNSampling(p=2, n=2),
        max_metric_calls=350,
        callbacks=[Cb()],
        run_dir=str(tmp_path / "combee-pxn-batched"),
        seed=0,
        display_progress_bar=False,
    )
    n_proposals = len(events["proposal_end"])
    assert n_proposals > 0
    assert lm.calls == 4 * n_proposals  # cost contract unchanged
    # Two-wave structure: wave1 batches are 3x their iteration's wave2 batch.
    assert lm.batch_sizes, "batch_complete was never used"
    waves = list(lm.batch_sizes)
    # Reconstruct (wave1, wave2) pairs; single-prompt waves use the plain call
    # path and don't appear, which only happens for wave2 when 1 job survived.
    i = 0
    while i < len(waves):
        w1 = waves[i]
        assert w1 % 3 == 0, f"wave1 size {w1} not a multiple of k=3 at index {i}"
        n_jobs = w1 // 3
        if n_jobs > 1:
            assert i + 1 < len(waves) and waves[i + 1] == n_jobs, f"expected wave2={n_jobs} after wave1={w1}: {waves}"
            i += 2
        else:
            i += 1  # wave2 for a single job went through the plain-call path
    assert result.val_aggregate_scores[result.best_idx] > result.val_aggregate_scores[0]


def test_reflect_many_handles_duplicate_component_names():
    """Pathological but legal: the same component listed twice must behave
    exactly like sequential reflect() — second occurrence overwrites, both
    occurrences' calls happen, RNG stream matches."""
    job = ({"a": "IA"}, {"a": RECORDS9}, ["a", "a"])
    seq = ComBEEReflectionLM(ContentKeyedLM(), rng=random.Random(5))
    sp, _ = seq.reflect(*job)
    bat = ComBEEReflectionLM(ContentKeyedLM(), rng=random.Random(5))
    bp, _ = bat.reflect_many([job])[0]
    assert bp.new_texts == sp.new_texts
    assert bp.prompts == sp.prompts
    assert bp.raw_lm_outputs == sp.raw_lm_outputs
    assert bp.metadata == sp.metadata


def test_pxn_1x1_equivalent_to_single_mutation_sampling(tmp_path):
    """PxNSampling(1,1) consumes the candidate selector and batch sampler in
    the same order as SingleMutationSampling, so a ComBEE run under either
    produces identical results, prompts, and event streams."""
    from gepa.strategies.proposal_sampling import PxNSampling, SingleMutationSampling

    def run(sampling, tag):
        lm = ContentKeyedLM()
        events = []

        class Cb:
            def __getattr__(self, name):
                if name.startswith("on_"):

                    def rec(e, n=name):
                        events.append(n)

                    return rec
                raise AttributeError(name)

        result = gepa.optimize(
            seed_candidate={"system_prompt": "LEVEL=0 SEED"},
            trainset=TRAIN,
            valset=TRAIN,
            adapter=SyntheticAdapter(),
            reflection_strategy=ComBEEReflectionLM(lm, rng=random.Random(0)),
            reflection_minibatch_size=9,
            sampling_strategy=sampling,
            max_metric_calls=120,
            callbacks=[Cb()],
            run_dir=str(tmp_path / tag),
            seed=0,
            display_progress_bar=False,
        )
        return result, lm, events

    r_single, lm_single, ev_single = run(SingleMutationSampling(), "single")
    r_pxn, lm_pxn, ev_pxn = run(PxNSampling(p=1, n=1), "pxn11")

    assert r_pxn.best_candidate == r_single.best_candidate
    assert r_pxn.total_metric_calls == r_single.total_metric_calls
    assert len(r_pxn.candidates) == len(r_single.candidates)
    assert lm_pxn.calls == lm_single.calls
    assert ev_pxn == ev_single


# ---------------------------------------------------------------------------
# Review-round fixes: validation, empty template, RNG binding/restore, oa cap
# ---------------------------------------------------------------------------


def test_duplication_factor_must_be_integer():
    with pytest.raises(ValueError, match="integer"):
        ComBEEReflectionLM(lambda p: "x", duplication_factor=1.5)
    with pytest.raises(ValueError, match="integer"):
        ComBEEReflectionLM(lambda p: "x", duplication_factor=True)


def test_empty_aggregation_template_is_rejected_not_defaulted():
    with pytest.raises(Exception, match="curr_param|side_info|template"):
        ComBEEReflectionLM(lambda p: "x", aggregation_prompt_template="")


def test_bind_rng_makes_default_shuffles_seed_sensitive(tmp_path):
    """With the default RNG, ``gepa.optimize(seed=...)`` binds ComBEE to the
    engine stream: changing the seed changes shuffles, while a fixed seed
    reproduces them."""

    def run(seed, tag):
        lm = ScriptedLM(["improved"] * 40)
        combee = ComBEEReflectionLM(lm)  # default rng -> bind_rng applies
        gepa.optimize(
            seed_candidate={"system_prompt": "LEVEL=0 SEED"},
            trainset=TRAIN,
            valset=TRAIN,
            adapter=SyntheticAdapter(),
            reflection_strategy=combee,
            reflection_minibatch_size=9,
            max_metric_calls=80,
            run_dir=str(tmp_path / tag),
            seed=seed,
            display_progress_bar=False,
        )
        return lm.prompts  # map prompts embed the shuffle order

    a1 = run(0, "s0a")
    a2 = run(0, "s0b")
    b = run(1, "s1")
    assert a1, "no reflection prompts recorded"
    assert a1 == a2, "same seed must reproduce the shuffle stream"
    assert a1 != b, "different seed must vary the derived shuffle stream"


def test_bind_rng_does_not_override_explicit_rng():
    combee = ComBEEReflectionLM(lambda p: "x", rng=random.Random(123))
    explicit = combee._rng
    combee.bind_rng(random.Random(999))
    assert combee._rng is explicit
    combee_default = ComBEEReflectionLM(lambda p: "x")
    bound = random.Random(999)
    combee_default.bind_rng(bound)
    assert combee_default._rng is bound


def test_failed_batch_restores_rng_state_for_failure_transparency():
    """A transient failure in the batched round must not advance the shuffle
    stream: the per-task retry (and thus the whole run) stays bit-identical
    to a run where the failure never happened."""

    class FlakyThenContentLM:
        def __init__(self, fail_first_batch):
            self.fail_first_batch = fail_first_batch
            self.inner = ContentKeyedLM()

        def __call__(self, prompt):
            return self.inner(prompt)

        def batch_complete(self, messages_list):
            if self.fail_first_batch:
                self.fail_first_batch = False
                raise RuntimeError("transient provider outage")
            return [self.inner(m[0]["content"]) for m in messages_list]

    job = ({"comp": "X"}, {"comp": RECORDS9}, ["comp"])

    flaky = ComBEEReflectionLM(FlakyThenContentLM(fail_first_batch=True), rng=random.Random(5))
    with pytest.raises(RuntimeError):
        flaky.reflect_many([job, job])  # advances RNG in planning, then fails
    retry = [prop for prop, _ in flaky.reflect_many([job, job])]  # RNG was restored

    clean = ComBEEReflectionLM(FlakyThenContentLM(fail_first_batch=False), rng=random.Random(5))
    clean_results = [prop for prop, _ in clean.reflect_many([job, job])]

    for rp, cp in zip(retry, clean_results, strict=True):
        assert rp.new_texts == cp.new_texts
        assert rp.prompts == cp.prompts


def test_optimize_anything_cost_cap_stops_on_strategy_cost(tmp_path):
    """Regression (review P1): oa validated strategy.total_cost but wired the
    stopper to reflection_lm — the cap never fired on strategy cost."""
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    class CostlyStrategy:
        def __init__(self):
            self.total_cost = 0.0
            self.calls = 0

        def reflect(self, candidate, reflective_dataset, components_to_update):
            self.calls += 1
            self.total_cost += 1.0
            from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal

            return ReflectionProposal(new_texts={c: f"v{self.calls}_{c}" for c in components_to_update}), self

    strategy = CostlyStrategy()

    def evaluator(candidate, example=None):
        return float(len(str(candidate)) % 7) / 7.0, {"feedback": "vary"}

    optimize_anything(
        seed_candidate="seed text here",
        evaluator=evaluator,
        dataset=list(range(10)),
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=5000,  # huge: only the cost cap can stop this
                max_reflection_cost=2.5,
                run_dir=str(tmp_path / "oa-cap"),
                parallel=False,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(reflection_strategy=strategy, reflection_minibatch_size=3),
        ),
    )
    assert strategy.calls <= 4, f"cost cap never fired: {strategy.calls} reflections, ${strategy.total_cost}"


# ---------------------------------------------------------------------------
# Second review round: legacy stream compatibility, opt-out, retries, protocol
# ---------------------------------------------------------------------------


def test_strategy_binding_preserves_legacy_shared_engine_stream(tmp_path):
    """#307 shared ComBEE's RNG with GEPA. The default binding must retain
    that behavior, while an explicit strategy RNG remains isolated."""

    def minibatch_stream(reflection_strategy, tag):
        sampled = []

        class Cb:
            def on_minibatch_sampled(self, event):
                sampled.append(tuple(event["minibatch_ids"]))

        kwargs = {}
        if reflection_strategy is not None:
            kwargs["reflection_strategy"] = reflection_strategy
        else:
            kwargs["reflection_lm"] = lambda p: "```\nLEVEL=1\n```"
        gepa.optimize(
            seed_candidate={"system_prompt": "LEVEL=0"},
            trainset=TRAIN,
            valset=TRAIN,
            adapter=SyntheticAdapter(),
            reflection_minibatch_size=4,
            max_metric_calls=60,
            run_dir=str(tmp_path / tag),
            seed=0,
            display_progress_bar=False,
            callbacks=[Cb()],
            **kwargs,
        )
        return sampled

    base = minibatch_stream(None, "none")
    with_default = minibatch_stream(ComBEEReflectionLM(lambda p: "```\nLEVEL=1\n```"), "default")
    with_explicit = minibatch_stream(
        ComBEEReflectionLM(lambda p: "```\nLEVEL=1\n```", rng=random.Random(7)), "explicit"
    )
    assert base, "no minibatches sampled"
    assert with_default != base, "default ComBEE must share and advance the engine RNG as #307 did"
    assert with_explicit == base, "explicit-rng strategy must remain isolated from the engine stream"


class OrderDependentLM:
    """Reply depends on GLOBAL call order — the adversarial case for batching."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return f"```\nreply-{self.calls}\n```"

    def batch_complete(self, messages_list):
        out = []
        for _ in messages_list:
            self.calls += 1
            out.append(f"```\nreply-{self.calls}\n```")
        return out


def test_batch_reflection_false_enforces_sequential_ordering():
    """The opt-out keeps a transactional reflect_many but executes every job
    completely before moving to the next, matching direct sequential calls
    even for an order-dependent callable."""
    jobs = [
        ({"comp": "I0"}, {"comp": RECORDS9}, ["comp"]),
        ({"comp": "I1"}, {"comp": RECORDS9}, ["comp"]),
    ]

    opted_out = ComBEEReflectionLM(OrderDependentLM(), rng=random.Random(3), batch_reflection=False)
    assert callable(opted_out.reflect_many)

    reference = ComBEEReflectionLM(OrderDependentLM(), rng=random.Random(3), batch_reflection=False)
    seq_results = []
    for job in jobs:
        prop, _ = reference.reflect(*job)
        seq_results.append(prop.new_texts)

    ordered_results = [prop.new_texts for prop, _ in opted_out.reflect_many(jobs)]
    assert ordered_results == seq_results
    # Default instances still batch.
    assert callable(ComBEEReflectionLM(lambda p: "x").reflect_many)


def test_non_batchable_lm_automatically_preserves_sequential_ordering():
    """Without ``batch_complete``, a map-first global schedule has no speed
    benefit and changes order-dependent results. Default ComBEE must instead
    follow #307's complete-job ordering."""

    class PlainOrderDependentLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            return f"```\nreply-{self.calls}\n```"

    jobs = [
        ({"comp": "I0"}, {"comp": RECORDS9}, ["comp"]),
        ({"comp": "I1"}, {"comp": RECORDS9}, ["comp"]),
    ]
    combee = ComBEEReflectionLM(PlainOrderDependentLM(), rng=random.Random(3))

    results = [proposal.new_texts for proposal, _ in combee.reflect_many(jobs)]

    assert results == [{"comp": "reply-4"}, {"comp": "reply-8"}]


def test_optimize_binds_configured_logger_to_combee(tmp_path):
    """The documented strategy construction omits ``logger=``; the front
    door must supply its configured logger so fallback warnings are visible."""

    logger = CollectingLogger()
    lm = ImprovingLM()
    combee = ComBEEReflectionLM(lm, rng=random.Random(0))
    template = "PUBLIC OPTIMIZE TEMPLATE <curr_param> :: <side_info>"

    gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=SyntheticAdapter(),
        reflection_strategy=combee,
        reflection_minibatch_size=3,
        reflection_prompt_template=template,
        max_metric_calls=80,
        logger=logger,
        run_dir=str(tmp_path / "combee-logger"),
        seed=0,
        display_progress_bar=False,
    )

    assert combee.logger is logger
    assert combee.reflection_prompt_template == template
    assert any("PUBLIC OPTIMIZE TEMPLATE" in prompt for prompt in lm.prompts)
    assert any("falling back to standard single-call reflection" in message for message in logger.messages)


def test_optimize_anything_binds_configured_logger_to_combee(tmp_path):
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        TrackingConfig,
        optimize_anything,
        optimize_anything_reflection_prompt_template,
    )

    logger = CollectingLogger()
    lm = ImprovingLM()
    combee = ComBEEReflectionLM(lm, rng=random.Random(0))

    optimize_anything(
        seed_candidate="LEVEL=0",
        evaluator=lambda candidate, example=None: (0.0, {"feedback": "bad"}),
        dataset=[0, 1, 2],
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=80, run_dir=str(tmp_path / "combee-oa-logger"), parallel=False),
            reflection=ReflectionConfig(reflection_strategy=combee, reflection_minibatch_size=3),
            tracking=TrackingConfig(logger=logger),
        ),
    )

    assert combee.logger is logger
    assert combee.reflection_prompt_template == optimize_anything_reflection_prompt_template


def test_optimize_binds_public_lm_kwargs_to_combee_model_name(monkeypatch, tmp_path):
    import gepa.lm

    _KwargRecordingLM.instances.clear()
    monkeypatch.setattr(gepa.lm, "LM", _KwargRecordingLM)
    combee = ComBEEReflectionLM("test/model", rng=random.Random(0))

    gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=SyntheticAdapter(),
        reflection_strategy=combee,
        reflection_lm_kwargs={"temperature": 0.25, "max_tokens": 77},
        reflection_minibatch_size=3,
        max_metric_calls=80,
        run_dir=str(tmp_path / "combee-lm-kwargs"),
        seed=0,
        display_progress_bar=False,
    )

    assert combee.lm.model == "test/model"
    assert combee.lm.kwargs == {"temperature": 0.25, "max_tokens": 77}


def test_optimize_anything_binds_public_lm_kwargs_to_combee_model_name(monkeypatch, tmp_path):
    import gepa.lm
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    _KwargRecordingLM.instances.clear()
    monkeypatch.setattr(gepa.lm, "LM", _KwargRecordingLM)
    combee = ComBEEReflectionLM("test/model", rng=random.Random(0))

    optimize_anything(
        seed_candidate="LEVEL=0",
        evaluator=lambda candidate, example=None: (0.0, {"feedback": "bad"}),
        dataset=[0, 1, 2],
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=20, run_dir=str(tmp_path / "combee-oa-lm-kwargs"), parallel=False),
            reflection=ReflectionConfig(
                reflection_strategy=combee,
                reflection_lm_kwargs={"temperature": 0.25, "max_tokens": 77},
                reflection_minibatch_size=3,
            ),
        ),
    )

    assert combee.lm.model == "test/model"
    assert combee.lm.kwargs == {"temperature": 0.25, "max_tokens": 77}


def test_optimize_rejects_cost_cap_for_combee_plain_callable():
    from unittest.mock import MagicMock

    with pytest.raises(ValueError, match="cost-tracking LM"):
        gepa.optimize(
            seed_candidate={"c": "seed"},
            trainset=[{"q": 1}],
            adapter=MagicMock(propose_new_texts=None),
            reflection_strategy=ComBEEReflectionLM(lambda prompt: "```\nupdate\n```"),
            max_reflection_cost=1.0,
            max_metric_calls=10,
        )


def test_optimize_anything_rejects_cost_cap_for_combee_plain_callable(tmp_path):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    with pytest.raises(ValueError, match="cost-tracking LM"):
        optimize_anything(
            seed_candidate="seed",
            evaluator=lambda candidate, example=None: (0.0, {"feedback": "bad"}),
            dataset=[0],
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=10,
                    max_reflection_cost=1.0,
                    run_dir=str(tmp_path / "combee-oa-cost-cap"),
                    parallel=False,
                ),
                reflection=ReflectionConfig(reflection_strategy=ComBEEReflectionLM(lambda prompt: "```\nupdate\n```")),
            ),
        )


def test_tracker_logs_combee_reflection_diagnostics():
    """Intermediate ComBEE outputs must be available to trackers, not only
    on the transient CandidateProposal object."""
    from gepa.core.engine import GEPAEngine
    from gepa.proposer.base import CandidateProposal

    class CapturingTracker:
        def __init__(self):
            self.tables = []

        def log_table(self, table_name, columns, data):
            self.tables.append((table_name, columns, data))

    tracker = CapturingTracker()
    engine = object.__new__(GEPAEngine)
    engine.experiment_tracker = tracker
    proposal = CandidateProposal(
        candidate={"comp": "final"},
        parent_program_ids=[0],
        metadata={
            "proposal_id": "1-0",
            "prompt:comp": "reduce prompt",
            "raw_lm_output:comp": "```final```",
            "combee:comp:level1_prompts": ["map prompt 1", "map prompt 2"],
            "combee:comp:level1_outputs": ["map 1", "map 2"],
            "combee:comp:num_lm_calls": 3,
            "combee:total_lm_calls": 3,
        },
    )

    engine._log_proposal_lm_calls(iteration=1, proposal=proposal, candidate_idx=-1)

    metadata_table = next(table for table in tracker.tables if table[0] == "proposal_reflection_metadata")
    assert metadata_table[1][-1] == "reflection_metadata"
    diagnostics = json.loads(metadata_table[2][0][-1])
    assert diagnostics["combee:comp:level1_prompts"] == ["map prompt 1", "map prompt 2"]
    assert diagnostics["combee:comp:level1_outputs"] == ["map 1", "map 2"]
    assert diagnostics["combee:total_lm_calls"] == 3


def test_failed_reduce_wave_is_cost_transparent():
    """RNG restore alone repeats paid map calls on retry; the completion memo
    must reuse them: total completions across failure+retry == clean run."""

    class FailSecondBatchLM:
        def __init__(self):
            self.completions = 0
            self.batch_invocations = 0
            self.prompt_log = []

        def _reply(self, text):
            self.completions += 1
            self.prompt_log.append(text)
            return f"```\nR{hash(text) % 99991}\n```"

        def __call__(self, prompt):
            return self._reply(prompt if isinstance(prompt, str) else str(prompt))

        def batch_complete(self, messages_list):
            self.batch_invocations += 1
            if self.batch_invocations == 2:  # the reduce wave
                raise RuntimeError("transient outage during reduce wave")
            return [self._reply(str(m[0]["content"])) for m in messages_list]

    jobs = [
        ({"comp": "X"}, {"comp": RECORDS9}, ["comp"]),
        ({"comp": "Y"}, {"comp": RECORDS9}, ["comp"]),
    ]

    flaky_lm = FailSecondBatchLM()
    combee = ComBEEReflectionLM(flaky_lm, rng=random.Random(5))
    with pytest.raises(RuntimeError):
        combee.reflect_many([tuple(j) for j in jobs])
    paid_before_retry = flaky_lm.completions
    assert paid_before_retry == 6  # both jobs' map waves completed and were paid

    # Engine retry path: per-task reflect() (as _propose_texts_batch_safe does).
    retry_results = []
    for job in jobs:
        prop, _ = combee.reflect(*job)
        retry_results.append(prop)

    # Cost transparency: the 6 paid map completions were REUSED, not re-bought;
    # only the 2 reduces were newly purchased.
    assert flaky_lm.completions == 8, f"paid {flaky_lm.completions} completions (clean run pays 8)"
    assert len(set(flaky_lm.prompt_log)) == len(flaky_lm.prompt_log), "a prompt was purchased twice"

    # Result transparency: identical to a run where the outage never happened.
    class CleanLM(FailSecondBatchLM):
        def batch_complete(self, messages_list):
            self.batch_invocations += 1
            return [self._reply(str(m[0]["content"])) for m in messages_list]

    clean = ComBEEReflectionLM(CleanLM(), rng=random.Random(5))
    clean_results = [prop for prop, _ in clean.reflect_many([tuple(j) for j in jobs])]
    for rp, cp in zip(retry_results, clean_results, strict=True):
        assert rp.new_texts == cp.new_texts


def test_bind_rng_true_attribute_does_not_crash_wiring(tmp_path):
    """A strategy with a non-callable bind_rng attribute must be ignored by
    the wiring, not called (previous code called any non-None attribute)."""

    class WeirdStrategy:
        bind_rng = True  # non-callable attribute

        def reflect(self, candidate, reflective_dataset, components_to_update):
            from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal

            return ReflectionProposal(new_texts=dict.fromkeys(components_to_update, "v")), self

    result = gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=SyntheticAdapter(),
        reflection_strategy=WeirdStrategy(),
        max_metric_calls=25,
        run_dir=str(tmp_path / "weird"),
        seed=0,
        display_progress_bar=False,
    )
    assert result is not None


def test_seedable_protocol_conformance():
    from gepa.proposer.reflective_mutation.reflection_lm import SeedableReflectionLM

    assert isinstance(ComBEEReflectionLM(lambda p: "x"), SeedableReflectionLM)


# ---------------------------------------------------------------------------
# Third review round: retry identity and ordered PxN recovery
# ---------------------------------------------------------------------------


def test_direct_reflect_does_not_cache_completed_prompts():
    """The public single-job method must issue a fresh completion on every
    successful invocation, including when strict ordering is selected."""

    class CountingLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            return f"```\nreply-{self.calls}\n```"

    lm = CountingLM()
    combee = ComBEEReflectionLM(lm, batch_reflection=False)
    job = ({"comp": "old"}, {"comp": [{"Feedback": "same"}]}, ["comp"])

    first, _ = combee.reflect(*job)
    second, _ = combee.reflect(*job)

    assert first.new_texts == {"comp": "reply-1"}
    assert second.new_texts == {"comp": "reply-2"}
    assert lm.calls == 2


def test_retry_does_not_conflate_identical_prompt_occurrences():
    """Two equal prompts are still two calls. If the second one fails, retry
    must reuse only the first logical occurrence, not its response for both."""

    class FailSecondCallLM:
        def __init__(self):
            self.calls = 0
            self.completions = 0

        def __call__(self, prompt):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient outage")
            self.completions += 1
            return f"```\nreply-{self.calls}\n```"

    lm = FailSecondCallLM()
    combee = ComBEEReflectionLM(lm)
    job = ({"comp": "old"}, {"comp": [{"Feedback": "same"}]}, ["comp"])

    with pytest.raises(RuntimeError):
        combee.reflect_many([job, job])

    first, _ = combee.reflect(*job)
    second, _ = combee.reflect(*job)

    assert first.new_texts == {"comp": "reply-1"}
    assert second.new_texts == {"comp": "reply-3"}
    assert lm.calls == 3
    assert lm.completions == 2


def test_direct_reflect_many_retry_reuses_completed_map_wave():
    """The public reflect_many API itself must preserve paid map calls when a
    reduce wave fails and the caller retries the same jobs."""

    class FailFirstReduceLM:
        def __init__(self):
            self.completions = 0
            self.batch_invocations = 0

        def __call__(self, prompt):
            self.completions += 1
            return f"```\nR{hash(str(prompt)) % 99991}\n```"

        def batch_complete(self, messages_list):
            self.batch_invocations += 1
            if self.batch_invocations == 2:
                raise RuntimeError("transient reduce outage")
            return [self(messages[0]["content"]) for messages in messages_list]

    jobs = [
        ({"comp": "X"}, {"comp": RECORDS9}, ["comp"]),
        ({"comp": "Y"}, {"comp": RECORDS9}, ["comp"]),
    ]
    lm = FailFirstReduceLM()
    combee = ComBEEReflectionLM(lm, rng=random.Random(5))

    with pytest.raises(RuntimeError):
        combee.reflect_many(jobs)
    assert lm.completions == 6

    combee.reflect_many(jobs)
    assert lm.completions == 8


def test_ordered_pxn_retry_is_cost_and_result_transparent():
    """The strict-ordering mode is still a PxN transaction: a failure in a
    later job must not rerun earlier completed jobs with a new shuffle."""

    class ContentLM:
        def __init__(self, fail_at=None):
            self.fail_at = fail_at
            self.calls = 0
            self.completions = 0

        def __call__(self, prompt):
            self.calls += 1
            if self.calls == self.fail_at:
                raise RuntimeError("transient outage")
            self.completions += 1
            return f"```\nR{hash(str(prompt)) % 99991}\n```"

    jobs = [
        ({"comp": "X"}, {"comp": RECORDS9}, ["comp"]),
        ({"comp": "Y"}, {"comp": RECORDS9}, ["comp"]),
    ]
    flaky_lm = ContentLM(fail_at=5)
    flaky = ComBEEReflectionLM(flaky_lm, rng=random.Random(5), batch_reflection=False)

    with pytest.raises(RuntimeError):
        flaky.reflect_many(jobs)
    retry = [flaky.reflect(*job)[0] for job in jobs]

    clean_lm = ContentLM()
    clean = ComBEEReflectionLM(clean_lm, rng=random.Random(5), batch_reflection=False)
    clean_results = [proposal for proposal, _ in clean.reflect_many(jobs)]

    assert flaky_lm.completions == 8
    assert flaky_lm.calls == 9  # includes the one failed, unpaid invocation
    assert clean_lm.completions == 8
    for retry_proposal, clean_proposal in zip(retry, clean_results, strict=True):
        assert retry_proposal.new_texts == clean_proposal.new_texts
        assert retry_proposal.prompts == clean_proposal.prompts
