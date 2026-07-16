# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for ComBEEReflectionLM (#307 re-hosted on the ReflectionLM seam, #329 Phase 3).

The characterization tests in this file were first validated against the
ORIGINAL PR #307 implementation (`_propose_new_texts_combee`), then against
this port — with a fixed RNG seed and fake LM, the two produce byte-identical
LM-call sequences and final texts across all scenarios, so these tests pin
behavior preserved from the original contribution by @nuglifeleoji.
"""

import random
import re
from collections import defaultdict

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


def test_cost_tracking_delegation_for_max_reflection_cost():
    """Plain callables are wrapped so total_cost exists — the config-time guard
    for max_reflection_cost accepts ComBEE without extra user work."""
    combee, _ = _combee(["m0", "m1", "m2", "fin"])
    assert hasattr(combee, "total_cost")
    combee.reflect({"comp": "X"}, {"comp": RECORDS9}, ["comp"])
    assert combee.total_tokens_in > 0
    assert combee.total_tokens_out > 0


def test_str_model_name_resolved_via_gepa_lm():
    """A model-name string constructs a gepa.lm.LM with cost tracking wired."""
    from gepa.lm import LM

    combee = ComBEEReflectionLM("openai/gpt-4.1-mini")
    assert isinstance(combee.lm, LM)
    assert combee.total_cost == 0.0
    assert combee.total_tokens_in == 0


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

    def __call__(self, prompt):
        self.calls += 1
        text = prompt if isinstance(prompt, str) else str(prompt)
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
    # ComBEE diagnostics flow into CandidateProposal.metadata.
    assert spy.proposals
    md = spy.proposals[0].metadata
    assert md["combee:system_prompt:mode"] == "map_reduce"
    assert md["combee:system_prompt:num_lm_calls"] == 4
    assert md["combee:total_lm_calls"] == 4
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
    """ComBEE x PxNSampling: multi-task iterations route each task through
    ComBEE's reflect() via the per-task fallback (ComBEE has no reflect_many),
    so every attempted proposal costs exactly k+1 reflection calls, and each
    proposal carries its own independent ComBEE diagnostics."""
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
        ({"comp": "I0"}, {"comp": RECORDS9}, ["comp"]),   # k=3 map/reduce
        ({"comp": "I1"}, {"comp": RECORDS3}, ["comp"]),   # k=1 fallback
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
            assert i + 1 < len(waves) and waves[i + 1] == n_jobs, (
                f"expected wave2={n_jobs} after wave1={w1}: {waves}"
            )
            i += 2
        else:
            i += 1  # wave2 for a single job went through the plain-call path
    assert result.val_aggregate_scores[result.best_idx] > result.val_aggregate_scores[0]
