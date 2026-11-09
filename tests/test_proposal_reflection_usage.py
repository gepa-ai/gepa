# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Per-proposal usage stays attached to the completions that produced it."""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from gepa.lm import LM, LMOutput, LMUsage, TrackingLM
from gepa.proposer.reflective_mutation.combee import ComBEEReflectionLM
from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM


def _response(cost, tokens_in, tokens_out, text="```\nimproved\n```", finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
        cost=cost,
    )


def _job(seed="seed", components=("a",), records=1):
    return (
        dict.fromkeys(components, seed),
        {comp: [{"Feedback": f"example {i}"} for i in range(records)] for comp in components},
        list(components),
    )


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: completion_response.cost)


@pytest.mark.parametrize("batched", [False, True])
def test_lm_outputs_retain_individual_usage(monkeypatch, provider, batched):
    responses = [_response(0.01, 100, 20), _response(0.03, 200, 40)]
    monkeypatch.setattr("litellm.batch_completion", lambda **kwargs: responses)
    responses_iter = iter(responses)
    monkeypatch.setattr("litellm.completion", lambda **kwargs: next(responses_iter))
    lm = LM("test/model")
    outputs = lm.batch_complete([[], []]) if batched else [lm("a"), lm("b")]

    assert [output.strip().usage for output in outputs] == [
        LMUsage(cost_usd=0.01, tokens_in=100, tokens_out=20),
        LMUsage(cost_usd=0.03, tokens_in=200, tokens_out=40),
    ]
    assert lm.total_cost == pytest.approx(0.04)
    assert (lm.total_tokens_in, lm.total_tokens_out) == (300, 60)


@pytest.mark.parametrize("cost", [None, 0.0])
def test_unknown_cost_stays_distinct_from_zero(monkeypatch, provider, cost):
    response = _response(cost, 0, None)
    monkeypatch.setattr("litellm.completion", lambda **kwargs: response)
    proposal, _ = StatelessReflectionLM(LM("test/model")).reflect(*_job())

    usage = proposal.metadata["reflection_usage"]
    assert usage["cost_usd"] == cost
    assert usage["tokens_in"] == 0
    assert usage["tokens_out"] is None
    assert json.loads(json.dumps(usage)) == usage


def test_batched_proposals_include_rejected_component_cost(monkeypatch, provider):
    responses = [
        _response(0.01, 100, 20),
        _response(0.02, 200, 30, text="<think>unfinished", finish_reason="length"),
        _response(0.04, 400, 50),
    ]
    monkeypatch.setattr("litellm.batch_completion", lambda **kwargs: responses)
    lm = LM("test/model")
    proposals = StatelessReflectionLM(lm).reflect_many([_job(components=("a", "b")), _job(), _job(components=())])
    first, second, empty = [result[0] for result in proposals]

    assert first.new_texts == {"a": "improved"}
    usage = first.metadata["reflection_usage"]
    assert usage["calls"] == 2
    assert usage["cost_usd"] == pytest.approx(0.03)
    assert (usage["tokens_in"], usage["tokens_out"]) == (300, 50)
    assert usage["components"]["b"]["cost_usd"] == 0.02
    assert second.metadata["reflection_usage"]["cost_usd"] == 0.04
    assert "reflection_usage" not in empty.metadata
    assert lm.total_cost == pytest.approx(0.07)


def test_concurrent_proposals_sharing_one_lm_have_separate_usage(monkeypatch, provider):
    barrier = Barrier(2)

    def complete(**kwargs):
        response = (
            _response(0.01, 100, 20) if "seed-a" in kwargs["messages"][0]["content"] else _response(0.03, 300, 60)
        )
        barrier.wait(timeout=5)
        return response

    monkeypatch.setattr("litellm.completion", complete)
    lm = LM("test/model")
    reflector = StatelessReflectionLM(lm)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda seed: reflector.reflect(*_job(seed)), ["seed-a", "seed-b"]))

    usages = [result[0].metadata["reflection_usage"] for result in results]
    assert [usage["cost_usd"] for usage in usages] == [0.01, 0.03]
    assert [usage["tokens_in"] for usage in usages] == [100, 300]
    assert lm.total_cost == pytest.approx(0.04)


@pytest.mark.parametrize("batched", [False, True])
def test_combee_attributes_map_and_reduce_calls(monkeypatch, provider, batched):
    monkeypatch.setattr("litellm.completion", lambda **kwargs: _response(0.01, 100, 20))
    monkeypatch.setattr(
        "litellm.batch_completion",
        lambda messages, **kwargs: [_response(0.01, 100, 20) for _ in messages],
    )
    lm = LM("test/model")
    reflector = ComBEEReflectionLM(lm, rng=random.Random(0), batch_reflection=batched)
    results = reflector.reflect_many([_job(records=4), _job(records=1)])

    usages = [result[0].metadata["reflection_usage"] for result in results]
    assert [usage["calls"] for usage in usages] == [3, 1]
    assert [usage["cost_usd"] for usage in usages] == pytest.approx([0.03, 0.01])
    assert [usage["tokens_in"] for usage in usages] == [300, 100]
    assert lm.total_cost == pytest.approx(0.04)


def test_combee_retry_counts_cached_completions_once(monkeypatch, provider):
    monkeypatch.setattr(
        "litellm.batch_completion", lambda messages, **kwargs: [_response(0.01, 100, 20) for _ in messages]
    )
    attempts = iter([RuntimeError("transient failure"), _response(0.02, 200, 40)])

    def complete(**kwargs):
        response = next(attempts)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("litellm.completion", complete)
    lm = LM("test/model")
    reflector = ComBEEReflectionLM(lm, rng=random.Random(0))
    jobs = [_job(records=4)]
    with pytest.raises(RuntimeError, match="transient failure"):
        reflector.reflect_many(jobs)
    proposal, _ = reflector.reflect_many(jobs)[0]

    usage = proposal.metadata["reflection_usage"]
    assert usage["calls"] == 3
    assert usage["cost_usd"] == pytest.approx(0.04)
    assert usage["cost_usd"] == pytest.approx(lm.total_cost)
    assert usage["tokens_in"] == lm.total_tokens_in == 400


def test_tracking_lm_marks_estimates_and_preserves_termination_metadata():
    lm = TrackingLM(lambda prompt: LMOutput("x" * 40, finish_reason="length"))
    output = lm("y" * 80)
    assert output.usage == LMUsage(tokens_in=20, tokens_out=10, estimated=True)
    assert output.strip().finish_reason == "length"


def test_plain_callable_has_unknown_usage():
    proposal, _ = StatelessReflectionLM(lambda prompt: "```\nimproved\n```").reflect(*_job())
    usage = proposal.metadata["reflection_usage"]
    assert usage["calls"] == 1
    assert usage["cost_usd"] is None
    assert usage["tokens_in"] is None
