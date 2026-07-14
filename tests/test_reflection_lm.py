# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the ReflectionLM protocol and StatelessReflectionLM (#329 Phase 1)."""

from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionLM,
    ReflectionProposal,
    StatelessReflectionLM,
)


class RecordingLM:
    """A fake reflection LM: records prompts, returns a fenced instruction."""

    def __init__(self, reply: str = "improved instruction"):
        self.reply = reply
        self.calls: list = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return f"Here is the update:\n```\n{self.reply}\n```"


class BatchRecordingLM(RecordingLM):
    """A fake LM that also exposes ``batch_complete`` (like ``gepa.lm.LM``)."""

    def __init__(self, reply: str = "improved instruction"):
        super().__init__(reply)
        self.batch_calls: list[list] = []

    def batch_complete(self, messages_list, max_workers: int = 10):
        self.batch_calls.append(messages_list)
        return [f"```\n{self.reply}\n```" for _ in messages_list]


class CollectingLogger:
    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def _reflective_dataset(components):
    return {name: [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "bad"}] for name in components}


def test_stateless_reflection_lm_satisfies_protocol():
    lm = StatelessReflectionLM(RecordingLM())
    assert isinstance(lm, ReflectionLM)


def test_reflect_returns_self_as_next_lm():
    """Stateless: the next LM is the same object (no carried context)."""
    lm = StatelessReflectionLM(RecordingLM())
    candidate = {"system_prompt": "old"}
    proposal, next_lm = lm.reflect(candidate, _reflective_dataset(["system_prompt"]), ["system_prompt"])
    assert isinstance(proposal, ReflectionProposal)
    assert next_lm is lm


def test_reflect_proposes_new_text_per_component():
    fake = RecordingLM(reply="brand new prompt")
    lm = StatelessReflectionLM(fake)
    candidate = {"a": "old_a", "b": "old_b"}
    proposal, _ = lm.reflect(candidate, _reflective_dataset(["a", "b"]), ["a", "b"])
    assert proposal.new_texts == {"a": "brand new prompt", "b": "brand new prompt"}
    # Diagnostics populated per component.
    assert set(proposal.prompts.keys()) == {"a", "b"}
    assert set(proposal.raw_lm_outputs.keys()) == {"a", "b"}
    assert len(fake.calls) == 2


def test_reflect_skips_components_without_feedback():
    fake = RecordingLM()
    logger = CollectingLogger()
    lm = StatelessReflectionLM(fake, logger=logger)
    candidate = {"a": "old_a", "b": "old_b"}
    # Only 'a' has data; 'b' is requested but absent from the reflective dataset.
    reflective_dataset = _reflective_dataset(["a"])
    proposal, _ = lm.reflect(candidate, reflective_dataset, ["a", "b"])
    assert set(proposal.new_texts.keys()) == {"a"}
    assert len(fake.calls) == 1
    assert any("'b' is not in reflective dataset" in m for m in logger.messages)


def test_reflect_many_batches_into_one_call():
    """N tasks with a batch-capable LM → a single batch_complete covering all prompts."""
    fake = BatchRecordingLM(reply="new text")
    lm = StatelessReflectionLM(fake)
    jobs = [
        ({"a": "old0"}, _reflective_dataset(["a"]), ["a"]),
        ({"a": "old1"}, _reflective_dataset(["a"]), ["a"]),
        ({"a": "old2"}, _reflective_dataset(["a"]), ["a"]),
    ]
    results = lm.reflect_many(jobs)
    assert len(results) == len(jobs)
    assert all(proposal.new_texts == {"a": "new text"} for proposal, _ in results)
    assert all(next_lm is lm for _, next_lm in results)
    # One batched request covering all 3 jobs — not 3 separate completions.
    assert len(fake.batch_calls) == 1
    assert len(fake.batch_calls[0]) == 3
    assert fake.calls == []  # the per-call path was not used


def test_reflect_many_scatters_results_per_job():
    """Each job's components must come back in its own proposal slot, in order."""
    fake = BatchRecordingLM(reply="X")
    lm = StatelessReflectionLM(fake)
    jobs = [
        ({"a": "0a", "b": "0b"}, _reflective_dataset(["a", "b"]), ["a", "b"]),
        ({"a": "1a"}, _reflective_dataset(["a"]), ["a"]),
    ]
    results = lm.reflect_many(jobs)
    assert set(results[0][0].new_texts.keys()) == {"a", "b"}
    assert set(results[1][0].new_texts.keys()) == {"a"}
    # 3 (job,component) prompts total, one batched call.
    assert len(fake.batch_calls) == 1
    assert len(fake.batch_calls[0]) == 3


def test_reflect_many_sequential_fallback_without_batch_complete():
    """A plain callable LM (no batch_complete) still works, one call per prompt."""
    fake = RecordingLM(reply="Y")
    lm = StatelessReflectionLM(fake)
    jobs = [
        ({"a": "0a"}, _reflective_dataset(["a"]), ["a"]),
        ({"a": "1a"}, _reflective_dataset(["a"]), ["a"]),
    ]
    results = lm.reflect_many(jobs)
    assert [p.new_texts for p, _ in results] == [{"a": "Y"}, {"a": "Y"}]
    assert len(fake.calls) == 2


def test_per_component_template_dict_and_missing_warning():
    fake = RecordingLM()
    logger = CollectingLogger()
    # Template provided for 'a' only; 'b' should warn once and fall back to default.
    template_a = "Improve <curr_param> using this feedback: <side_info>"
    lm = StatelessReflectionLM(fake, reflection_prompt_template={"a": template_a}, logger=logger)
    candidate = {"a": "old_a", "b": "old_b"}
    reflective_dataset = _reflective_dataset(["a", "b"])
    lm.reflect(candidate, reflective_dataset, ["a", "b"])
    # Warn again on a second call — but only once per component overall.
    lm.reflect(candidate, reflective_dataset, ["a", "b"])
    missing_warnings = [m for m in logger.messages if "No reflection_prompt_template found for parameter 'b'" in m]
    assert len(missing_warnings) == 1


# ---------------------------------------------------------------------------
# reflection_strategy injection seam (#329 Phase 2 entry point)
# ---------------------------------------------------------------------------


class _ReflectOnlyLM:
    """A ReflectionLM implementation that provides only reflect() — no reflect_many."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def reflect(self, candidate, reflective_dataset, components_to_update):
        self.calls.append(list(components_to_update))
        proposal = ReflectionProposal(new_texts={c: f"new_{c}" for c in components_to_update})
        return proposal, self


def _make_proposer(**overrides):
    from unittest.mock import MagicMock

    from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

    adapter = MagicMock()
    adapter.propose_new_texts = None
    kwargs = {
        "logger": MagicMock(),
        "trainset": [{"q": 1}, {"q": 2}],
        "adapter": adapter,
        "candidate_selector": MagicMock(),
        "module_selector": MagicMock(),
        "batch_sampler": MagicMock(),
        "perfect_score": None,
        "skip_perfect_score": False,
        "experiment_tracker": MagicMock(),
        "reflection_lm": None,
        "custom_candidate_proposer": None,
    }
    kwargs.update(overrides)
    return ReflectiveMutationProposer(**kwargs)


def test_injected_reflection_strategy_takes_precedence_over_stateless_default():
    stub = _ReflectOnlyLM()
    proposer = _make_proposer(reflection_strategy=stub, reflection_lm=RecordingLM())
    assert proposer._reflection_lm is stub


def test_without_injection_stateless_default_is_built():
    proposer = _make_proposer(reflection_lm=RecordingLM())
    assert isinstance(proposer._reflection_lm, StatelessReflectionLM)


def test_reflect_only_strategy_used_per_task_in_batch_path():
    """A ReflectionLM without reflect_many must be called once per job by
    _propose_texts_batch (the seam a ComBEE-style reflector plugs into)."""
    stub = _ReflectOnlyLM()
    proposer = _make_proposer(reflection_strategy=stub)
    jobs = [
        ({"c": "old1"}, {"c": [{"feedback": "f1"}]}, ["c"]),
        ({"c": "old2"}, {"c": [{"feedback": "f2"}]}, ["c"]),
    ]
    results = proposer._propose_texts_batch(jobs)
    assert stub.calls == [["c"], ["c"]]
    assert [r[0] for r in results] == [{"c": "new_c"}, {"c": "new_c"}]
