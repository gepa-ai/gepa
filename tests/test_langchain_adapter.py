# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the LangChain adapter.

Tests cover:
- ``last_message_text`` text extraction across message shapes
- ``make_reflection_lm`` wrapping a ``BaseChatModel`` into the
  ``(prompt: str) -> str`` callable GEPA expects
- ``LangChainAdapter.evaluate`` happy-path, trace capture, and
  rollout-error handling
- ``LangChainAdapter.make_reflective_dataset`` record building
- An end-to-end smoke test running ``gepa.optimize`` with a fake LLM

All tests use ``FakeListChatModel`` so no network calls occur.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="requires gepa[langchain] extra")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from gepa import optimize
from gepa.adapters.langchain_adapter import (
    LangChainAdapter,
    last_message_text,
    make_reflection_lm,
)

# ============================================================================
# last_message_text
# ============================================================================


class TestLastMessageText:
    def test_string_content(self):
        state = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}
        assert last_message_text(state) == "hello"

    def test_list_content_anthropic_style(self):
        msg = AIMessage(content=[{"type": "text", "text": "part-a"}, {"type": "text", "text": "part-b"}])
        assert last_message_text({"messages": [msg]}) == "part-apart-b"

    def test_list_content_with_non_dict_part(self):
        msg = AIMessage(content=[{"type": "text", "text": "hi"}, "raw"])
        assert last_message_text({"messages": [msg]}) == "hiraw"

    def test_empty_messages(self):
        assert last_message_text({"messages": []}) == ""

    def test_missing_messages_key(self):
        assert last_message_text({}) == ""

    def test_non_dict_state(self):
        assert last_message_text("not a dict") == ""  # type: ignore[arg-type]


# ============================================================================
# make_reflection_lm
# ============================================================================


class TestMakeReflectionLM:
    def test_wraps_basechatmodel(self):
        llm = FakeListChatModel(responses=["reflected output"])
        fn = make_reflection_lm(llm)
        assert fn("any prompt") == "reflected output"

    def test_flattens_list_content(self):
        class ListContentModel(FakeListChatModel):
            def invoke(self, messages, **kwargs):
                return AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])

        fn = make_reflection_lm(ListContentModel(responses=["unused"]))
        assert fn("prompt") == "ab"

    def test_invoked_with_human_message(self):
        captured: list = []

        class CapturingModel(FakeListChatModel):
            def invoke(self, messages, **kwargs):
                captured.append(messages)
                return AIMessage(content="ok")

        fn = make_reflection_lm(CapturingModel(responses=["ok"]))
        fn("hello prompt")
        assert len(captured) == 1
        assert len(captured[0]) == 1
        assert isinstance(captured[0][0], HumanMessage)
        assert captured[0][0].content == "hello prompt"


# ============================================================================
# LangChainAdapter.evaluate
# ============================================================================


def _simple_eval(example, state):
    """Score 1.0 if the last message text equals example['answer']."""
    pred = last_message_text(state).strip()
    expected = str(example["answer"]).strip()
    if pred == expected:
        return 1.0, "correct"
    return 0.0, f"expected {expected!r}, got {pred!r}"


def _make_rollout(responses_by_input: dict[str, str]):
    def rollout(candidate, example):
        text = responses_by_input[example["input"]]
        return {
            "messages": [
                SystemMessage(content=candidate["system_prompt"]),
                HumanMessage(content=example["input"]),
                AIMessage(content=text),
            ]
        }

    return rollout


class TestEvaluate:
    def test_happy_path_scores_match(self):
        batch = [
            {"input": "q1", "answer": "a1"},
            {"input": "q2", "answer": "a2"},
            {"input": "q3", "answer": "wrong"},
        ]
        rollout = _make_rollout({"q1": "a1", "q2": "a2", "q3": "a3"})
        adapter = LangChainAdapter(
            rollout_fn=rollout,
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
        )
        result = adapter.evaluate(batch=batch, candidate={"system_prompt": "p"}, capture_traces=False)

        assert result.scores == [1.0, 1.0, 0.0]
        assert result.trajectories is None
        assert len(result.outputs) == 3
        assert result.outputs[0]["response"] == "a1"

    def test_capture_traces_populates_trajectories(self):
        batch = [{"input": "q1", "answer": "a1"}]
        rollout = _make_rollout({"q1": "a1"})
        adapter = LangChainAdapter(
            rollout_fn=rollout,
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
        )
        result = adapter.evaluate(batch=batch, candidate={"system_prompt": "p"}, capture_traces=True)

        assert result.trajectories is not None
        assert len(result.trajectories) == 1
        traj = result.trajectories[0]
        assert traj["data"] == batch[0]
        assert traj["score"] == 1.0
        assert traj["feedback"] == "correct"
        assert "messages" in traj["state"]

    def test_rollout_returns_non_dict_recorded_as_failure(self):
        def bad_rollout(candidate, example):
            return "not a dict"

        adapter = LangChainAdapter(
            rollout_fn=bad_rollout,
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
        )
        result = adapter.evaluate(
            batch=[{"input": "q1", "answer": "a1"}],
            candidate={"system_prompt": "p"},
            capture_traces=True,
        )

        assert result.scores == [0.0]
        state = result.outputs[0]["state"]
        assert isinstance(state["error"], TypeError)
        assert "rollout_fn must return a dict" in str(state["error"])

    def test_rollout_exception_is_caught(self):
        class BoomError(RuntimeError):
            pass

        def raising_rollout(candidate, example):
            raise BoomError("kaboom")

        adapter = LangChainAdapter(
            rollout_fn=raising_rollout,
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
        )
        result = adapter.evaluate(
            batch=[{"input": "q1", "answer": "a1"}],
            candidate={"system_prompt": "p"},
            capture_traces=True,
        )

        assert result.scores == [0.0]
        state = result.outputs[0]["state"]
        assert isinstance(state["error"], BoomError)
        assert "ERROR: BoomError: kaboom" in last_message_text(state)


# ============================================================================
# LangChainAdapter.make_reflective_dataset
# ============================================================================


class TestMakeReflectiveDataset:
    def _build_eval_batch(self, adapter):
        return adapter.evaluate(
            batch=[{"input": "q1", "answer": "a1"}, {"input": "q2", "answer": "a2"}],
            candidate={"system_prompt": "p"},
            capture_traces=True,
        )

    def test_default_record_shape(self):
        adapter = LangChainAdapter(
            rollout_fn=_make_rollout({"q1": "a1", "q2": "wrong"}),
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
        )
        eval_batch = self._build_eval_batch(adapter)

        ds = adapter.make_reflective_dataset(
            candidate={"system_prompt": "p"},
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        assert set(ds.keys()) == {"system_prompt"}
        records = ds["system_prompt"]
        assert len(records) == 2
        assert records[0]["Inputs"] == "q1"
        assert records[0]["Generated Outputs"] == "a1"
        assert records[0]["Feedback"] == "correct"

    def test_empty_trajectories_raises(self):
        adapter = LangChainAdapter(
            rollout_fn=_make_rollout({}),
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
        )
        from gepa.core.adapter import EvaluationBatch

        empty = EvaluationBatch(outputs=[], scores=[], trajectories=[])
        with pytest.raises(RuntimeError, match="No trajectories captured"):
            adapter.make_reflective_dataset(
                candidate={"system_prompt": "p"},
                eval_batch=empty,
                components_to_update=["system_prompt"],
            )

    def test_custom_reflective_record_fn(self):
        def custom_record(data, state, score, feedback):
            return {"q": data["input"], "score": score}

        adapter = LangChainAdapter(
            rollout_fn=_make_rollout({"q1": "a1", "q2": "a2"}),
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
            reflective_record_fn=custom_record,
        )
        eval_batch = self._build_eval_batch(adapter)
        ds = adapter.make_reflective_dataset(
            candidate={"system_prompt": "p"},
            eval_batch=eval_batch,
            components_to_update=["system_prompt", "other"],
        )

        assert set(ds.keys()) == {"system_prompt", "other"}
        assert ds["system_prompt"] == ds["other"]
        assert ds["system_prompt"][0] == {"q": "q1", "score": 1.0}


# ============================================================================
# End-to-end smoke test through gepa.optimize
# ============================================================================


class TestOptimizeSmoke:
    def test_optimize_runs_with_fake_llm(self):
        """Run the full GEPA loop with a fake task LLM and fake reflection LLM.

        This catches regressions in adapter <-> optimizer wiring. We don't
        assert on optimization quality — just that the loop completes and
        returns a usable result.
        """
        train_set = [{"input": f"q{i}", "answer": "a"} for i in range(2)]
        val_set = [{"input": f"v{i}", "answer": "a"} for i in range(2)]

        task_llm = FakeListChatModel(responses=["a"] * 50)
        reflection_llm = FakeListChatModel(responses=["new system prompt"] * 20)

        def rollout(candidate, example):
            messages = [
                SystemMessage(content=candidate["system_prompt"]),
                HumanMessage(content=example["input"]),
            ]
            result = task_llm.invoke(messages)
            return {"messages": messages + [result]}

        adapter = LangChainAdapter(
            rollout_fn=rollout,
            eval_fn=_simple_eval,
            num_threads=1,
            show_progress=False,
        )

        result = optimize(
            seed_candidate={"system_prompt": "seed"},
            trainset=train_set,
            valset=val_set,
            adapter=adapter,
            reflection_lm=make_reflection_lm(reflection_llm),
            max_metric_calls=10,
            reflection_minibatch_size=2,
            display_progress_bar=False,
            seed=0,
        )

        assert result.best_candidate is not None
        assert "system_prompt" in result.best_candidate
        assert isinstance(result.val_aggregate_scores[result.best_idx], float)
