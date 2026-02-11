from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from opik_optimizer import ChatPrompt

try:
    from gepa.adapters.opik_adapter.opik_adapter import (
        OpikAdapter,
        OpikDataInst,
    )
except ImportError:
    pytest.skip(
        "gepa.adapters.opik_adapter not available; install gepa with Opik adapter",
        allow_module_level=True,
    )


class DummyMetricResult:
    def __init__(self, value: float) -> None:
        self.value = value


class DummyAgent:
    def __init__(self) -> None:
        self.invoke_calls: list[list[dict[str, Any]]] = []

    def invoke(self, messages: list[dict[str, Any]]) -> str:
        self.invoke_calls.append(messages)
        return "A"


class DummyCandidateAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.invoke_agent_candidates_calls: list[dict[str, Any]] = []
        self._last_candidate_logprobs: list[float] | None = None

    def invoke_agent_candidates(
        self,
        prompts: dict[str, ChatPrompt],
        dataset_item: dict[str, Any],
        allow_tool_use: bool = False,
    ) -> list[str]:
        self.invoke_agent_candidates_calls.append(
            {"prompts": prompts, "dataset_item": dataset_item, "allow_tool_use": allow_tool_use}
        )
        self._last_candidate_logprobs = [0.05, 0.9]
        return ["bad", "A"]

    def invoke_agent(
        self,
        prompts: dict[str, ChatPrompt],
        dataset_item: dict[str, Any],
        allow_tool_use: bool = False,
    ) -> str:
        self.invoke_calls.append(prompts["prompt"].get_messages(dataset_item))
        return "A"


class DummyDataset:
    name = "dummy"

    def get_items(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []


def _make_metric(score_value: float):
    def metric(dataset_item: dict[str, Any], llm_output: str) -> DummyMetricResult:
        expected = str(dataset_item.get("answer", ""))
        if expected and expected in llm_output:
            return DummyMetricResult(score_value)
        return DummyMetricResult(0.0)

    return metric


def test_opik_adapter_local_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = ChatPrompt(system="Answer", user="{input}")
    metric = _make_metric(1.0)

    batch = [
        OpikDataInst(
            input_text="Which?",
            answer="A",
            additional_context={},
            opik_item={"input": "Which?", "answer": "A"},
        )
    ]

    calls: list[str] = []
    monkeypatch.setattr(
        "gepa.adapters.opik_adapter.opik_adapter.evaluate_with_result",
        lambda *args, **kwargs: calls.append("called"),
    )

    tracker_count = {"count": 0}

    adapter = OpikAdapter(
        base_prompt=prompt,
        dataset=DummyDataset(),
        metric=metric,
        instantiate_agent=lambda _prompt, _project=None: DummyAgent(),
        prepare_experiment_config=lambda **_: {"project_name": "TestProject"},
        system_fallback="Answer",
        optimizer_metric_tracker=lambda: tracker_count.__setitem__(
            "count", tracker_count["count"] + 1
        ),
    )

    result = adapter.evaluate(batch, {"system_prompt": "Answer"}, capture_traces=True)

    assert not calls, "evaluate_with_result should not run when ids are missing"
    assert result.scores == [1.0]
    assert result.outputs == [{"output": "A"}]
    assert result.trajectories and result.trajectories[0]["score"] == 1.0
    assert tracker_count["count"] == 1


def test_opik_adapter_task_evaluator_used(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = ChatPrompt(system="Answer", user="{input}")
    metric = _make_metric(0.5)

    batch = [
        OpikDataInst(
            input_text="Which?",
            answer="A",
            additional_context={},
            opik_item={"id": "item-1", "input": "Which?", "answer": "A"},
        )
    ]

    def fake_evaluate_with_result(**kwargs: Any) -> tuple[float, Any]:
        evaluated_task = kwargs["evaluated_task"]
        output = evaluated_task(batch[0].opik_item)
        score_result = SimpleNamespace(name=metric.__name__, value=0.75)
        test_result = SimpleNamespace(
            test_case=SimpleNamespace(
                dataset_item_id=batch[0].opik_item["id"],
                task_output=output,
            ),
            score_results=[score_result],
        )
        return 0.75, SimpleNamespace(test_results=[test_result])

    monkeypatch.setattr(
        "gepa.adapters.opik_adapter.opik_adapter.evaluate_with_result",
        fake_evaluate_with_result,
    )

    tracker_count = {"count": 0}
    agent = DummyAgent()

    adapter = OpikAdapter(
        base_prompt=prompt,
        dataset=DummyDataset(),
        metric=metric,
        instantiate_agent=lambda _prompt, _project=None: agent,
        prepare_experiment_config=lambda **_: {"project_name": "TestProject"},
        system_fallback="Answer",
        optimizer_metric_tracker=lambda: tracker_count.__setitem__(
            "count", tracker_count["count"] + 1
        ),
    )

    result = adapter.evaluate(batch, {"system_prompt": "Answer"}, capture_traces=False)

    assert result.scores == [0.75]
    assert result.outputs == [{"output": "A"}]
    assert result.trajectories is None
    assert tracker_count["count"] == 1


def test_opik_adapter_reflective_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = ChatPrompt(system="Answer", user="{input}")
    metric = _make_metric(1.0)

    batch = [
        OpikDataInst(
            input_text="Which?",
            answer="A",
            additional_context={},
            opik_item={"id": "item-1", "input": "Which?", "answer": "A"},
        )
    ]

    def fake_evaluate_with_result(**kwargs: Any) -> tuple[float, Any]:
        evaluated_task = kwargs["evaluated_task"]
        output = evaluated_task(batch[0].opik_item)
        score_result = SimpleNamespace(name=metric.__name__, value=1.0)
        test_result = SimpleNamespace(
            test_case=SimpleNamespace(
                dataset_item_id=batch[0].opik_item["id"],
                task_output=output,
            ),
            score_results=[score_result],
        )
        return 1.0, SimpleNamespace(test_results=[test_result])

    monkeypatch.setattr(
        "gepa.adapters.opik_adapter.opik_adapter.evaluate_with_result",
        fake_evaluate_with_result,
    )

    tracker_count = {"count": 0}
    agent = DummyAgent()

    adapter = OpikAdapter(
        base_prompt=prompt,
        dataset=DummyDataset(),
        metric=metric,
        instantiate_agent=lambda _prompt, _project=None: agent,
        prepare_experiment_config=lambda **_: {"project_name": "TestProject"},
        system_fallback="Answer",
        optimizer_metric_tracker=lambda: tracker_count.__setitem__(
            "count", tracker_count["count"] + 1
        ),
    )

    eval_result = adapter.evaluate(batch, {"system_prompt": "Answer"}, capture_traces=True)
    reflective = adapter.make_reflective_dataset(
        {"system_prompt": "Answer"},
        eval_result,
        components_to_update=["system_prompt"],
    )

    assert reflective["system_prompt"]
    entry = reflective["system_prompt"][0]
    assert "Inputs" in entry
    assert "Generated Outputs" in entry
    assert "Feedback" in entry
    assert tracker_count["count"] == 1


def test_opik_adapter_sampling_selects_best_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = ChatPrompt(system="Answer", user="{input}", model_parameters={"n": 2})
    metric = _make_metric(1.0)

    batch = [
        OpikDataInst(
            input_text="Which?",
            answer="A",
            additional_context={},
            opik_item={"input": "Which?", "answer": "A"},
        )
    ]

    # Force local path so adapter chooses candidate itself.
    monkeypatch.setattr(
        "gepa.adapters.opik_adapter.opik_adapter.evaluate_with_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("force local path")),
    )

    agent = DummyCandidateAgent()
    adapter = OpikAdapter(
        base_prompt=prompt,
        dataset=DummyDataset(),
        metric=metric,
        instantiate_agent=lambda _prompt, _project=None: agent,
        prepare_experiment_config=lambda **_: {"project_name": "TestProject"},
        system_fallback="Answer",
        candidate_selection_policy="best_by_metric",
    )

    result = adapter.evaluate(batch, {"system_prompt": "Answer"}, capture_traces=False)
    assert result.outputs == [{"output": "A"}]
    assert result.scores == [1.0]
    assert agent.invoke_agent_candidates_calls


def test_opik_adapter_enables_tool_use_for_sampling() -> None:
    prompt = ChatPrompt(
        system="Answer",
        user="{input}",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ],
    )
    metric = _make_metric(1.0)
    agent = DummyCandidateAgent()

    adapter = OpikAdapter(
        base_prompt=prompt,
        dataset=DummyDataset(),
        metric=metric,
        instantiate_agent=lambda _prompt, _project=None: agent,
        prepare_experiment_config=lambda **_: {"project_name": "TestProject"},
        system_fallback="Answer",
        allow_tool_use=True,
    )

    batch = [
        OpikDataInst(
            input_text="Which?",
            answer="A",
            additional_context={},
            opik_item={"input": "Which?", "answer": "A"},
        )
    ]
    adapter.evaluate(batch, {"system_prompt": "Answer"}, capture_traces=False)
    assert agent.invoke_agent_candidates_calls
    assert agent.invoke_agent_candidates_calls[0]["allow_tool_use"] is True
