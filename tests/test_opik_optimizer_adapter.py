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
