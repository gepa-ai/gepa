from typing import Any

from gepa.adapters.code_mode_adapter import CodeModeAdapter


class SuccessRunner:
    def __call__(
        self,
        *,
        user_query: str,
        system_prompt: str,
        codemode_description: str,
        tool_alias_map: dict[str, str] | None = None,
        tool_description_overrides: dict[str, str] | None = None,
        additional_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del (
            system_prompt,
            codemode_description,
            additional_context,
            tool_alias_map,
            tool_description_overrides,
        )
        if "17 + 25" in user_query:
            return {
                "final_answer": "42",
                "generated_code": "async () => await codemode.addNumbers({ a: 17, b: 25 })",
                "selected_tool": "addNumbers",
                "tool_calls": [{"name": "addNumbers", "arguments": {"a": 17, "b": 25}}],
                "logs": ["used addNumbers"],
                "error": None,
            }

        return {
            "final_answer": "unknown",
            "generated_code": "async () => 'unknown'",
            "selected_tool": None,
            "tool_calls": [],
            "logs": [],
            "error": None,
        }


class ErrorRunner:
    def __call__(
        self,
        *,
        user_query: str,
        system_prompt: str,
        codemode_description: str,
        tool_alias_map: dict[str, str] | None = None,
        tool_description_overrides: dict[str, str] | None = None,
        additional_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del (
            user_query,
            system_prompt,
            codemode_description,
            additional_context,
            tool_alias_map,
            tool_description_overrides,
        )
        raise RuntimeError("runner failed")


class CapturingRunner:
    def __init__(self) -> None:
        self.last_alias_map: dict[str, str] | None = None
        self.last_description_overrides: dict[str, str] | None = None

    def __call__(
        self,
        *,
        user_query: str,
        system_prompt: str,
        codemode_description: str,
        tool_alias_map: dict[str, str] | None = None,
        tool_description_overrides: dict[str, str] | None = None,
        additional_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del user_query, system_prompt, codemode_description, additional_context
        self.last_alias_map = tool_alias_map
        self.last_description_overrides = tool_description_overrides
        return {
            "final_answer": "42",
            "generated_code": "async () => 42",
            "selected_tool": "runPlan",
            "tool_calls": [{"name": "call_tool_chain", "arguments": {}}],
            "logs": [],
            "error": None,
        }


def metric_fn(item: dict[str, Any], output: str) -> float:
    expected = item.get("reference_answer") or ""
    return 1.0 if expected and expected in output else 0.0


def test_code_mode_adapter_evaluate_success() -> None:
    adapter = CodeModeAdapter(runner=SuccessRunner(), metric_fn=metric_fn)

    batch = [
        {
            "user_query": "What is 17 + 25?",
            "reference_answer": "42",
            "additional_context": {},
        }
    ]

    result = adapter.evaluate(batch=batch, candidate={}, capture_traces=True)

    assert result.scores == [1.0]
    assert result.outputs[0]["final_answer"] == "42"
    assert result.outputs[0]["tool_calls"][0]["name"] == "addNumbers"
    assert result.trajectories is not None
    assert result.trajectories[0]["score"] == 1.0


def test_code_mode_adapter_evaluate_failure_uses_failure_score() -> None:
    adapter = CodeModeAdapter(runner=ErrorRunner(), metric_fn=metric_fn, failure_score=0.25)

    batch = [
        {
            "user_query": "Any query",
            "reference_answer": "42",
            "additional_context": {},
        }
    ]

    result = adapter.evaluate(batch=batch, candidate={}, capture_traces=True)

    assert result.scores == [0.25]
    assert result.outputs[0]["error"] == "runner failed"
    assert result.trajectories is not None
    assert result.trajectories[0]["error"] == "runner failed"


def test_code_mode_adapter_reflective_dataset_for_components() -> None:
    adapter = CodeModeAdapter(runner=SuccessRunner(), metric_fn=metric_fn)

    batch = [
        {
            "user_query": "What is 17 + 25?",
            "reference_answer": "42",
            "additional_context": {},
        }
    ]

    candidate = {
        "system_prompt": "Be precise.",
        "codemode_description": "Use codemode tools.",
    }
    eval_batch = adapter.evaluate(batch=batch, candidate=candidate, capture_traces=True)

    reflective = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=eval_batch,
        components_to_update=["system_prompt", "codemode_description", "tool_alias_map"],
    )

    assert "system_prompt" in reflective
    assert "codemode_description" in reflective
    assert "tool_alias_map" in reflective
    assert len(reflective["system_prompt"]) == 1
    assert len(reflective["codemode_description"]) == 1
    assert len(reflective["tool_alias_map"]) == 1
    assert "Feedback" in reflective["system_prompt"][0]


def test_code_mode_adapter_uses_default_candidate_fields() -> None:
    adapter = CodeModeAdapter(
        runner=SuccessRunner(),
        metric_fn=metric_fn,
        default_system_prompt="default-system",
        default_codemode_description="default-codemode",
    )

    batch = [
        {
            "user_query": "unknown query",
            "reference_answer": "42",
            "additional_context": {},
        }
    ]

    result = adapter.evaluate(batch=batch, candidate={}, capture_traces=True)
    assert result.scores == [0.0]
    assert result.trajectories is not None
    assert result.trajectories[0]["system_prompt_used"] == "default-system"
    assert result.trajectories[0]["codemode_description_used"] == "default-codemode"


def test_code_mode_adapter_passes_alias_and_description_overrides_to_runner() -> None:
    runner = CapturingRunner()
    adapter = CodeModeAdapter(runner=runner, metric_fn=metric_fn)

    batch = [
        {
            "user_query": "Run a workflow",
            "reference_answer": "42",
            "additional_context": {},
        }
    ]
    candidate = {
        "tool_alias_map": '{"search_tools":"findTools","call_tool_chain":"runPlan"}',
        "tool_description_overrides": '{"call_tool_chain":"Execute a full code plan"}',
    }

    result = adapter.evaluate(batch=batch, candidate=candidate, capture_traces=True)

    assert result.scores == [1.0]
    assert runner.last_alias_map == {"search_tools": "findTools", "call_tool_chain": "runPlan"}
    assert runner.last_description_overrides == {"call_tool_chain": "Execute a full code plan"}
    assert result.trajectories is not None
    assert result.trajectories[0]["selected_tool"] == "runPlan"
    assert result.trajectories[0]["tool_alias_map_used"]["search_tools"] == "findTools"


def test_code_mode_adapter_invalid_json_maps_fallback_to_empty() -> None:
    runner = CapturingRunner()
    adapter = CodeModeAdapter(runner=runner, metric_fn=metric_fn)

    batch = [
        {
            "user_query": "Run a workflow",
            "reference_answer": "42",
            "additional_context": {},
        }
    ]
    candidate = {
        "tool_alias_map": "not-json",
        "tool_description_overrides": '["invalid"]',
    }

    adapter.evaluate(batch=batch, candidate=candidate, capture_traces=False)
    assert runner.last_alias_map == {}
    assert runner.last_description_overrides == {}
