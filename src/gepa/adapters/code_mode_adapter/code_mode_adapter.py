# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Code Mode adapter for GEPA.

The adapter is intentionally thin: runtime-specific Code Mode execution lives in
an injected ``runner`` callable. This keeps the adapter portable across
Cloudflare, UTCP/MCP, local Node runtimes, and Python-native fallbacks.
"""

import json
import logging
from typing import Any, Callable, Protocol, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

logger = logging.getLogger(__name__)


class CodeModeDataInst(TypedDict):
    """Dataset item for Code Mode optimization."""

    user_query: str
    reference_answer: str | None
    additional_context: dict[str, str]


class CodeModeToolCall(TypedDict):
    """A tool call emitted during code execution."""

    name: str
    arguments: dict[str, Any]


class CodeModeRunnerResult(TypedDict):
    """Canonical output contract from a Code Mode runner."""

    final_answer: str
    generated_code: str
    selected_tool: str | None
    tool_calls: list[CodeModeToolCall]
    logs: list[str]
    error: str | None


class CodeModeTrajectory(TypedDict):
    """Execution trace captured by the adapter."""

    user_query: str
    system_prompt_used: str
    codemode_description_used: str
    tool_alias_map_used: dict[str, str]
    tool_description_overrides_used: dict[str, str]
    generated_code: str
    selected_tool: str | None
    tool_calls: list[CodeModeToolCall]
    logs: list[str]
    error: str | None
    final_answer: str
    score: float


class CodeModeOutput(TypedDict):
    """Per-example rollout output."""

    final_answer: str
    generated_code: str
    selected_tool: str | None
    tool_calls: list[CodeModeToolCall]
    logs: list[str]
    error: str | None


class CodeModeRunner(Protocol):
    """Runner protocol used by :class:`CodeModeAdapter`."""

    def __call__(
        self,
        *,
        user_query: str,
        system_prompt: str,
        codemode_description: str,
        tool_alias_map: dict[str, str] | None = None,
        tool_description_overrides: dict[str, str] | None = None,
        additional_context: dict[str, str] | None = None,
    ) -> CodeModeRunnerResult:
        """Execute one Code Mode request and return structured outputs."""


class CodeModeAdapter(GEPAAdapter[CodeModeDataInst, CodeModeTrajectory, CodeModeOutput]):
    """GEPA adapter for Code Mode optimization with pluggable runtimes.

    Candidate components optimized by default:
    - ``system_prompt``
    - ``codemode_description``
    - ``tool_alias_map`` (optional JSON object string)
    - ``tool_description_overrides`` (optional JSON object string)
    """

    def __init__(
        self,
        runner: CodeModeRunner,
        metric_fn: Callable[[CodeModeDataInst, str], float],
        default_system_prompt: str = "You are a helpful assistant.",
        default_codemode_description: str = "Execute code to solve the task using codemode tools.",
        failure_score: float = 0.0,
    ):
        self.runner = runner
        self.metric_fn = metric_fn
        self.default_system_prompt = default_system_prompt
        self.default_codemode_description = default_codemode_description
        self.failure_score = failure_score

    def evaluate(
        self,
        batch: list[CodeModeDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[CodeModeTrajectory, CodeModeOutput]:
        outputs: list[CodeModeOutput] = []
        scores: list[float] = []
        trajectories: list[CodeModeTrajectory] | None = [] if capture_traces else None

        system_prompt = candidate.get("system_prompt", self.default_system_prompt)
        codemode_description = candidate.get("codemode_description", self.default_codemode_description)
        tool_alias_map = self._parse_json_map(candidate.get("tool_alias_map"), "tool_alias_map")
        tool_description_overrides = self._parse_json_map(
            candidate.get("tool_description_overrides"),
            "tool_description_overrides",
        )

        for item in batch:
            try:
                result = self.runner(
                    user_query=item["user_query"],
                    system_prompt=system_prompt,
                    codemode_description=codemode_description,
                    tool_alias_map=tool_alias_map,
                    tool_description_overrides=tool_description_overrides,
                    additional_context=item.get("additional_context"),
                )

                final_answer = result.get("final_answer", "")
                score = self.metric_fn(item, final_answer)

                output: CodeModeOutput = {
                    "final_answer": final_answer,
                    "generated_code": result.get("generated_code", ""),
                    "selected_tool": result.get("selected_tool"),
                    "tool_calls": result.get("tool_calls", []),
                    "logs": result.get("logs", []),
                    "error": result.get("error"),
                }

                outputs.append(output)
                scores.append(score)

                if trajectories is not None:
                    trajectories.append(
                        {
                            "user_query": item["user_query"],
                            "system_prompt_used": system_prompt,
                            "codemode_description_used": codemode_description,
                            "tool_alias_map_used": tool_alias_map,
                            "tool_description_overrides_used": tool_description_overrides,
                            "generated_code": output["generated_code"],
                            "selected_tool": output["selected_tool"],
                            "tool_calls": output["tool_calls"],
                            "logs": output["logs"],
                            "error": output["error"],
                            "final_answer": final_answer,
                            "score": score,
                        }
                    )
            except Exception as exc:
                logger.exception("CodeMode runner failed for query: %s", item["user_query"])
                error_msg = str(exc)
                outputs.append(
                    {
                        "final_answer": "",
                        "generated_code": "",
                        "selected_tool": None,
                        "tool_calls": [],
                        "logs": [],
                        "error": error_msg,
                    }
                )
                scores.append(self.failure_score)

                if trajectories is not None:
                    trajectories.append(
                        {
                            "user_query": item["user_query"],
                            "system_prompt_used": system_prompt,
                            "codemode_description_used": codemode_description,
                            "tool_alias_map_used": tool_alias_map,
                            "tool_description_overrides_used": tool_description_overrides,
                            "generated_code": "",
                            "selected_tool": None,
                            "tool_calls": [],
                            "logs": [],
                            "error": error_msg,
                            "final_answer": "",
                            "score": self.failure_score,
                        }
                    )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[CodeModeTrajectory, CodeModeOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        reflective_data: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            examples: list[dict[str, Any]] = []

            for traj in eval_batch.trajectories or []:
                if component == "system_prompt":
                    inputs = {
                        "user_query": traj["user_query"],
                        "system_prompt": traj["system_prompt_used"],
                    }
                elif component == "codemode_description":
                    inputs = {
                        "user_query": traj["user_query"],
                        "codemode_description": traj["codemode_description_used"],
                    }
                elif component == "tool_alias_map":
                    inputs = {
                        "user_query": traj["user_query"],
                        "tool_alias_map": traj["tool_alias_map_used"],
                    }
                elif component == "tool_description_overrides":
                    inputs = {
                        "user_query": traj["user_query"],
                        "tool_description_overrides": traj["tool_description_overrides_used"],
                    }
                else:
                    continue

                examples.append(
                    {
                        "Inputs": inputs,
                        "Generated Outputs": {
                            "generated_code": traj["generated_code"],
                            "selected_tool": traj["selected_tool"],
                            "tool_calls": traj["tool_calls"],
                            "final_answer": traj["final_answer"],
                            "error": traj["error"],
                        },
                        "Feedback": self._generate_feedback(traj),
                    }
                )

            reflective_data[component] = examples

        return reflective_data

    def _generate_feedback(self, traj: CodeModeTrajectory) -> str:
        if traj["error"]:
            return (
                f"Execution failed with error: {traj['error']}. "
                "Improve instructions so generated code is executable and robust."
            )

        if traj["score"] > 0.5:
            return (
                f"Good result (score={traj['score']:.2f}). "
                "Preserve this behavior while keeping code concise and deterministic."
            )

        tool_count = len(traj["tool_calls"])
        selected_tool = traj.get("selected_tool")
        selected_tool_feedback = f" selected_tool={selected_tool}." if selected_tool else ""
        return (
            f"Low score (score={traj['score']:.2f}) with {tool_count} tool call(s).{selected_tool_feedback} "
            "Improve planning and code-generation instructions to produce a more accurate final answer."
        )

    def _parse_json_map(self, raw_value: str | None, field_name: str) -> dict[str, str]:
        """Parse a JSON object string candidate field into ``dict[str, str]``.

        Invalid JSON or non-object payloads are treated as empty maps to keep
        per-example evaluation resilient.
        """
        if not raw_value:
            return {}

        try:
            parsed = json.loads(raw_value)
            if not isinstance(parsed, dict):
                logger.warning("%s must decode to a JSON object; got %s", field_name, type(parsed).__name__)
                return {}

            result: dict[str, str] = {}
            for k, v in parsed.items():
                if isinstance(k, str) and isinstance(v, str):
                    result[k] = v
            return result
        except Exception:
            logger.warning("Failed to parse %s as JSON object", field_name)
            return {}
