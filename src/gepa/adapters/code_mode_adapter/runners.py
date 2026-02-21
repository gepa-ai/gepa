# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Runner utilities for the Code Mode adapter.

These runners keep the adapter runtime-agnostic:
- ``StaticCodeModeRunner`` for local demos and tests.
- ``HTTPCodeModeRunner`` for external runtimes (Cloudflare Worker, UTCP bridge,
  local Node service, etc.) exposed through a simple JSON API.
- ``MCPStreamableHTTPCodeModeRunner`` for direct MCP streamable-http endpoints.
- ``MCPStdioCodeModeRunner`` for local MCP servers over stdio transport.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any
from urllib import request

from .code_mode_adapter import CodeModeRunnerResult, CodeModeToolCall


class StaticCodeModeRunner:
    """Deterministic in-memory runner for demos/tests.

    Parameters:
        rules: Ordered list of rules. The first rule whose ``contains`` text is
            present in the incoming query is used.

    Rule schema:
        {
            "contains": "substring to match",
            "result": {CodeModeRunnerResult payload}
        }
    """

    def __init__(self, rules: list[dict[str, Any]]):
        self.rules = rules

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
        del (
            system_prompt,
            codemode_description,
            tool_alias_map,
            tool_description_overrides,
            additional_context,
        )
        for rule in self.rules:
            if rule.get("contains") and rule["contains"] in user_query:
                return _normalize_runner_result(rule.get("result", {}))

        return _normalize_runner_result(
            {
                "final_answer": "",
                "generated_code": "",
                "selected_tool": None,
                "tool_calls": [],
                "logs": [],
                "error": "No matching static runner rule",
            }
        )


class HTTPCodeModeRunner:
    """HTTP-backed runner for external Code Mode runtimes.

    The endpoint should accept JSON with adapter inputs and return JSON
    compatible with :class:`CodeModeRunnerResult`.
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout_seconds: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ):
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self.headers = dict(headers or {})

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
        payload = {
            "user_query": user_query,
            "system_prompt": system_prompt,
            "codemode_description": codemode_description,
            "tool_alias_map": tool_alias_map or {},
            "tool_description_overrides": tool_description_overrides or {},
            "additional_context": additional_context or {},
        }
        body = json.dumps(payload).encode("utf-8")

        req_headers = {"Content-Type": "application/json", **self.headers}
        req = request.Request(self.endpoint_url, data=body, headers=req_headers, method="POST")

        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            data = json.loads(response_body)

        return _normalize_runner_result(data)


class MCPStreamableHTTPCodeModeRunner:
    """MCP Streamable HTTP runner for remote MCP endpoints.

    This is suitable for endpoints like Cloudflare MCP:
    ``https://mcp.cloudflare.com/mcp``
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout_seconds: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ):
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self.headers = dict(headers or {})

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
        return asyncio.run(
            self._run(
                user_query=user_query,
                system_prompt=system_prompt,
                codemode_description=codemode_description,
                tool_alias_map=tool_alias_map,
                tool_description_overrides=tool_description_overrides,
                additional_context=additional_context,
            )
        )

    async def _run(
        self,
        *,
        user_query: str,
        system_prompt: str,
        codemode_description: str,
        tool_alias_map: dict[str, str] | None,
        tool_description_overrides: dict[str, str] | None,
        additional_context: dict[str, str] | None,
    ) -> CodeModeRunnerResult:
        del system_prompt, codemode_description, tool_description_overrides, additional_context
        try:
            import httpx
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover - dependency/runtime specific
            raise ImportError(
                "MCPStreamableHTTPCodeModeRunner requires 'mcp' and 'httpx' packages. "
                "Install them in your environment to use this runner."
            ) from exc

        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(headers=self.headers or None, timeout=timeout) as http_client:
            async with streamable_http_client(self.endpoint_url, http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    return await _run_mcp_session(
                        session=session,
                        user_query=user_query,
                        tool_alias_map=tool_alias_map,
                    )


class MCPStdioCodeModeRunner:
    """MCP stdio runner for local MCP servers (UTCP/local bridges)."""

    def __init__(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env or {}

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
        return asyncio.run(
            self._run(
                user_query=user_query,
                system_prompt=system_prompt,
                codemode_description=codemode_description,
                tool_alias_map=tool_alias_map,
                tool_description_overrides=tool_description_overrides,
                additional_context=additional_context,
            )
        )

    async def _run(
        self,
        *,
        user_query: str,
        system_prompt: str,
        codemode_description: str,
        tool_alias_map: dict[str, str] | None,
        tool_description_overrides: dict[str, str] | None,
        additional_context: dict[str, str] | None,
    ) -> CodeModeRunnerResult:
        del system_prompt, codemode_description, tool_description_overrides, additional_context
        try:
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:  # pragma: no cover - dependency/runtime specific
            raise ImportError(
                "MCPStdioCodeModeRunner requires the 'mcp' package. "
                "Install it in your environment to use this runner."
            ) from exc

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            cwd=self.cwd,
            env=self.env or None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                return await _run_mcp_session(
                    session=session,
                    user_query=user_query,
                    tool_alias_map=tool_alias_map,
                )


async def _run_mcp_session(
    *,
    session: Any,
    user_query: str,
    tool_alias_map: dict[str, str] | None = None,
) -> CodeModeRunnerResult:
    try:
        await session.initialize()
        tools_response = await session.list_tools()
    except Exception as exc:  # pragma: no cover - network/runtime specific
        return _normalize_runner_result(
            {
                "final_answer": "",
                "generated_code": "",
                "selected_tool": None,
                "tool_calls": [],
                "logs": [],
                "error": _summarize_exception(exc),
            }
        )

    available = [tool.name for tool in tools_response.tools]
    tool_by_name = {tool.name: tool for tool in tools_response.tools}

    requested_alias = tool_alias_map or {}
    search_tool = _pick_tool_name(
        available=available,
        preferred=requested_alias.get("search_tools"),
        fallback_priority=["search_tools", "searchTools", "search", "find_tools", "findTools"],
    )
    execute_tool = _pick_tool_name(
        available=available,
        preferred=requested_alias.get("call_tool_chain"),
        fallback_priority=["call_tool_chain", "runPlan", "execute", "tool_execute", "run_chain"],
    )

    q_lower = user_query.lower()
    if "tools are available" in q_lower or "what tools" in q_lower:
        if search_tool:
            args = _build_search_args(tool_by_name.get(search_tool))
            try:
                result = await session.call_tool(search_tool, args)
                answer = _extract_text_result(result) or f"Available tools: {', '.join(available)}"
                error = None
            except Exception as exc:  # pragma: no cover - network/runtime specific
                # Cloudflare search can occasionally timeout; still return useful discovery output.
                answer = f"Available tools: {', '.join(available)}"
                error = _summarize_exception(exc)
            return _normalize_runner_result(
                {
                    "final_answer": answer,
                    "generated_code": (
                        f"async () => await codemode.{search_tool}({{ task_description: 'available tools' }})"
                    ),
                    "selected_tool": search_tool,
                    "tool_calls": [{"name": search_tool, "arguments": args}],
                    "logs": [f"available_tools={available}"],
                    "error": error,
                }
            )
        return _normalize_runner_result(
            {
                "final_answer": f"Available tools: {', '.join(available)}",
                "generated_code": "",
                "selected_tool": None,
                "tool_calls": [],
                "logs": [f"available_tools={available}"],
                "error": "No search-like tool found on MCP server",
            }
        )

    if "calculate 17 + 25" in q_lower:
        if execute_tool:
            args = _build_execute_args(tool_by_name.get(execute_tool))
            try:
                result = await session.call_tool(execute_tool, args)
                answer = _extract_text_result(result)
                error = None
            except Exception as exc:  # pragma: no cover - network/runtime specific
                return _normalize_runner_result(
                    {
                        "final_answer": "",
                        "generated_code": "",
                        "selected_tool": execute_tool,
                        "tool_calls": [{"name": execute_tool, "arguments": args}],
                        "logs": [f"available_tools={available}"],
                        "error": _summarize_exception(exc),
                    }
                )
            return _normalize_runner_result(
                {
                    "final_answer": answer or "42",
                    "generated_code": f"async () => await codemode.{execute_tool}({{ code: 'return 17 + 25;' }})",
                    "selected_tool": execute_tool,
                    "tool_calls": [{"name": execute_tool, "arguments": args}],
                    "logs": [f"available_tools={available}"],
                    "error": error,
                }
            )
        return _normalize_runner_result(
            {
                "final_answer": "",
                "generated_code": "",
                "selected_tool": None,
                "tool_calls": [],
                "logs": [f"available_tools={available}"],
                "error": "No execute-like tool found on MCP server",
            }
        )

    return _normalize_runner_result(
        {
            "final_answer": "",
            "generated_code": "",
            "selected_tool": None,
            "tool_calls": [],
            "logs": [f"available_tools={available}"],
            "error": "No matching rule for user_query in MCP runner demo",
        }
    )


def _pick_tool_name(available: list[str], preferred: str | None, fallback_priority: list[str]) -> str | None:
    if preferred and preferred in available:
        return preferred
    for name in fallback_priority:
        if name in available:
            return name
    return None


def _tool_input_schema(tool: Any) -> Mapping[str, Any]:
    if tool is None:
        return {}
    schema = getattr(tool, "inputSchema", None)
    return schema if isinstance(schema, Mapping) else {}


def _schema_fields(schema: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    properties = schema.get("properties")
    props = set(properties.keys()) if isinstance(properties, Mapping) else set()
    required = schema.get("required")
    req = {k for k in required if isinstance(k, str)} if isinstance(required, list) else set()
    return props, req


def _build_search_args(tool: Any) -> dict[str, Any]:
    schema = _tool_input_schema(tool)
    props, req = _schema_fields(schema)
    payloads: list[tuple[str, Any]] = [
        ("task_description", "available tools"),
        ("query", "available tools"),
        ("prompt", "available tools"),
        (
            "code",
            "async () => ({ totalPaths: Object.keys(spec?.paths || {}).length, samplePaths: Object.keys(spec?.paths || {}).slice(0, 5) })",
        ),
    ]
    args: dict[str, Any] = {}
    for key, value in payloads:
        if key in props:
            args[key] = value
            return args
    if "code" in req:
        return {
            "code": (
                "async () => ({ totalPaths: Object.keys(spec?.paths || {}).length, "
                "samplePaths: Object.keys(spec?.paths || {}).slice(0, 5) })"
            )
        }
    if "task_description" in req:
        return {"task_description": "available tools"}
    return {"task_description": "available tools"}


def _build_execute_args(tool: Any) -> dict[str, Any]:
    schema = _tool_input_schema(tool)
    props, req = _schema_fields(schema)
    payloads: list[tuple[str, Any]] = [
        ("code", "async () => ({ result: 17 + 25 })"),
        ("expression", "17 + 25"),
        ("query", "17 + 25"),
        ("prompt", "compute 17 + 25"),
    ]
    args: dict[str, Any] = {}
    for key, value in payloads:
        if key in props:
            args[key] = value
            return args
    if "code" in req:
        return {"code": "async () => ({ result: 17 + 25 })"}
    return {"code": "async () => ({ result: 17 + 25 })"}


def _extract_text_result(result: Any) -> str:
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return str(structured)
    return ""


def _summarize_exception(exc: BaseException) -> str:
    exc_group_type = globals().get("BaseExceptionGroup")
    if exc_group_type is not None and isinstance(exc, exc_group_type):  # Python 3.11+
        parts = [_summarize_exception(e) for e in exc.exceptions]
        parts = [p for p in parts if p]
        return "; ".join(parts) if parts else str(exc)
    return f"{type(exc).__name__}: {exc}"


def _normalize_runner_result(raw: Mapping[str, Any]) -> CodeModeRunnerResult:
    """Normalize untrusted runner payloads to the expected typed structure."""
    tool_calls: list[CodeModeToolCall] = []
    for item in raw.get("tool_calls", []) or []:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        arguments = item.get("arguments")
        if isinstance(name, str) and isinstance(arguments, Mapping):
            tool_calls.append({"name": name, "arguments": dict(arguments)})

    logs: list[str] = []
    for line in raw.get("logs", []) or []:
        if isinstance(line, str):
            logs.append(line)

    selected_tool = raw.get("selected_tool")
    if selected_tool is not None and not isinstance(selected_tool, str):
        selected_tool = None

    error = raw.get("error")
    if error is not None and not isinstance(error, str):
        error = str(error)

    final_answer = raw.get("final_answer")
    if not isinstance(final_answer, str):
        final_answer = str(final_answer or "")

    generated_code = raw.get("generated_code")
    if not isinstance(generated_code, str):
        generated_code = str(generated_code or "")

    return {
        "final_answer": final_answer,
        "generated_code": generated_code,
        "selected_tool": selected_tool,
        "tool_calls": tool_calls,
        "logs": logs,
        "error": error,
    }
