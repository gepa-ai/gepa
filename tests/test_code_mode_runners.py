from typing import Any

from gepa.adapters.code_mode_adapter.runners import (
    _build_execute_args,
    _build_search_args,
    _pick_tool_name,
    _summarize_exception,
)


def test_pick_tool_name_prefers_explicit_alias() -> None:
    selected = _pick_tool_name(
        available=["search_tools", "search"],
        preferred="search",
        fallback_priority=["search_tools", "search"],
    )
    assert selected == "search"


def test_pick_tool_name_falls_back_by_priority() -> None:
    selected = _pick_tool_name(
        available=["execute", "misc_tool"],
        preferred=None,
        fallback_priority=["call_tool_chain", "execute"],
    )
    assert selected == "execute"


class _Tool:
    def __init__(self, input_schema: dict[str, Any]):
        self.inputSchema = input_schema


def test_build_search_args_prefers_task_description() -> None:
    tool = _Tool({"type": "object", "properties": {"task_description": {"type": "string"}}})
    assert _build_search_args(tool) == {"task_description": "available tools"}


def test_build_search_args_uses_code_when_required() -> None:
    tool = _Tool({"type": "object", "required": ["code"]})
    args = _build_search_args(tool)
    assert "code" in args
    assert "async" in str(args["code"])


def test_build_execute_args_prefers_code() -> None:
    tool = _Tool({"type": "object", "properties": {"code": {"type": "string"}}})
    args = _build_execute_args(tool)
    assert "code" in args
    assert "async" in str(args["code"])


def test_build_execute_args_falls_back_to_expression() -> None:
    tool = _Tool({"type": "object", "properties": {"expression": {"type": "string"}}})
    assert _build_execute_args(tool) == {"expression": "17 + 25"}


def test_summarize_exception_simple() -> None:
    msg = _summarize_exception(RuntimeError("boom"))
    assert "RuntimeError" in msg
    assert "boom" in msg
