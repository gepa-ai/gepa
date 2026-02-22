# Code Mode Adapter for GEPA

The Code Mode Adapter enables optimization of Code Mode text components while keeping runtime execution pluggable.

## What it optimizes

- `system_prompt`
- `codemode_description`
- `tool_alias_map` (optional JSON object string)
- `tool_description_overrides` (optional JSON object string)

## Why this adapter is simple

The adapter is intentionally thin and delegates runtime details to a `runner` callable. This lets you use:

- Cloudflare Code Mode runtimes
- Local Node runtimes
- UTCP/MCP-backed local servers
- Python-native fallback runtimes

without changing GEPA adapter logic.

Built-in runner helpers:

- `MCPStreamableHTTPCodeModeRunner` for direct MCP streamable-http endpoints (e.g. Cloudflare MCP)
- `MCPStdioCodeModeRunner` for local MCP servers over stdio (e.g. UTCP local)

## Runner contract

The runner must accept:

- `user_query: str`
- `system_prompt: str`
- `codemode_description: str`
- `tool_alias_map: dict[str, str] | None`
- `tool_description_overrides: dict[str, str] | None`
- `additional_context: dict[str, str] | None`

and return:

```python
{
    "final_answer": str,
    "generated_code": str,
    "selected_tool": str | None,
    "tool_calls": [{"name": str, "arguments": dict}],
    "logs": list[str],
    "error": str | None,
}
```

## Minimal usage

```python
from gepa.adapters.code_mode_adapter import CodeModeAdapter


def runner(*, user_query, system_prompt, codemode_description, additional_context=None):
    # Replace with your runtime (Cloudflare / UTCP / local Node / Python fallback)
    return {
        "final_answer": "42",
        "generated_code": "async () => 42",
        "tool_calls": [],
        "logs": [],
        "error": None,
    }


def metric_fn(item, output):
    return 1.0 if item["reference_answer"] and item["reference_answer"] in output else 0.0


adapter = CodeModeAdapter(
    runner=runner,
    metric_fn=metric_fn,
)

# Optional candidate components for tool-surface optimization
candidate = {
    "system_prompt": "You are a precise coding assistant.",
    "codemode_description": "Write code that plans and executes tools safely.",
    "tool_alias_map": '{"search_tools":"findTools","call_tool_chain":"runPlan"}',
    "tool_description_overrides": '{"call_tool_chain":"Execute a complete multi-step plan in one run."}',
}
```

## Notes

- Keep approval-required tools out of Code Mode flows if your runtime does not support approval within sandbox execution.
- Prefer concise traces: generated code, tool calls, logs, and errors are usually sufficient for reflection.
