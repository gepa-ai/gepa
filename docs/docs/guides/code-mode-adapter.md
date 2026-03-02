# Code Mode Adapter Guide

The `CodeModeAdapter` lets GEPA optimize Code Mode text components while keeping
runtime execution pluggable.

## What Code Mode means here

Code Mode here refers to a code-first MCP orchestration pattern where the model
uses a small tool surface (often discovery + execution) and generates code for
multi-step work.

Background references:

- Cloudflare MCP Code Mode blog: `https://blog.cloudflare.com/code-mode-mcp/`
- UTCP Code Mode implementation: `https://github.com/universal-tool-calling-protocol/code-mode`

## What gets optimized

- `system_prompt`
- `codemode_description`
- `tool_alias_map` (optional JSON object string)
- `tool_description_overrides` (optional JSON object string)

## Design principle

Use a thin adapter and transport-specific runners:

- Adapter handles scoring, traces, and reflective dataset formatting.
- Runner handles runtime specifics.

## Built-in runner helpers (current scope)

### `MCPStreamableHTTPCodeModeRunner`

Use for remote MCP endpoints over streamable HTTP, such as Cloudflare MCP:

- `https://mcp.cloudflare.com/mcp`

### `MCPStdioCodeModeRunner`

Use for local MCP servers over stdio (for example local UTCP or custom MCP servers).

## Cloudflare and local MCP pattern

A common setup is a two-tool flow:

- discovery tool (for example `search` or `search_tools`)
- execution tool (for example `execute` or `call_tool_chain`)

The MCP runners inspect tool names and input schemas and adapt to common variants.

## Example files (kept minimal)

- `src/gepa/examples/code_mode_adapter/code_mode_mcp_cloudflare_example.py`
- `src/gepa/examples/code_mode_adapter/code_mode_mcp_local_stdio_example.py`

## Example commands

### Cloudflare MCP example

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_cloudflare_example.py \
  --auth-bearer "$CODEMODE_TOKEN"
```

Expected demo behavior:

- case 1 selects `search`
- case 2 selects `execute` and returns `42`
- average may be `0.5` because case 1 often returns structured JSON and the demo metric is string-match based

### Local MCP stdio example

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_local_stdio_example.py \
  --server-module your_mcp_server_module
```

Alternative launch forms:

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_local_stdio_example.py \
  --command python \
  --server-script /path/to/your_local_mcp_server.py
```

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_local_stdio_example.py \
  --command python \
  --arg -m \
  --arg your_server_module
```

## Minimal snippets

### Cloudflare MCP

```python
from gepa.adapters.code_mode_adapter import CodeModeAdapter
from gepa.adapters.code_mode_adapter.runners import MCPStreamableHTTPCodeModeRunner

runner = MCPStreamableHTTPCodeModeRunner(endpoint_url="https://mcp.cloudflare.com/mcp")
adapter = CodeModeAdapter(runner=runner, metric_fn=my_metric)
```

### Local MCP (stdio)

```python
from gepa.adapters.code_mode_adapter import CodeModeAdapter
from gepa.adapters.code_mode_adapter.runners import MCPStdioCodeModeRunner

runner = MCPStdioCodeModeRunner(command="python", args=["-m", "your_mcp_server_module"])
adapter = CodeModeAdapter(runner=runner, metric_fn=my_metric)
```

Then run `gepa.optimize(...)` with your dataset and seed candidate.

## Notes

- Cloudflare MCP example may show an average score of `0.5` with the demo metric even when integration is working (case 1 often returns structured JSON from `search`).
- Use tool selection and response structure as the primary integration signal in the demo example.
