# Code Mode Adapter Guide

The `CodeModeAdapter` lets GEPA optimize Code Mode text components while keeping
runtime execution pluggable.

## What gets optimized

- `system_prompt`
- `codemode_description`
- `tool_alias_map` (optional JSON object string)
- `tool_description_overrides` (optional JSON object string)

## Design principle

Use a **thin adapter, fat runner** architecture:

- Adapter handles scoring, traces, and reflective dataset formatting.
- Runner handles runtime specifics (Cloudflare, UTCP/MCP, local Node, etc.).

## Runner options

### Static runner (local demo)

Use `StaticCodeModeRunner` for deterministic examples and CI-safe tests.

### HTTP runner (real runtime bridge)

Use `HTTPCodeModeRunner` when your runtime is exposed via HTTP.
The endpoint should accept:

```json
{
  "user_query": "...",
  "system_prompt": "...",
  "codemode_description": "...",
  "tool_alias_map": {},
  "tool_description_overrides": {},
  "additional_context": {}
}
```

and return:

```json
{
  "final_answer": "...",
  "generated_code": "...",
  "selected_tool": "call_tool_chain",
  "tool_calls": [{"name": "call_tool_chain", "arguments": {}}],
  "logs": ["..."],
  "error": null
}
```

## Cloudflare/UTCP two-tool pattern

A common setup is:

- `search_tools` to discover capabilities
- `call_tool_chain` (or equivalent) to execute code plans

You can optimize model-facing tool semantics without changing backend tool IDs by
using `tool_alias_map` and `tool_description_overrides` in the candidate.

## Example files

- `src/gepa/examples/code_mode_adapter/code_mode_optimization_example.py`
- `src/gepa/examples/code_mode_adapter/code_mode_two_tool_showcase.py`

## Minimal snippet

```python
from gepa.adapters.code_mode_adapter import CodeModeAdapter
from gepa.adapters.code_mode_adapter.runners import HTTPCodeModeRunner

runner = HTTPCodeModeRunner(endpoint_url="http://localhost:8080/run-codemode")
adapter = CodeModeAdapter(runner=runner, metric_fn=my_metric)
```

Then run `gepa.optimize(...)` with your `seed_candidate` and dataset.
