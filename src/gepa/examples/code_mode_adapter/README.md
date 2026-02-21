# Code Mode Adapter Examples

This folder intentionally keeps only two real examples.

## 1) Cloudflare MCP example (direct MCP endpoint)

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_cloudflare_example.py
```

Optional auth token:

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_cloudflare_example.py \
  --auth-bearer "$CODEMODE_TOKEN"
```

Expected output notes:

- case 1 should use `selected_tool=search`
- case 2 should use `selected_tool=execute` and return a `42` result payload
- average can be `0.5` with the current demo metric because case 1 checks for
  literal word matching in `final_answer`

## 2) Local MCP example (stdio)

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_local_stdio_example.py \
  --server-module your_mcp_server_module
```

Custom launch options:

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_local_stdio_example.py \
  --command python \
  --server-script /path/to/your_mcp_server.py
```

Or pass raw args directly:

```bash
python src/gepa/examples/code_mode_adapter/code_mode_mcp_local_stdio_example.py \
  --command python \
  --arg -m \
  --arg your_server_module
```

For a broader end-to-end demo set, see:
`https://github.com/SuperagenticAI/supercodemode`
