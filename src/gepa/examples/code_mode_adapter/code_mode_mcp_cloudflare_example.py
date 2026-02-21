#!/usr/bin/env python3
"""Code Mode adapter example using Cloudflare MCP over streamable HTTP.

Default MCP endpoint comes from Cloudflare's public MCP server URL.
"""

from __future__ import annotations

import argparse
from typing import Any

from gepa.adapters.code_mode_adapter import CodeModeAdapter
from gepa.adapters.code_mode_adapter.runners import MCPStreamableHTTPCodeModeRunner


def metric_fn(item: dict[str, Any], output: str) -> float:
    expected = item.get("reference_answer") or ""
    return 1.0 if expected and expected.lower() in output.lower() else 0.0


def build_dataset() -> list[dict[str, Any]]:
    return [
        {
            "user_query": "What tools are available for this task?",
            "reference_answer": "search",
            "additional_context": {},
        },
        {
            "user_query": "Calculate 17 + 25 using the execution tool.",
            "reference_answer": "42",
            "additional_context": {},
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA Code Mode adapter against Cloudflare MCP")
    parser.add_argument(
        "--endpoint",
        default="https://mcp.cloudflare.com/mcp",
        help="Cloudflare MCP streamable-http endpoint",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--auth-bearer",
        default="",
        help="Optional bearer token if your MCP endpoint requires Authorization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    headers: dict[str, str] = {}
    if args.auth_bearer:
        headers["Authorization"] = f"Bearer {args.auth_bearer}"

    runner = MCPStreamableHTTPCodeModeRunner(
        endpoint_url=args.endpoint,
        timeout_seconds=args.timeout,
        headers=headers,
    )
    adapter = CodeModeAdapter(runner=runner, metric_fn=metric_fn)

    candidate = {
        "system_prompt": "You are a coding assistant. Use MCP tools when needed.",
        "codemode_description": "Discover capabilities with search tools and execute with code tools.",
    }

    result = adapter.evaluate(batch=build_dataset(), candidate=candidate, capture_traces=True)

    print("scores:", result.scores)
    print("avg:", sum(result.scores) / len(result.scores))
    if result.trajectories:
        for i, traj in enumerate(result.trajectories, start=1):
            print(
                f"[{i}] selected_tool={traj['selected_tool']} "
                f"tool_calls={len(traj['tool_calls'])} "
                f"error={traj['error']} final={traj['final_answer']}"
            )


if __name__ == "__main__":
    main()
