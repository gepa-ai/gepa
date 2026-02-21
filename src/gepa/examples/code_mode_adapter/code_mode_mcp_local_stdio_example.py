#!/usr/bin/env python3
"""Code Mode adapter example using local MCP server over stdio.

Use this for local UTCP/local MCP workflows.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from gepa.adapters.code_mode_adapter import CodeModeAdapter
from gepa.adapters.code_mode_adapter.runners import MCPStdioCodeModeRunner


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
    parser = argparse.ArgumentParser(description="Run GEPA Code Mode adapter against local MCP stdio server")
    parser.add_argument("--command", default=sys.executable, help="MCP server command")
    parser.add_argument(
        "--server-module",
        default="",
        help="Python module to run with -m (used when --arg is not provided)",
    )
    parser.add_argument(
        "--server-script",
        default="",
        help="Optional server script path. If provided, used instead of --server-module",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Repeatable raw server args. Overrides --server-module/--server-script when set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    server_args = list(args.arg)
    if not server_args:
        if args.server_script:
            server_args = [args.server_script]
        elif args.server_module:
            server_args = ["-m", args.server_module]
        else:
            raise SystemExit(
                "Provide one of: --server-module <module>, --server-script <path>, "
                "or one or more --arg values."
            )

    runner = MCPStdioCodeModeRunner(
        command=args.command,
        args=server_args,
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
