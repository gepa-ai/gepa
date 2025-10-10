#!/usr/bin/env python3 -u
"""
Remote MCP Server Optimization

Demonstrates GEPA optimization with remote MCP servers.

What this shows:
- Connecting to remote MCP servers via SSE/HTTP
- Using public/hosted MCP tools
- Authentication with API headers
- Transport selection (SSE vs StreamableHTTP)

Requirements:
- Access to a remote MCP server URL
- Optional: API token for authentication

Compare with:
- cloud_api.py: Cloud models, local server
- local_ollama.py: 100% local, no network

Usage:
    python remote_server.py \
        --url https://mcp-server.com/sse \
        --transport sse \
        --tool-name search_web \
        --auth-header "Authorization: Bearer TOKEN"
"""

import argparse
import logging
import sys

import gepa
from gepa.adapters.mcp_adapter import MCPAdapter

# Enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
)

# Suppress verbose litellm logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def create_dataset_for_tool(tool_name: str):
    """
    Create a sample dataset based on the tool.

    In production, you would customize this based on your actual tool's capabilities.
    """
    # Generic dataset that works with many tools
    dataset = [
        {
            "user_query": "Test query 1",
            "tool_arguments": {},
            "reference_answer": "test",
            "additional_context": {},
        },
        {
            "user_query": "Test query 2",
            "tool_arguments": {},
            "reference_answer": "result",
            "additional_context": {},
        },
    ]
    return dataset


def simple_metric(item, output: str) -> float:
    """
    Simple metric: check if reference answer appears in output.

    For production, create a more sophisticated metric based on your use case.
    """
    if not output:
        return 0.0
    reference = item.get("reference_answer", "")
    if reference and reference.lower() in output.lower():
        return 1.0
    return 0.0


def parse_auth_header(header_str: str) -> dict[str, str]:
    """Parse header string like 'Authorization: Bearer token' into dict."""
    if ":" not in header_str:
        raise ValueError("Header must be in format 'Header-Name: value'")

    key, value = header_str.split(":", 1)
    return {key.strip(): value.strip()}


def main():
    parser = argparse.ArgumentParser(description="Remote MCP Tool Optimization Example")
    parser.add_argument("--url", required=True, help="Remote MCP server URL")
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable_http"],
        default="sse",
        help="Transport type (default: sse)",
    )
    parser.add_argument("--tool-name", required=True, help="Name of the tool to optimize")
    parser.add_argument(
        "--auth-header",
        help="Authentication header (e.g., 'Authorization: Bearer TOKEN')",
    )
    parser.add_argument(
        "--task-model",
        default="gpt-4o-mini",
        help="Task execution model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--reflection-model",
        default="gpt-4o",
        help="Reflection model for generating improvements (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=20,
        help="Maximum metric evaluations (default: 20)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GEPA Remote MCP Tool Optimization")
    print("=" * 60)

    print("\nRemote Server:")
    print(f"  URL: {args.url}")
    print(f"  Transport: {args.transport}")
    print(f"  Tool: {args.tool_name}")

    # Parse authentication header if provided
    headers = {}
    if args.auth_header:
        headers = parse_auth_header(args.auth_header)
        print(f"  Auth: {next(iter(headers.keys()))} header provided")

    print("\nModels:")
    print(f"  Task Model: {args.task_model}")
    print(f"  Reflection Model: {args.reflection_model}")

    # Create dataset
    dataset = create_dataset_for_tool(args.tool_name)
    trainset = dataset[: len(dataset) // 2] if len(dataset) > 1 else dataset
    valset = dataset[len(dataset) // 2 :] if len(dataset) > 1 else dataset

    print("\nDataset:")
    print(f"  Training: {len(trainset)} examples")
    print(f"  Validation: {len(valset)} examples")

    # Create adapter with remote MCP server
    print("\n" + "=" * 60)
    print("Initializing Remote MCP Adapter...")
    print("=" * 60)

    try:
        adapter = MCPAdapter(
            tool_name=args.tool_name,
            task_model=args.task_model,
            metric_fn=simple_metric,
            remote_url=args.url,
            remote_transport=args.transport,
            remote_headers=headers,
            remote_timeout=30,
            base_system_prompt=f"You are a helpful assistant with access to the {args.tool_name} tool.",
            enable_two_pass=True,
        )

        print("✓ Adapter initialized")

        # Seed candidate
        seed_candidate = {
            "tool_description": f"Use the {args.tool_name} tool to help answer questions.",
        }

        print("\nSeed Candidate:")
        print(f"  '{seed_candidate['tool_description']}'")

        # Run optimization
        print("\n" + "=" * 60)
        print("Starting Optimization...")
        print("=" * 60)

        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=args.reflection_model,
            max_metric_calls=args.max_metric_calls,
        )

        print("\n" + "=" * 60)
        print("✓ Optimization Complete!")
        print("=" * 60)

        print("\nBest Candidate:")
        for component, text in result.best_candidate.items():
            print(f"\n{component}:")
            print(f"  {text}")

        print("\nPerformance:")
        print(f"  Best Score: {result.val_aggregate_scores[result.best_idx]:.2f}")
        print(f"  Total Metric Calls: {result.total_metric_calls}")
        print(f"  Candidates Evaluated: {result.num_candidates}")

        print("\n" + "=" * 60)
        print("Before vs After")
        print("=" * 60)
        print(f"\nOriginal: {seed_candidate['tool_description']}")
        print(f"\nOptimized: {result.best_candidate['tool_description']}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()

        print("\n\nTroubleshooting:")
        print("1. Verify the remote MCP server is accessible")
        print("2. Check authentication headers are correct")
        print("3. Ensure the tool name matches available tools on the server")
        print("4. Try testing with a simple MCP client first")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
