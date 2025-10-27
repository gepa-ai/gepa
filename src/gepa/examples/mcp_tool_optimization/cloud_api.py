#!/usr/bin/env python3
"""
Cloud API MCP Tool Optimization

Demonstrates GEPA optimization using cloud-based LLM APIs. You can extend this to your cloud providers or models.

What this shows:
- Using cloud APIs (OpenAI GPT-4o-mini or whatever cloud based model)
- Local MCP server (npx filesystem server)
- Production-quality optimization

Requirements:
- API key (export OPENAI_API_KEY=your-key or simialar cloud based provider)
- Node.js for npx


Usage:
    export OPENAI_API_KEY=your-key
    python cloud_api.py
"""

import logging
import os
import tempfile
from pathlib import Path

from mcp import StdioServerParameters

import gepa
from gepa.adapters.mcp_adapter import MCPAdapter

# Suppress verbose litellm logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def create_test_files():
    """Create temporary test files for the example."""
    temp_dir = tempfile.mkdtemp(prefix="gepa_mcp_example_")

    # Create sample files
    files = {
        "notes.txt": "Meeting scheduled for 3pm tomorrow with the engineering team.",
        "config.json": '{"debug": true, "port": 8080, "host": "localhost"}',
        "readme.md": "# Project Documentation\n\nThis is a sample project for testing MCP tools.",
        "data.csv": "name,age,city\nAlice,30,NYC\nBob,25,SF\nCarol,35,LA",
    }

    for filename, content in files.items():
        filepath = Path(temp_dir) / filename
        filepath.write_text(content)

    print(f"Created test files in: {temp_dir}")
    return temp_dir


def create_dataset(temp_dir: str):
    """Create evaluation dataset for MCP tool optimization."""
    dataset = [
        {
            "user_query": "What's in the notes.txt file?",
            "tool_arguments": {"path": f"{temp_dir}/notes.txt"},
            "reference_answer": "3pm tomorrow",
            "additional_context": {},
        },
        {
            "user_query": "Show me the configuration settings",
            "tool_arguments": {"path": f"{temp_dir}/config.json"},
            "reference_answer": "port: 8080",
            "additional_context": {},
        },
        {
            "user_query": "Read the project documentation",
            "tool_arguments": {"path": f"{temp_dir}/readme.md"},
            "reference_answer": "sample project",
            "additional_context": {},
        },
        {
            "user_query": "What data is in the CSV file?",
            "tool_arguments": {"path": f"{temp_dir}/data.csv"},
            "reference_answer": "Alice",
            "additional_context": {},
        },
        {
            "user_query": "Tell me about the debug setting in config",
            "tool_arguments": {"path": f"{temp_dir}/config.json"},
            "reference_answer": "debug",
            "additional_context": {},
        },
    ]
    return dataset


def simple_metric(item, output: str) -> float:
    """
    Simple metric: check if reference answer appears in output.

    Returns 1.0 if reference is found, 0.0 otherwise.
    """
    if item["reference_answer"] and item["reference_answer"].lower() in output.lower():
        return 1.0
    return 0.0


def main():
    """Run MCP tool optimization example."""
    print("=" * 60)
    print("GEPA MCP Tool Optimization Example")
    print("=" * 60)

    # Create test environment
    temp_dir = create_test_files()
    dataset = create_dataset(temp_dir)

    print(f"\nDataset size: {len(dataset)} examples")
    print(f"Example query: {dataset[0]['user_query']}")

    # Configure MCP server (filesystem server)
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", temp_dir],
    )

    print("\nMCP Server Configuration:")
    print(f"  Command: {server_params.command}")
    print(f"  Args: {' '.join(server_params.args)}")

    # Create adapter
    adapter = MCPAdapter(
        server_params=server_params,
        tool_name="read_file",
        task_model="openai/gpt-4o-mini",
        metric_fn=simple_metric,
        base_system_prompt="You are a helpful file assistant.",
        enable_two_pass=True,
    )

    print("\nAdapter Configuration:")
    print("  Tool: read_file")
    print("  Model: openai/gpt-4o-mini")
    print("  Two-pass: Enabled")

    # Seed candidate with basic tool description
    seed_candidate = {
        "tool_description": "Read the contents of a file from the filesystem.",
    }

    print("\nSeed Candidate:")
    print(f"  Tool Description: {seed_candidate['tool_description']}")

    # Split dataset
    trainset = dataset[:3]
    valset = dataset[3:]

    print("\nDataset Split:")
    print(f"  Training: {len(trainset)} examples")
    print(f"  Validation: {len(valset)} examples")

    # Run optimization
    print("\n" + "=" * 60)
    print("Starting Optimization...")
    print("=" * 60 + "\n")

    try:
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm="openai/gpt-4o",  # Just an example replace your model
            max_metric_calls=30,  # Small number for quick demo
        )

        print("\n" + "=" * 60)
        print("Optimization Complete!")
        print("=" * 60)

        print("\nBest Candidate:")
        print(f"  Tool Description: {result.best_candidate['tool_description']}")

        print("\nPerformance:")
        print(f"  Best Score: {result.best_score:.2f}")

        print("\n" + "=" * 60)
        print("Comparison:")
        print("=" * 60)
        print(f"\nOriginal: {seed_candidate['tool_description']}")
        print(f"\nOptimized: {result.best_candidate['tool_description']}")

    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print(f"\n\nTest files remain in: {temp_dir}")
        print("(You can manually delete this directory when done)")


if __name__ == "__main__":
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your-key-here")
        exit(1)

    main()
