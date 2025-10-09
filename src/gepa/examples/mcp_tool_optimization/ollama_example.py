#!/usr/bin/env python3
"""
MCP tool optimization example using LOCAL Ollama models.

This example demonstrates GEPA MCP optimization running entirely locally:
- Task model: Ollama (e.g., llama:3.2:1b, llama3.1:8b)
- Reflection model: Ollama (llama3.1:8b or larger model for better reasoning)
- MCP server: Local filesystem server

No API keys or external services required!

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull models:
       ollama pull llama:3.1:8b
       ollama pull llama:3.2:1b
    3. Install dependencies:
       pip install mcp gepa litellm

Usage:
    python ollama_example.py
"""

import tempfile
from pathlib import Path

from mcp import StdioServerParameters

import gepa
from gepa.adapters.mcp_adapter import MCPAdapter


def create_test_files():
    """Create temporary test files for the example."""
    temp_dir = tempfile.mkdtemp(prefix="gepa_mcp_ollama_")

    # Create sample files with clear, structured content
    files = {
        "meeting_notes.txt": """Team Meeting Notes - Oct 09, 2025

Attendees: Engineering team
Time: 3pm
Location: Conference Room B

Action Items:
- Complete API documentation by Friday
- Review pull requests before standup
- Schedule architecture review for next week
""",
        "server_config.json": """{
  "server": {
    "host": "localhost",
    "port": 8080,
    "debug_mode": true,
    "max_connections": 100
  },
  "database": {
    "host": "db.example.com",
    "port": 5432,
    "name": "production_db"
  }
}""",
        "project_readme.md": """# AI Assistant Project

This is a sample project demonstrating MCP tool integration with GEPA optimization.

## Features
- Local model support with Ollama
- MCP protocol integration
- Tool description optimization

## Getting Started
Run the examples in the examples/ directory.
""",
        "sales_data.csv": """month,product,revenue,units_sold
January,Widget A,15000,150
January,Widget B,22000,200
February,Widget A,18000,180
February,Widget B,25000,230
March,Widget A,20000,195
March,Widget B,28000,260
""",
    }

    for filename, content in files.items():
        filepath = Path(temp_dir) / filename
        filepath.write_text(content)

    print(f"✓ Created test files in: {temp_dir}")
    return temp_dir


def create_dataset(temp_dir: str):
    """Create evaluation dataset optimized for local models."""
    dataset = [
        {
            "user_query": "What time is the team meeting scheduled?",
            "tool_arguments": {"path": f"{temp_dir}/meeting_notes.txt"},
            "reference_answer": "3pm",
            "additional_context": {},
        },
        {
            "user_query": "What port is the server configured to run on?",
            "tool_arguments": {"path": f"{temp_dir}/server_config.json"},
            "reference_answer": "8080",
            "additional_context": {},
        },
        {
            "user_query": "Is debug mode enabled in the server configuration?",
            "tool_arguments": {"path": f"{temp_dir}/server_config.json"},
            "reference_answer": "true",
            "additional_context": {},
        },
        {
            "user_query": "What is this project about according to the README?",
            "tool_arguments": {"path": f"{temp_dir}/project_readme.md"},
            "reference_answer": "MCP tool integration",
            "additional_context": {},
        },
        {
            "user_query": "How much revenue did Widget B generate in February?",
            "tool_arguments": {"path": f"{temp_dir}/sales_data.csv"},
            "reference_answer": "25000",
            "additional_context": {},
        },
        {
            "user_query": "Where is the team meeting being held?",
            "tool_arguments": {"path": f"{temp_dir}/meeting_notes.txt"},
            "reference_answer": "Conference Room B",
            "additional_context": {},
        },
    ]
    return dataset


def simple_metric(item, output: str) -> float:
    """
    Simple metric: check if reference answer appears in output.

    Returns 1.0 if reference is found, 0.0 otherwise.
    """
    if not output:
        return 0.0

    reference = item["reference_answer"]
    if reference and reference.lower() in output.lower():
        return 1.0
    return 0.0


def check_ollama_setup():
    """Check if Ollama is running and models are available."""
    import subprocess

    print("\n" + "=" * 60)
    print("Checking Ollama Setup")
    print("=" * 60)

    # Check if ollama is installed
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            print("✗ Ollama is not running or not installed")
            print("\nPlease install Ollama from: https://ollama.com/")
            return False

        available_models = result.stdout
        print("✓ Ollama is running")
        print("\nAvailable models:")
        print(available_models)

        return True

    except FileNotFoundError:
        print("✗ Ollama not found in PATH")
        print("\nPlease install Ollama from: https://ollama.com/")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False


def main():
    """Run MCP tool optimization with local Ollama models."""
    print("=" * 60)
    print("GEPA MCP Tool Optimization - LOCAL OLLAMA EDITION")
    print("=" * 60)

    # Check Ollama setup
    if not check_ollama_setup():
        print("\nSetup instructions:")
        print("1. Install Ollama: https://ollama.com/")
        print("2. Start Ollama: ollama serve")
        print("3. Pull models:")
        print("   ollama pull llama:3.2:1b")
        print("   ollama pull llama:3.1:8b")
        return

    # Create test environment
    temp_dir = create_test_files()
    dataset = create_dataset(temp_dir)

    print(f"\n✓ Dataset: {len(dataset)} examples")
    print(f"  Example: '{dataset[0]['user_query']}'")

    # Configure MCP server (local filesystem)
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", temp_dir],
    )

    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print("\nMCP Server (Local):")
    print(f"  Command: {server_params.command}")
    print("  Tool: read_file")
    print(f"  Directory: {temp_dir}")

    # Configure models - using Ollama via litellm
    # Litellm automatically detects ollama/ prefix
    task_model = "ollama/llama:3.2:1b"  # OR Replace with Smaller, faster model for task execution
    reflection_model = "ollama/llama:3.1:8b"  # OR Replace with Larger model for better reasoning

    print("\nModels (Local Ollama):")
    print(f"  Task Model: {task_model}")
    print(f"  Reflection Model: {reflection_model}")
    print("  No API keys required! Running 100% locally.")

    # Create adapter
    adapter = MCPAdapter(
        server_params=server_params,
        tool_name="read_file",
        task_model=task_model,
        metric_fn=simple_metric,
        base_system_prompt="You are a helpful file reading assistant. Answer questions based on file contents.",
        enable_two_pass=True,
    )

    # Seed candidate with basic tool description
    seed_candidate = {
        "tool_description": "Read the contents of a file from the filesystem.",
    }

    print("\nSeed Candidate:")
    print(f"  '{seed_candidate['tool_description']}'")

    # Split dataset
    trainset = dataset[:4]
    valset = dataset[4:]

    print("\nDataset Split:")
    print(f"  Training: {len(trainset)} examples")
    print(f"  Validation: {len(valset)} examples")

    # Run optimization
    print("\n" + "=" * 60)
    print("Starting Local Optimization...")
    print("=" * 60)
    print("\nNote: Local models may take longer than API calls.")

    try:
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_model,
            max_metric_calls=20,  # Smaller number for local demo
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("✓ Optimization Complete!")
        print("=" * 60)

        print("\nBest Candidate:")
        for component, text in result.best_candidate.items():
            print(f"\n{component}:")
            print(f"  {text}")

        print("\nPerformance:")
        print(f"  Best Score: {result.best_score:.2f}")
        print(f"  Metric Calls: {len(result.pareto_frontier) if hasattr(result, 'pareto_frontier') else 'N/A'}")

        print("\n" + "=" * 60)
        print("Before vs After")
        print("=" * 60)
        print("\nOriginal Description:")
        print(f"  {seed_candidate['tool_description']}")
        print("\nOptimized Description:")
        print(f"  {result.best_candidate['tool_description']}")

        print("\n" + "=" * 60)
        print("Key Results")
        print("=" * 60)
        print("  - Task execution: Local Ollama model")
        print("  - Reflection/proposals: Local Ollama model")
        print("  - MCP server: Local filesystem")

    except Exception as e:
        print(f"\n✗ Error during optimization: {e}")
        import traceback

        traceback.print_exc()

        print("\n\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check models are pulled:")
        print("3. Verify npx/node is installed for MCP server")

    finally:
        print(f"\n\nTest files: {temp_dir}")
        print("(You can manually delete this directory when done)")


if __name__ == "__main__":
    # No API keys needed for local operation!
    print("\n🚀 Running with local models..\n")
    main()
