#!/usr/bin/env python3 -u
"""
Local Ollama MCP Tool Optimization

Demonstrates GEPA optimization using 100% local models. Feel free to replace models of your choice.

What this shows:
- Using local Ollama models (llama3.1:8b for both tasks and reflection)
- Local Python MCP server
- Completely offline operation
- No API keys or external services
- Demonstrates MCP adapter with GEPA optimization

Requirements:
- Ollama installed (https://ollama.com)
- Model: ollama pull llama3.1:8b

Compare with:
- cloud_api.py: Cloud APIs, requires key
- remote_server.py: Remote MCP servers

Usage:
    ollama pull llama3.1:8b
    python local_ollama.py
"""

import logging

# Enable logging to see progress with unbuffered output
import sys
import tempfile
from pathlib import Path

from mcp import StdioServerParameters

import gepa
from gepa.adapters.mcp_adapter import MCPAdapter

sys.stdout = sys.stderr  # Redirect stdout to stderr (unbuffered)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # Log to stderr (unbuffered)
    force=True,
)

# Suppress verbose litellm logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
    """Create evaluation dataset with mix of explicit and implicit queries."""
    dataset = [
        # Explicit queries (easier - specify to read file)
        {
            "user_query": f"Read the file {temp_dir}/meeting_notes.txt and tell me what time is the team meeting scheduled?",
            "tool_arguments": {"path": f"{temp_dir}/meeting_notes.txt"},
            "reference_answer": "3pm",
            "additional_context": {},
        },
        {
            "user_query": f"Read {temp_dir}/server_config.json and tell me what port is the server configured to run on?",
            "tool_arguments": {"path": f"{temp_dir}/server_config.json"},
            "reference_answer": "8080",
            "additional_context": {},
        },
        # Implicit queries (harder - don't explicitly say "read")
        {
            "user_query": f"What time is the team meeting? Check {temp_dir}/meeting_notes.txt",
            "tool_arguments": {"path": f"{temp_dir}/meeting_notes.txt"},
            "reference_answer": "3pm",
            "additional_context": {},
        },
        {
            "user_query": f"Is debug mode enabled? File: {temp_dir}/server_config.json",
            "tool_arguments": {"path": f"{temp_dir}/server_config.json"},
            "reference_answer": "true",
            "additional_context": {},
        },
        {
            "user_query": f"Widget B revenue in February? Source: {temp_dir}/sales_data.csv",
            "tool_arguments": {"path": f"{temp_dir}/sales_data.csv"},
            "reference_answer": "25000",
            "additional_context": {},
        },
        {
            "user_query": f"Where is the meeting? ({temp_dir}/meeting_notes.txt)",
            "tool_arguments": {"path": f"{temp_dir}/meeting_notes.txt"},
            "reference_answer": "Conference Room B",
            "additional_context": {},
        },
        {
            "user_query": f"Widget A units sold in January - see {temp_dir}/sales_data.csv",
            "tool_arguments": {"path": f"{temp_dir}/sales_data.csv"},
            "reference_answer": "150",
            "additional_context": {},
        },
        {
            "user_query": f"Database host address from {temp_dir}/server_config.json?",
            "tool_arguments": {"path": f"{temp_dir}/server_config.json"},
            "reference_answer": "db.example.com",
            "additional_context": {},
        },
    ]
    return dataset


def simple_metric(item, output: str) -> float:
    """
    Stricter metric: checks multiple aspects for better differentiation.

    Returns:
    - 1.0 if exact reference answer found
    - 0.5 if partial match or related info
    - 0.0 if wrong or missing
    """
    if not output:
        return 0.0

    reference = item["reference_answer"].lower()
    output_lower = output.lower()

    # Exact match
    if reference in output_lower:
        return 1.0

    # Partial credit for numerical answers that are close
    if reference.isdigit():
        import re

        numbers = re.findall(r"\b\d+\b", output_lower)
        if reference in numbers:
            return 1.0
        elif numbers:
            return 0.3  # Found a number but wrong one

    # Partial credit for having related keywords
    keywords = reference.split()
    if len(keywords) > 1 and any(kw in output_lower for kw in keywords):
        return 0.3

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
        print("3. Pull model:")
        print("   ollama pull llama3.1:8b")
        return

    # Create test environment
    temp_dir = create_test_files()
    dataset = create_dataset(temp_dir)

    print(f"\n✓ Dataset: {len(dataset)} examples")
    print(f"  Example: '{dataset[0]['user_query']}'")

    # Configure MCP server (local filesystem)
    # Using our simple Python-based server instead of the npm version
    # which has issues with stdio communication
    import os
    import sys
    from pathlib import Path

    server_script = Path(__file__).parent / "simple_mcp_server.py"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script), temp_dir],
        env={**os.environ},
    )

    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print("\nMCP Server (Local Python):")
    print("  Server: simple_mcp_server.py")
    print("  Tool: read_file")
    print(f"  Directory: {temp_dir}")

    # Configure models - using Ollama via litellm
    # Litellm automatically detects ollama/ prefix
    # Using same 8B model for both - reliable but benefits from good descriptions
    task_model = "ollama/llama3.1:8b"  # 8B model for task execution
    reflection_model = "ollama/llama3.1:8b"  # 8B model for reflection/proposals

    print("\nModels (Local Ollama):")
    print(f"  Task Model: {task_model}")
    print(f"  Reflection Model: {reflection_model}")
    print("  No API keys required! Running 100% locally.")

    # Create adapter
    adapter = MCPAdapter(
        tool_name="read_file",
        task_model=task_model,
        metric_fn=simple_metric,
        server_params=server_params,  # Local stdio server
        base_system_prompt="You are a helpful file reading assistant. Answer questions based on file contents.",
        enable_two_pass=True,
    )

    # Seed candidate with basic but incomplete description
    # GEPA will improve this by adding specifics about file types, usage, etc.
    seed_candidate = {
        "tool_description": "Reads file contents. Provide the file path.",
    }

    print("\nSeed Candidate (basic):")
    print(f"  '{seed_candidate['tool_description']}'")

    # Split dataset - use more for training to give GEPA more examples
    trainset = dataset[:6]
    valset = dataset[6:]

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
            max_metric_calls=15,  # Smaller number for faster local demo
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
        print(f"  Total Metric Calls: {result.total_metric_calls if result.total_metric_calls else 'N/A'}")
        print(f"  Candidates Evaluated: {result.num_candidates}")

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
        print("2. Check model is pulled:")
        print("   ollama pull llama3.1:8b")
        print("3. Verify Python is available for MCP server")

    finally:
        print(f"\n\nTest files: {temp_dir}")
        print("(You can manually delete this directory when done)")


if __name__ == "__main__":
    # No API keys needed for local operation!
    print("\n🚀 Running with local models..\n", flush=True)
    main()
