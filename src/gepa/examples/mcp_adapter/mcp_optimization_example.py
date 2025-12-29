#!/usr/bin/env python3
"""
MCP Tool Optimization with GEPA

This example demonstrates how to use GEPA to optimize MCP tool descriptions
and system prompts. It shows both local (stdio) and remote (SSE) server support.

What you'll learn:
- Setting up MCPAdapter with local or remote servers
- Defining evaluation datasets
- Running optimization to improve tool descriptions
- Multi-tool support

MODEL CONFIGURATION:
Defaults to Ollama models (no API key needed). Requires Ollama installed: https://ollama.com
Pull models: ollama pull llama3.1:8b && ollama pull qwen3:8b

To use OpenAI models: --task-model gpt-4o-mini (requires OPENAI_API_KEY)

Requirements:
    pip install gepa mcp litellm

Usage Examples:
    # Run with default Ollama models (no flags needed)
    python mcp_optimization_example.py

    # Use OpenAI models (requires OPENAI_API_KEY)
    python mcp_optimization_example.py --task-model gpt-4o-mini

    # Remote MCP server
    python mcp_optimization_example.py --mode remote --url YOUR_URL
"""

import logging
import sys
import tempfile
from pathlib import Path

from mcp import StdioServerParameters

import gepa
from gepa.adapters.mcp_adapter import MCPAdapter

# Suppress verbose output from dependencies
try:
    import litellm
    litellm.set_verbose = False
    litellm.drop_params = True
    # Set LiteLLM logger to WARNING to suppress INFO messages
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
except ImportError:
    pass

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configure logging for GEPA output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Simple MCP Server (for local testing)
# ============================================================================

SIMPLE_MCP_SERVER = '''"""Simple MCP server with file operations."""
import asyncio
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("File Server")

# Base directory for file operations
BASE_DIR = Path("/tmp/mcp_test")
BASE_DIR.mkdir(exist_ok=True)


@mcp.tool()
def read_file(path: str) -> str:
    """Read contents of a file.

    Args:
        path: Relative path to the file
    """
    try:
        file_path = BASE_DIR / path
        if not file_path.exists():
            return f"Error: File {path} not found"
        return file_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: Relative path to the file
        content: Content to write
    """
    try:
        file_path = BASE_DIR / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@mcp.tool()
def list_files() -> str:
    """List all files in the base directory."""
    try:
        files = [str(p.relative_to(BASE_DIR)) for p in BASE_DIR.rglob("*") if p.is_file()]
        if not files:
            return "No files found"
        return "\\\\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"


if __name__ == "__main__":
    # Run the server
    mcp.run()
'''


def create_test_server():
    """Create a test MCP server file."""
    temp_dir = Path(tempfile.mkdtemp(prefix="gepa_mcp_"))
    server_file = temp_dir / "server.py"
    server_file.write_text(SIMPLE_MCP_SERVER)
    return server_file


def create_test_files():
    """Create test files for the example."""
    base_dir = Path("/tmp/mcp_test")
    base_dir.mkdir(exist_ok=True)

    (base_dir / "notes.txt").write_text("Meeting at 3pm in Room B\nDiscuss Q4 goals")
    (base_dir / "data.txt").write_text("Revenue: $50000\nExpenses: $30000\nProfit: $20000")


# ============================================================================
# Dataset & Metric Definition
# ============================================================================

def create_dataset():
    """Create evaluation dataset for file operations."""
    return [
        {
            "user_query": "What's in the notes.txt file?",
            "tool_arguments": {"path": "notes.txt"},
            "reference_answer": "3pm",
            "additional_context": {},
        },
        {
            "user_query": "Read the content of data.txt",
            "tool_arguments": {"path": "data.txt"},
            "reference_answer": "50000",
            "additional_context": {},
        },
        {
            "user_query": "Show me what's in notes.txt",
            "tool_arguments": {"path": "notes.txt"},
            "reference_answer": "Room B",
            "additional_context": {},
        },
    ]


def metric_fn(data_inst, output: str) -> float:
    """
    Simple metric: 1.0 if reference answer appears in output, 0.0 otherwise.

    In practice, you'd use more sophisticated metrics based on your use case.
    """
    reference = data_inst.get("reference_answer", "")
    return 1.0 if reference and reference.lower() in output.lower() else 0.0


# ============================================================================
# Local Server Example
# ============================================================================

def run_local_example(task_model: str = "ollama/llama3.1:8b", reflection_model: str = "ollama/qwen3:8b"):
    """Run optimization with local stdio MCP server."""
    logger.info("=" * 60)
    logger.info("GEPA MCP Tool Optimization")
    logger.info("=" * 60)

    server_file = create_test_server()
    create_test_files()

    logger.info(f"MCP Server: Local stdio server ({server_file.name})")
    logger.info("Tools: read_file")
    logger.info(f"Task Model: {task_model}")
    logger.info(f"Reflection Model: {reflection_model}")

    adapter = MCPAdapter(
        tool_names="read_file",
        task_model=task_model,
        metric_fn=metric_fn,
        server_params=StdioServerParameters(
            command="python",
            args=[str(server_file)],
        ),
        base_system_prompt="You are a helpful file assistant.",
        enable_two_pass=True,
    )

    dataset = create_dataset()
    seed_candidate = {
        "tool_description": "Read file contents from disk."
    }

    logger.info("")
    logger.info("Seed Prompt (Initial Tool Description):")
    logger.info(f"  {seed_candidate['tool_description']}")
    logger.info("")
    logger.info(f"Dataset: {len(dataset)} examples")
    logger.info("")
    logger.info("Starting GEPA optimization...")
    logger.info("-" * 60)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=dataset,
        valset=dataset,
        adapter=adapter,
        reflection_lm=reflection_model,
        max_metric_calls=10,
    )

    logger.info("-" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)
    best_score = result.val_aggregate_scores[result.best_idx] if result.val_aggregate_scores else 0.0
    best_candidate = result.candidates[result.best_idx]
    logger.info(f"Best Score: {best_score:.2f}")
    logger.info("")
    logger.info("Optimized Tool Description:")
    logger.info(f"  {best_candidate.get('tool_description', 'N/A')}")
    logger.info("=" * 60)

    return result


# ============================================================================
# Remote Server Example
# ============================================================================

def run_remote_example(url: str, task_model: str = "ollama/llama3.1:8b", reflection_model: str = "ollama/qwen3:8b"):
    """Run optimization with remote SSE MCP server."""
    logger.info("=" * 60)
    logger.info("GEPA MCP Tool Optimization")
    logger.info("=" * 60)

    logger.info(f"MCP Server: Remote SSE server ({url})")
    logger.info("Tools: search")
    logger.info(f"Task Model: {task_model}")
    logger.info(f"Reflection Model: {reflection_model}")

    adapter = MCPAdapter(
        tool_names="search",
        task_model=task_model,
        metric_fn=metric_fn,
        remote_url=url,
        remote_transport="sse",
        remote_headers={},
    )

    dataset = [
        {
            "user_query": "Search for information about Python",
            "tool_arguments": {"query": "Python"},
            "reference_answer": "programming",
            "additional_context": {},
        },
    ]

    seed_candidate = {
        "tool_description": "Search for information."
    }

    logger.info("")
    logger.info("Seed Prompt (Initial Tool Description):")
    logger.info(f"  {seed_candidate['tool_description']}")
    logger.info("")
    logger.info(f"Dataset: {len(dataset)} examples")
    logger.info("")
    logger.info("Starting GEPA optimization...")
    logger.info("-" * 60)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=dataset,
        valset=dataset,
        adapter=adapter,
        reflection_lm=reflection_model,
        max_metric_calls=10,
    )

    logger.info("-" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)
    best_score = result.val_aggregate_scores[result.best_idx] if result.val_aggregate_scores else 0.0
    best_candidate = result.candidates[result.best_idx]
    logger.info(f"Best Score: {best_score:.2f}")
    logger.info("")
    logger.info("Optimized Tool Description:")
    logger.info(f"  {best_candidate.get('tool_description', 'N/A')}")
    logger.info("=" * 60)

    return result


# ============================================================================
# Multi-Tool Example
# ============================================================================

def run_multitool_example(task_model: str = "ollama/llama3.1:8b", reflection_model: str = "ollama/qwen3:8b"):
    """Run optimization with multiple tools."""
    logger.info("=" * 60)
    logger.info("GEPA MCP Tool Optimization (Multi-Tool)")
    logger.info("=" * 60)

    server_file = create_test_server()
    create_test_files()

    logger.info(f"MCP Server: Local stdio server ({server_file.name})")
    logger.info("Tools: read_file, write_file, list_files")
    logger.info(f"Task Model: {task_model}")
    logger.info(f"Reflection Model: {reflection_model}")

    adapter = MCPAdapter(
        tool_names=["read_file", "write_file", "list_files"],
        task_model=task_model,
        metric_fn=metric_fn,
        server_params=StdioServerParameters(
            command="python",
            args=[str(server_file)],
        ),
    )

    dataset = [
        {
            "user_query": "What files are available?",
            "tool_arguments": {},
            "reference_answer": "notes.txt",
            "additional_context": {},
        },
        {
            "user_query": "Read notes.txt",
            "tool_arguments": {"path": "notes.txt"},
            "reference_answer": "3pm",
            "additional_context": {},
        },
    ]

    seed_candidate = {
        "tool_description_read_file": "Read a file.",
        "tool_description_write_file": "Write a file.",
        "tool_description_list_files": "List files.",
    }

    logger.info("")
    logger.info("Seed Prompts (Initial Tool Descriptions):")
    for tool_name in adapter.tool_names:
        key = f"tool_description_{tool_name}"
        logger.info(f"  {tool_name}: {seed_candidate.get(key, 'N/A')}")
    logger.info("")
    logger.info(f"Dataset: {len(dataset)} examples")
    logger.info("")
    logger.info("Starting GEPA optimization...")
    logger.info("-" * 60)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=dataset,
        valset=dataset,
        adapter=adapter,
        reflection_lm=reflection_model,
        max_metric_calls=10,
    )

    logger.info("-" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)
    best_candidate = result.candidates[result.best_idx]
    logger.info("Optimized Tool Descriptions:")
    for tool_name in adapter.tool_names:
        key = f"tool_description_{tool_name}"
        logger.info(f"  {tool_name}: {best_candidate.get(key, 'N/A')}")
    logger.info("=" * 60)

    return result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Tool Optimization Example")
    parser.add_argument(
        "--mode",
        choices=["local", "remote", "multitool"],
        default="local",
        help="Example mode to run",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Remote MCP server URL (for remote mode)",
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="ollama/llama3.1:8b",
        help='Model for task execution (default: "ollama/llama3.1:8b")',
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="ollama/qwen3:8b",
        help='Model for reflection (default: "ollama/qwen3:8b")',
    )

    args = parser.parse_args()

    try:
        if args.mode == "local":
            run_local_example(args.task_model, args.reflection_model)
        elif args.mode == "remote":
            if not args.url:
                logger.error("Remote mode requires --url argument")
                sys.exit(1)
            run_remote_example(args.url, args.task_model, args.reflection_model)
        elif args.mode == "multitool":
            run_multitool_example(args.task_model, args.reflection_model)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)
