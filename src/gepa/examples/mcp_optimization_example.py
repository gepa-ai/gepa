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

Requirements:
    pip install gepa mcp litellm

For local example:
    - Create a simple MCP server (see simple_mcp_server.py below)
    - Run: python mcp_optimization_example.py --mode local

For remote example:
    - Set up a remote MCP server with SSE endpoint
    - Run: python mcp_optimization_example.py --mode remote --url YOUR_URL
"""

import logging
import sys
import tempfile
from pathlib import Path

from mcp import StdioServerParameters

import gepa
from gepa.adapters.mcp_adapter import MCPAdapter

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
        return "\\n".join(files)
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
    logger.info(f"Created test server at: {server_file}")
    return server_file


def create_test_files():
    """Create test files for the example."""
    base_dir = Path("/tmp/mcp_test")
    base_dir.mkdir(exist_ok=True)

    (base_dir / "notes.txt").write_text("Meeting at 3pm in Room B\\nDiscuss Q4 goals")
    (base_dir / "data.txt").write_text("Revenue: $50000\\nExpenses: $30000\\nProfit: $20000")

    logger.info(f"Created test files in: {base_dir}")


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

def run_local_example():
    """Run optimization with local stdio MCP server."""
    logger.info("=" * 60)
    logger.info("LOCAL MCP SERVER EXAMPLE")
    logger.info("=" * 60)

    # Create test server and files
    server_file = create_test_server()
    create_test_files()

    # Create adapter with local server
    adapter = MCPAdapter(
        tool_names="read_file",  # Single tool
        task_model="gpt-4o-mini",  # Use any litellm-compatible model
        metric_fn=metric_fn,
        server_params=StdioServerParameters(
            command="python",
            args=[str(server_file)],
        ),
        base_system_prompt="You are a helpful file assistant.",
        enable_two_pass=True,
    )

    # Create dataset
    dataset = create_dataset()

    # Define seed candidate (initial tool description)
    seed_candidate = {
        "tool_description": "Read file contents from disk."
    }

    logger.info("\\nRunning optimization...")
    logger.info(f"Dataset size: {len(dataset)} examples")
    logger.info(f"Initial tool description: {seed_candidate['tool_description']}")

    # Run GEPA optimization
    optimizer = gepa.GEPA(
        adapter=adapter,
        n_iterations=2,  # Small number for demo
        batch_size=2,
    )

    result = optimizer.optimize(
        train_dataset=dataset,
        val_dataset=dataset,  # In practice, use separate validation set
        seed_candidate=seed_candidate,
        components_to_update=["tool_description"],
    )

    logger.info("\\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best score: {result.best_score:.2f}")
    logger.info(f"Optimized tool description: {result.best_candidate.get('tool_description', 'N/A')}")

    return result


# ============================================================================
# Remote Server Example
# ============================================================================

def run_remote_example(url: str):
    """Run optimization with remote SSE MCP server."""
    logger.info("=" * 60)
    logger.info("REMOTE MCP SERVER EXAMPLE (SSE)")
    logger.info("=" * 60)

    # Create adapter with remote server
    adapter = MCPAdapter(
        tool_names="search",  # Example: search tool
        task_model="gpt-4o-mini",
        metric_fn=metric_fn,
        remote_url=url,
        remote_transport="sse",
        remote_headers={
            # Add auth headers if needed
            # "Authorization": "Bearer YOUR_TOKEN"
        },
    )

    # Define your dataset based on remote server's capabilities
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

    logger.info(f"\\nConnecting to remote server: {url}")
    logger.info(f"Dataset size: {len(dataset)} examples")

    # Run optimization
    optimizer = gepa.GEPA(
        adapter=adapter,
        n_iterations=2,
    )

    result = optimizer.optimize(
        train_dataset=dataset,
        val_dataset=dataset,
        seed_candidate=seed_candidate,
        components_to_update=["tool_description"],
    )

    logger.info("\\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best score: {result.best_score:.2f}")
    logger.info(f"Optimized description: {result.best_candidate.get('tool_description', 'N/A')}")

    return result


# ============================================================================
# Multi-Tool Example
# ============================================================================

def run_multitool_example():
    """Run optimization with multiple tools."""
    logger.info("=" * 60)
    logger.info("MULTI-TOOL EXAMPLE")
    logger.info("=" * 60)

    server_file = create_test_server()
    create_test_files()

    # Create adapter with multiple tools
    adapter = MCPAdapter(
        tool_names=["read_file", "write_file", "list_files"],  # Multiple tools
        task_model="gpt-4o-mini",
        metric_fn=metric_fn,
        server_params=StdioServerParameters(
            command="python",
            args=[str(server_file)],
        ),
    )

    # Dataset with queries requiring different tools
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

    # Optimize descriptions for each tool
    seed_candidate = {
        "tool_description_read_file": "Read a file.",
        "tool_description_write_file": "Write a file.",
        "tool_description_list_files": "List files.",
    }

    logger.info(f"\\nOptimizing {len(adapter.tool_names)} tools...")

    optimizer = gepa.GEPA(adapter=adapter, n_iterations=2)

    result = optimizer.optimize(
        train_dataset=dataset,
        val_dataset=dataset,
        seed_candidate=seed_candidate,
        components_to_update=[
            "tool_description_read_file",
            "tool_description_write_file",
            "tool_description_list_files",
        ],
    )

    logger.info("\\n" + "=" * 60)
    logger.info("MULTI-TOOL OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    for tool_name in adapter.tool_names:
        key = f"tool_description_{tool_name}"
        logger.info(f"{tool_name}: {result.best_candidate.get(key, 'N/A')}")

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

    args = parser.parse_args()

    try:
        if args.mode == "local":
            run_local_example()
        elif args.mode == "remote":
            if not args.url:
                logger.error("Remote mode requires --url argument")
                sys.exit(1)
            run_remote_example(args.url)
        elif args.mode == "multitool":
            run_multitool_example()

    except KeyboardInterrupt:
        logger.info("\\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)
