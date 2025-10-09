# MCP Adapter for GEPA

The MCP Adapter enables optimization of [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tool usage through GEPA's reflective mutation approach.

## Overview

This adapter optimizes:
- **Tool descriptions**: Improve how tools are described to the model
- **System prompts**: Optimize guidance for when and how to use tools
- **Tool usage patterns**: Learn better tool invocation strategies

## Installation

Install the MCP Python SDK:

```bash
pip install mcp
```

## Quick Start

### Option 1: Local Models (Ollama)

```python
import gepa
from gepa.adapters.mcp_adapter import MCPAdapter
from mcp import StdioServerParameters

# Configure MCP server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

# Create dataset
dataset = [
    {
        "user_query": "What's in the file notes.txt?",
        "tool_arguments": {"path": "/tmp/notes.txt"},
        "reference_answer": "Meeting at 3pm",
        "additional_context": {},
    },
    # ... more examples
]

# Create adapter with LOCAL Ollama models
adapter = MCPAdapter(
    server_params=server_params,
    tool_name="read_file",
    task_model="ollama/llama3.2:1b",  # Local model via Ollama, replace with your model 
    metric_fn=lambda item, output: 1.0 if item["reference_answer"] in output else 0.0,
)

# Optimize with local models - no API costs!
result = gepa.optimize(
    seed_candidate={"tool_description": "Read the contents of a file"},
    trainset=dataset[:20],
    valset=dataset[20:],
    adapter=adapter,
    reflection_lm="ollama/llama3.1:8b",  # Larger local model for reflection replace with our choice 
    max_metric_calls=150,
)

print("Optimized tool description:", result.best_candidate["tool_description"])
# Total cost: $0.00 - runs 100% locally!
```

**Setup for Ollama:**
```bash
Install Ollama: https://ollama.com

# Pull models
ollama pull llama3.1:8b
ollama pull llama3.2:1b
```

### Option 2: OpenAI API 

```python
# Same as above, but use OpenAI models
adapter = MCPAdapter(
    server_params=server_params,
    tool_name="read_file",
    task_model="openai/gpt-4o-mini",  # OpenAI API,  replace with your model choice 
    metric_fn=lambda item, output: 1.0 if item["reference_answer"] in output else 0.0,
)

result = gepa.optimize(
    seed_candidate={"tool_description": "Read the contents of a file"},
    trainset=dataset[:20],
    valset=dataset[20:],
    adapter=adapter,
    reflection_lm="openai/gpt-5",  # OpenAI for reflection, replace with yout model choice 
    max_metric_calls=150,
)
```

**Setup for OpenAI:**
```bash
export OPENAI_API_KEY=your-key-here
```

## Architecture

### Two-Pass Workflow

The adapter uses a two-pass workflow for better tool integration:

1. **First Pass**: Model receives user query and decides whether to call the tool
   - Input: User query + system prompt with tool info
   - Output: Tool call decision + arguments OR direct answer

2. **Second Pass**: Model receives tool response and generates final answer
   - Input: Original query + tool response
   - Output: Final answer incorporating tool results

This workflow ensures the model can effectively utilize tool outputs.

### Implementation Approach

The adapter uses `asyncio.run()` to bridge GEPA's synchronous API with MCP's async SDK:

```python
def evaluate(self, batch, candidate, capture_traces):
    # Run async evaluation in new event loop
    return asyncio.run(self._evaluate_async(batch, candidate, capture_traces))
```

Each evaluation creates a fresh MCP session, avoiding state management complexity.

**Performance Note**: Subprocess startup adds ~100-500ms per evaluation. For a typical optimization run with 150 metric calls, expect ~15-75 seconds of MCP overhead.

## Component Optimization

### Tool Description

Optimizes the description field of the MCP tool, improving how the model understands when and how to use it.

```python
seed_candidate = {
    "tool_description": "Search through documentation files"
}

# GEPA will optimize this to something like:
# "Search documentation by keyword. Use when user asks about API usage,
#  code examples, or feature documentation. Returns relevant doc sections."
```

### System Prompt

Optimizes the overall system prompt to provide better guidance on tool usage strategy.

```python
seed_candidate = {
    "tool_description": "Read file contents",
    "system_prompt": "You are a helpful assistant with file access."
}

# GEPA optimizes both components jointly
```

## Dataset Format

The `MCPDataInst` TypedDict defines the expected dataset format:

```python
{
    "user_query": str,              # User's question/request
    "tool_arguments": dict,          # Expected tool arguments
    "reference_answer": str | None,  # Reference answer for scoring
    "additional_context": dict,      # Additional context
}
```

Example:

```python
{
    "user_query": "Show me the config file",
    "tool_arguments": {"path": "/app/config.json"},
    "reference_answer": '{"debug": true}',
    "additional_context": {"file_location": "/app"},
}
```

## Metric Functions

The metric function scores model outputs. Higher scores are better.

### Simple Exact Match

```python
def exact_match(item, output):
    return 1.0 if item["reference_answer"] in output else 0.0
```

### Fuzzy Matching

```python
from difflib import SequenceMatcher

def fuzzy_match(item, output):
    ratio = SequenceMatcher(None, item["reference_answer"], output).ratio()
    return ratio  # 0.0 to 1.0
```

### LLM-as-Judge

```python
import litellm

def llm_judge(item, output):
    messages = [{
        "role": "user",
        "content": f"Rate this answer (0-1):\nQuestion: {item['user_query']}\n"
                   f"Reference: {item['reference_answer']}\nAnswer: {output}"
    }]
    response = litellm.completion(model="openai/gpt-4o", messages=messages)
    return float(response.choices[0].message.content)
```

## MCP Server Examples

### Filesystem Server

```python
from mcp import StdioServerParameters

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

adapter = MCPAdapter(
    server_params=server_params,
    tool_name="read_file",
    task_model="openai/gpt-4o-mini",
    metric_fn=exact_match,
)
```

### Custom Python Server

```python
# Create custom MCP server: my_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def search_docs(query: str) -> str:
    """Search documentation."""
    # Your search logic
    return f"Results for: {query}"

if __name__ == "__main__":
    mcp.run()
```

```python
# Use in GEPA
server_params = StdioServerParameters(
    command="python",
    args=["my_server.py"],
)

adapter = MCPAdapter(
    server_params=server_params,
    tool_name="search_docs",
    task_model="openai/gpt-4o-mini",
    metric_fn=custom_metric,
)
```

## Advanced Configuration

### Custom Model Functions

Instead of litellm model strings, you can provide a custom callable:

```python
def my_model(messages):
    # Your custom model logic
    return "response"

adapter = MCPAdapter(
    server_params=server_params,
    tool_name="my_tool",
    task_model=my_model,  # Custom callable
    metric_fn=my_metric,
)
```

### Disable Two-Pass Workflow

For simpler scenarios, disable the two-pass workflow:

```python
adapter = MCPAdapter(
    server_params=server_params,
    tool_name="my_tool",
    task_model="openai/gpt-4o-mini",
    metric_fn=my_metric,
    enable_two_pass=False,  # Single-pass only
)
```

### Error Handling

Configure failure scores for robustness:

```python
adapter = MCPAdapter(
    server_params=server_params,
    tool_name="my_tool",
    task_model="openai/gpt-4o-mini",
    metric_fn=my_metric,
    failure_score=0.0,  # Score for failed executions
)
```

## Reflective Dataset

The adapter generates reflective datasets for each component showing:

- Successful and failed tool calls
- Cases where tools should/shouldn't be called
- How well tool responses were utilized

Example reflective entry for `tool_description`:

```python
{
    "Inputs": {
        "user_query": "What's in config.json?",
        "tool_description": "Read file contents",
    },
    "Generated Outputs": {
        "tool_called": True,
        "tool_arguments": {"path": "config.json"},
        "final_answer": "The config file contains...",
    },
    "Feedback": "The tool was not called. Consider whether calling the tool would help..."
}
```

## Performance Notes

### Subprocess Overhead

Each `evaluate()` call spawns a new MCP server process:
- Startup time: ~100-500ms
- Total overhead for 150 evals: ~15-75 seconds

This is early development MVP and overhead is expected as MCP is async and GEPA is still syc but plan is to add following features later 
- Session pooling (reuse processes)
- Background event loop (persistent session)
- Async GEPA core (native async support)

## License

Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
