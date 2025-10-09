# MCP Tool Optimization Examples

This directory contains examples demonstrating how to use GEPA's MCP Adapter to optimize tool usage.

## Quick Start

Choose your setup:

| Example | Models | API Keys | Best For |
|---------|--------|----------|----------|
| **ollama_example.py** | 🏠 Local Ollama | ❌ None | 
| **simple_example.py** | ☁️ OpenAI API | ✅ Required |

## Prerequisites

### For Local (Ollama) Setup - RECOMMENDED for getting started!

```bash
# 1. Install dependencies
pip install mcp gepa litellm

# 2. Install Ollama
# Visit: https://ollama.com/

# 3. Pull models (do this while installing other tools)
# Replace with your local models with ollama list if needed 
ollama pull llama:3.2:1b   # Task model 
ollama pull llama:3.1:8b  # Reflection model 
```

### For OpenAI Setup

```bash
# 1. Install dependencies
pip install mcp gepa litellm

# 2. Set API key
export OPENAI_API_KEY=your-key-here
```

## Examples

### ollama_example.py ⭐ RECOMMENDED

**100% local optimization - no API keys, no cost, maximum privacy!**

**What it does:**
- Runs entirely on your machine with Ollama
- Creates temporary test files
- Sets up MCP filesystem server (local)
- Optimizes the `read_file` tool description
- Shows before/after comparison
- **Total cost: $0.00**

**Run it:**

```bash
# Make sure Ollama is running
ollama serve  # In a separate terminal

# Run the example
python ollama_example.py
```

**Expected runtime:** ~3-10 minutes (depends on your hardware)

**Models used:**
- Task model: `llama:3.2:1b` (smaller, faster for task execution)
- Reflection model: `llama:3.1:8b` (larger, better reasoning for proposals)

**System requirements:**
- RAM: 16GB+ recommended (32GB for comfortable use)
- Disk: ~30GB for models
- CPU/GPU: Faster is better, GPU highly recommended

### simple_example.py

**Fast optimization with OpenAI API**

**What it does:**
- Creates temporary test files
- Sets up MCP filesystem server
- Optimizes the `read_file` tool description using GPT-4
- Shows before/after comparison

**Run it:**

```bash
export OPENAI_API_KEY=your-key-here
python simple_example.py
```

**Expected output:**
- Creates test files in a temp directory
- Runs GEPA optimization for ~30 metric calls
- Shows optimized tool description
- Original: "Read the contents of a file from the filesystem."
- Optimized: "Read and return the complete text contents of a file. Use when the user asks about file contents, configuration settings, or data stored in files. Provide the file path as an argument."

**Expected cost:**  30 iterations (depends)

## Understanding the Output

The optimization process will show:

1. **Initial Setup**: Test files created, MCP server configuration
2. **Optimization Progress**: GEPA's reflective mutation iterations
3. **Final Results**: Best candidate and performance metrics
4. **Comparison**: Original vs. optimized tool description



**Why different models for task vs reflection?**
- **Task model**: Runs many times, needs to be fast
- **Reflection model**: Runs fewer times, benefits from better reasoning


## Customization

### Use Different MCP Servers

Replace the server configuration:

```python
# Custom server
server_params = StdioServerParameters(
    command="python",
    args=["my_server.py"],
)
```

### Add More Examples

Extend the dataset:

```python
dataset.append({
    "user_query": "Your new query",
    "tool_arguments": {"param": "value"},
    "reference_answer": "expected answer",
    "additional_context": {},
})
```

### Change Metrics

Use different scoring functions:

```python
def llm_judge_metric(item, output):
    # Use LLM to score quality
    ...
    return score  # 0.0 to 1.0
```

### Optimize Multiple Components

```python
seed_candidate = {
    "tool_description": "Read file contents",
    "system_prompt": "You are a helpful file assistant with read access.",
}
```

## Next Steps

After running the example:

1. **Inspect trajectories**: Add `capture_traces=True` to see detailed execution
2. **Try different tools**: Use other MCP servers (web search, database, etc.)
3. **Scale up**: Use larger datasets and more optimization iterations
4. **Production use**: Deploy optimized tool descriptions in your application

## Troubleshooting



### npx not found

Install Node.js and npm:
- macOS: `brew install node`
- Ubuntu: `sudo apt install nodejs npm`

### MCP server connection errors

Check that the MCP server runs standalone:

```bash
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### Optimization gets stuck

Try:
- Reduce `max_metric_calls` for faster iterations
- Check API rate limits (for OpenAI example)
- Verify dataset quality (clear reference answers)
- For Ollama: Check model is responding (`ollama run model-name "test"`)

## Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [GEPA Documentation](https://github.com/gepa-ai/gepa)
- [Available MCP Servers](https://github.com/modelcontextprotocol/servers)
