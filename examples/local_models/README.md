# GEPA with local and self-hosted models

Run GEPA end to end against an open-weight model that you serve yourself, with no paid API keys. GEPA talks to any OpenAI-compatible server over HTTP, so the inference engine only needs to be installed where the model is served.

See the full guide, including the settings that matter for GEPA (context length, reasoning parsers, cost tracking): [`docs/docs/guides/local-models.md`](../../docs/docs/guides/local-models.md).

## 1. Start a server

Pick one engine and run it in its own terminal:

| Engine | Launch command | `--model` | `--api-base` |
|---|---|---|---|
| vLLM | `vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768 --reasoning-parser qwen3` | `Qwen/Qwen3-8B` | `http://localhost:8000/v1` |
| SGLang | `python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --context-length 32768 --reasoning-parser qwen3` | `Qwen/Qwen3-8B` | `http://localhost:30000/v1` |
| Ollama | `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`, then `ollama pull qwen3:8b` | `qwen3:8b` | `http://localhost:11434/v1` |
| llama.cpp | `llama-server -hf Qwen/Qwen3-8B-GGUF --alias qwen3-8b --port 8080 -c 32768 --jinja --reasoning-format deepseek` | `qwen3-8b` | `http://localhost:8080/v1` |

Any other OpenAI-compatible server works too. Use the model name that `curl <api-base>/models` lists.

## 2. Run the example

In another terminal, for example against vLLM:

```bash
uv run python examples/local_models/main.py \
    --model Qwen/Qwen3-8B \
    --api-base http://localhost:8000/v1
```

Flags:

- `--model` (required): the model name as the server lists it at `/v1/models`
- `--api-base` (required): the server's OpenAI-compatible base URL, **including `/v1`**
- `--api-key`: defaults to `EMPTY`, which local servers accept
- `--reflection-model`: a hosted reflection model for a mixed setup, e.g. `openai/gpt-6.1-sol` (default: the same local model)
- `--max-metric-calls`: the optimization budget (default `20`)

The script optimizes a short system prompt on a toy QA task and prints the best prompt and its score. To adapt it, swap in your own `TRAINSET` and seed prompt.
