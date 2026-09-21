# GEPA with vLLM + Hugging Face (local, open-weight models)

Run GEPA end-to-end against a **local** Hugging Face model served by
[vLLM](https://docs.vllm.ai) — no paid API keys. GEPA talks to the server over
HTTP via LiteLLM, so core GEPA stays dependency-free; `vllm` only needs to be
installed where the model is served.

See the full guide: [`docs/docs/guides/local-models-vllm.md`](../../docs/docs/guides/local-models-vllm.md).

## 1. Start a vLLM server

In one terminal (GPU recommended). Install vLLM directly on the serving box — it
is heavy/GPU-specific and is deliberately **not** bundled into GEPA:

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b-instruct \
    --host 0.0.0.0 \
    --port 8000
```

## 2. Run the example

In another terminal:

```bash
uv run python examples/vllm_huggingface/main.py \
    --served-model-name qwen2.5-7b-instruct \
    --api-base http://localhost:8000/v1
```

Flags (all optional, sensible defaults shown):

- `--served-model-name` — must match vLLM's `--served-model-name` (default `qwen2.5-7b-instruct`)
- `--api-base` — the server's OpenAI-compatible base URL, **including `/v1`** (default `http://localhost:8000/v1`)
- `--reflection-model` — override the reflection model (default: same local model). Pass a hosted string like `openai/gpt-4o` for a mixed setup.
- `--max-metric-calls` — optimization budget (default `20`)

The script optimizes a tiny system prompt on a toy QA task and prints the best
prompt and score. Swap in your own `trainset` / seed prompt to adapt it.
