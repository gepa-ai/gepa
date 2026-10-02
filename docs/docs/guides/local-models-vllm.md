# Local Models with vLLM & Hugging Face

You can run GEPA entirely against **local, open-weight models** — no paid API keys — by serving a Hugging Face model through [vLLM](https://docs.vllm.ai)'s OpenAI-compatible endpoint and pointing GEPA's `LM` at it.

GEPA reaches the server over plain HTTP (via LiteLLM), so **core GEPA stays dependency-free**: `vllm`/`transformers` only need to be installed on the machine *serving* the model, never in the GEPA process. The local path is fully optional and does not change any default behaviour — hosted-provider workflows keep working unchanged.

## 1. Serve a model with vLLM

Install vLLM in the serving environment (GPU recommended). vLLM is heavy and
GPU/platform-specific, so it is **not** bundled into GEPA — install it directly
on the machine that will serve the model:

```bash
pip install vllm
```

In your **GEPA / client** environment you only need GEPA's LiteLLM client (GEPA
reaches the server over HTTP and never imports `vllm`):

```bash
pip install "gepa[vllm]"    # == gepa[full]; the LiteLLM client used to call the endpoint
```

Start an OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b-instruct \
    --host 0.0.0.0 \
    --port 8000
```

`--served-model-name` is the name GEPA will refer to; `--model` is the Hugging Face repo to load.

## 2. Point GEPA at the endpoint

Use the `make_vllm_lm` helper:

```python
from gepa.lm import make_vllm_lm

lm = make_vllm_lm(
    "qwen2.5-7b-instruct",              # the --served-model-name
    api_base="http://localhost:8000/v1",  # include the /v1 suffix
    api_key="EMPTY",                     # ignored by vLLM, required by the client
)

print(lm("Say hello in one word."))
```

`make_vllm_lm` is a thin convenience wrapper. The equivalent explicit form is:

```python
from gepa.lm import LM

lm = LM("openai/qwen2.5-7b-instruct", api_base="http://localhost:8000/v1", api_key="EMPTY")
```

The `openai/` provider prefix routes LiteLLM to the OpenAI-compatible endpoint; `make_vllm_lm` adds it for you when it's missing.

## 3. Optimize with local models

The returned `LM` is a plain callable, so it works as **both** the task model and the reflection model — and you can mix local and hosted models freely:

```python
import gepa
from gepa.lm import make_vllm_lm

local_lm = make_vllm_lm("qwen2.5-7b-instruct")

trainset = [
    {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
    {"input": "Capital of France?", "additional_context": {}, "answer": "Paris"},
]

result = gepa.optimize(
    seed_candidate={"system_prompt": "Answer concisely."},
    trainset=trainset,
    task_lm=local_lm,          # vLLM-served task model
    reflection_lm=local_lm,    # vLLM-served reflection model
    max_metric_calls=20,
)
print("Best prompt:", result.best_candidate["system_prompt"])
```

**Mixed setups** work too — e.g. a local task model with a stronger hosted reflection model:

```python
result = gepa.optimize(
    seed_candidate={"system_prompt": "Answer concisely."},
    trainset=trainset,
    task_lm=make_vllm_lm("qwen2.5-7b-instruct"),   # local
    reflection_lm="openai/gpt-4o",                  # hosted (needs OPENAI_API_KEY)
    max_metric_calls=20,
)
```

A runnable end-to-end example lives in [`examples/vllm_huggingface/`](https://github.com/gepa-ai/gepa/tree/main/examples/vllm_huggingface).

## Cost & token tracking

vLLM does not report USD pricing, so `LM.total_cost` will be `0.0` for local calls. Token counts (`total_tokens_in` / `total_tokens_out`) are still populated from the server's usage response. See [Cost & Token Tracking](cost-tracking.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `litellm.NotFoundError` / model not found | `LM(model=...)` must match `--served-model-name` exactly. With `make_vllm_lm`, pass the served name (the `openai/` prefix is added for you). |
| Connection refused / 404 | Check the `/v1` suffix on `api_base` (`http://localhost:8000/v1`, not `:8000`), and that the server is up on that host/port. |
| CUDA out of memory | Use a smaller model, lower `--gpu-memory-utilization`, or serve a quantized checkpoint (e.g. AWQ/GPTQ). |
| 401 / gated model on server start | Authenticate to Hugging Face on the serving box (`huggingface-cli login`) for gated repos. |
| Chat template / tokenizer errors | Pass `--chat-template` to vLLM for base models lacking a built-in template. |
| Timeouts / overload under load | Lower adapter concurrency (`max_litellm_workers` / `max_workers`) or pass `timeout=...` through to `make_vllm_lm`. |

!!! note
    Any OpenAI-compatible server (Ollama's `/v1`, LM Studio, TGI, SGLang, …) works the same way — point `api_base` at it and use `make_vllm_lm` / `LM("openai/<name>", ...)`.
