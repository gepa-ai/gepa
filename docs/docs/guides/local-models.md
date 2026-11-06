# Local & Self-Hosted Models

GEPA can run entirely on **open-weight models that you serve yourself**, with no paid API keys. Every major inference engine (vLLM, SGLang, Ollama, llama.cpp, LM Studio, …) exposes an OpenAI-compatible `/v1` endpoint, and GEPA's `LM` reaches all of them the same way:

```python
from gepa.lm import LM

lm = LM(
    "openai/<model-name>",               # "openai/" = speak the OpenAI-compatible protocol
    api_base="http://localhost:8000/v1",  # your server, including the /v1 suffix
    api_key="EMPTY",                     # most local servers ignore it, but the client requires one
)
```

GEPA talks to the server over HTTP through LiteLLM, so nothing engine-specific is installed in the GEPA process. The engine only needs to be installed on the machine that serves the model.

## 1. Start a server

Pick an engine. Each tab gives a launch command and the matching `LM(...)` line. The examples serve [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) with a 32k context; substitute any model the engine supports.

=== "vLLM"

    GPU serving with high throughput ([docs](https://docs.vllm.ai)).

    ```bash
    pip install vllm
    vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768 --reasoning-parser qwen3
    ```

    ```python
    lm = LM("openai/Qwen/Qwen3-8B", api_base="http://localhost:8000/v1", api_key="EMPTY")
    ```

=== "SGLang"

    GPU serving with high throughput ([docs](https://docs.sglang.io)).

    ```bash
    uv pip install --prerelease=allow sglang
    python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 \
        --context-length 32768 --reasoning-parser qwen3
    ```

    ```python
    lm = LM("openai/Qwen/Qwen3-8B", api_base="http://localhost:30000/v1", api_key="EMPTY")
    ```

=== "Ollama"

    Laptop and desktop serving with one-command model downloads ([docs](https://docs.ollama.com)).

    ```bash
    OLLAMA_CONTEXT_LENGTH=32768 ollama serve
    ollama pull qwen3:8b    # in another terminal
    ```

    ```python
    lm = LM("openai/qwen3:8b", api_base="http://localhost:11434/v1", api_key="EMPTY")
    ```

=== "llama.cpp"

    CPU, Apple Silicon or GPU serving of GGUF models ([docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)).

    ```bash
    llama-server -hf Qwen/Qwen3-8B-GGUF --alias qwen3-8b --port 8080 -c 32768 \
        --jinja --reasoning-format deepseek
    ```

    ```python
    lm = LM("openai/qwen3-8b", api_base="http://localhost:8080/v1", api_key="EMPTY")
    ```

    Always pass `--port`: the default port has changed between `llama-server` releases.

Any other OpenAI-compatible server (LM Studio, TensorRT-LLM, LMDeploy, a hosted gateway, …) works the same way: point `api_base` at it and use `openai/<model-name>`.

To check that the server is up and see the model name it expects:

```bash
curl http://localhost:8000/v1/models
```

## 2. Optimize with local models

An `LM` is a plain callable, so it can be the task model, the reflection model, or both:

```python
import gepa
from gepa.lm import LM

local_lm = LM("openai/Qwen/Qwen3-8B", api_base="http://localhost:8000/v1", api_key="EMPTY")

trainset = [
    {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
    {"input": "Capital of France?", "additional_context": {}, "answer": "Paris"},
]

result = gepa.optimize(
    seed_candidate={"system_prompt": "Answer concisely."},
    trainset=trainset,
    task_lm=local_lm,
    reflection_lm=local_lm,
    max_metric_calls=20,
)
print("Best prompt:", result.best_candidate["system_prompt"])
```

**Mixed setups** work too. A common one pairs a small local task model with a stronger hosted reflection model:

```python
result = gepa.optimize(
    seed_candidate={"system_prompt": "Answer concisely."},
    trainset=trainset,
    task_lm=local_lm,                    # served locally
    reflection_lm="openai/gpt-6.1-sol",  # hosted (needs OPENAI_API_KEY)
    max_metric_calls=20,
)
```

With `optimize_anything`, your evaluator calls the local model itself, and the same `LM` can serve as the reflection model:

```python
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything


def evaluate(candidate, example):
    output = local_lm(
        [
            {"role": "system", "content": candidate["system_prompt"]},
            {"role": "user", "content": example["input"]},
        ]
    )
    score = 1.0 if example["answer"].lower() in output.lower() else 0.0
    return score, {"Output": output, "Expected": example["answer"]}  # feedback for reflection


result = optimize_anything(
    seed_candidate={"system_prompt": "Answer concisely."},
    evaluator=evaluate,
    dataset=trainset,
    objective="Write a system prompt that makes the assistant answer each question correctly.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=20),
        reflection=ReflectionConfig(reflection_lm=local_lm),
    ),
)
```

A runnable end-to-end example covering both entry points is in [`examples/local_models/`](https://github.com/gepa-ai/gepa/tree/main/examples/local_models).

## 3. Settings that matter for GEPA

**Context length.** A reflection prompt holds the current candidate plus several examples with their outputs and feedback, so it is much longer than a typical chat turn. Serve with a context window of at least 16k–32k tokens (`--max-model-len`, `--context-length`, `OLLAMA_CONTEXT_LENGTH` or `-c`). Ollama's default depends on available memory and can be as small as 4k. When a prompt doesn't fit, Ollama truncates it instead of returning an error.

**Reasoning models.** For thinking models (Qwen3, DeepSeek-R1, gpt-oss, …), turn on the engine's reasoning parser: `--reasoning-parser` in vLLM and SGLang, `--reasoning-format` in llama.cpp. Ollama separates thinking automatically. The parser moves the chain of thought out of the response text. Without it:

- task outputs include the model's thinking, and your metric scores it too (a substring check can match a word in the thinking rather than in the answer);
- a reflection response that runs out of tokens mid-thought is rejected as incomplete, and that iteration is wasted.

Set `max_tokens` on the reflection `LM` high enough to cover the thinking plus the new candidate.

**Cost tracking.** Local servers don't report prices, so `LM.total_cost` stays at `0.0` and `max_reflection_cost` won't stop a local run. Token counts (`total_tokens_in` / `total_tokens_out`) are still taken from the server's usage response. To bound a run, use `max_metric_calls` or another stop condition such as [`TimeoutStopCondition`](../api/stop_conditions/TimeoutStopCondition.md). See [Cost & Token Tracking](cost-tracking.md).

**Throughput.** vLLM and SGLang batch concurrent requests, so evaluation parallelism directly speeds up a run. With `gepa.optimize`, the default adapter sends up to 10 task requests at a time; to change that, pass `adapter=DefaultAdapter(model=local_lm, max_litellm_workers=...)`. With `optimize_anything`, set `EngineConfig(max_workers=...)` (default: the CPU count). By default, Ollama and llama.cpp process only a few requests in parallel (`OLLAMA_NUM_PARALLEL`, `llama-server --parallel`). Raising client-side concurrency beyond that only queues the extra requests.

!!! note "Engine-specific LiteLLM routes"
    LiteLLM also has engine-specific prefixes such as `hosted_vllm/` and `ollama_chat/`. They work with `LM` too, but GEPA doesn't need them: `openai/` plus `api_base` is the one pattern that works for every engine.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `NotFoundError` / model does not exist | The name after `openai/` must match what the server lists at `/v1/models`: the Hugging Face repo id, or the `--served-model-name` / `--alias` you set. |
| Connection refused / 404 | `api_base` needs the `/v1` suffix (`http://localhost:8000/v1`, not `http://localhost:8000`). Check that the host and port match the server's. |
| Outputs or reflections start with `<think>` | Turn on the engine's reasoning parser (see above). |
| Reflections ignore most examples, or quality drops sharply | The context window is too small and prompts are being truncated. Raise the server's context length. |
| CUDA out of memory | Use a smaller or quantized model (AWQ, GPTQ, FP8, or a smaller GGUF quant), or lower the context length. |
| 401 / gated model when the server starts | Run `hf auth login` on the serving machine to access gated Hugging Face repos. |
| Timeouts under load | Lower the evaluation concurrency, or pass `timeout=...` to `LM(...)`. |
