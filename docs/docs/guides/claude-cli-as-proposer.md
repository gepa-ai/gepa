# Using Claude Code as a Proposer

GEPA's reflection LM — the model that reads evaluation feedback and proposes improved candidates — accepts any callable matching `(str) -> str`. This means you can use [Claude Code](https://docs.anthropic.com/en/docs/claude-code)'s `claude -p` command as a drop-in proposer, powered by your existing Claude subscription with no API key required.

The same approach works for any CLI-based model (e.g., `ollama run`, a local model server, or any tool that accepts a prompt on stdin and returns text on stdout).

---

## Setup

Make sure `claude` is installed and authenticated:

```bash
claude  # opens interactive mode; complete login if needed
claude -p "say hello"  # verify non-interactive mode works
```

## Wrapping `claude -p` as a Reflection LM

The `LanguageModel` protocol is `(str) -> str`. Wrap the CLI call in a function:

```python
import subprocess


def claude_cli_lm(prompt: str) -> str:
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr}")
    return result.stdout
```

## Using with `gepa.optimize`

Pass the wrapper as `reflection_lm`:

```python
import gepa

result = gepa.optimize(
    seed_candidate={"instructions": "Answer the question correctly."},
    trainset=dataset,
    task_lm=your_task_lm,
    reflection_lm=claude_cli_lm,
    max_metric_calls=50,
)
```

## Using with `optimize_anything`

Set it on `ReflectionConfig`:

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, ReflectionConfig

config = GEPAConfig(
    reflection=ReflectionConfig(
        reflection_lm=claude_cli_lm,
    ),
)

result = optimize_anything(config=config, ...)
```

## Using `claude -p` as Both Task LM and Reflection LM

You can also use `claude -p` to power the system being optimized. The `task_lm` in `gepa.optimize` expects a chat-completion callable `(list[dict]) -> str`, so add a thin wrapper that flattens messages:

```python
def claude_cli_chat(messages: list) -> str:
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"[System]\n{m['content']}")
        elif m["role"] == "user":
            parts.append(f"[User]\n{m['content']}")
    prompt = "\n\n".join(parts)
    return claude_cli_lm(prompt)


result = gepa.optimize(
    seed_candidate={"instructions": "Answer the question correctly."},
    trainset=dataset,
    task_lm=claude_cli_chat,
    reflection_lm=claude_cli_lm,
    max_metric_calls=50,
)
```

## Full Example

A minimal end-to-end script that optimizes a Q&A prompt using only `claude -p`:

```python
import subprocess
import gepa


def claude_cli_lm(prompt: str) -> str:
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr}")
    return result.stdout


def claude_cli_chat(messages: list) -> str:
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"[System]\n{m['content']}")
        elif m["role"] == "user":
            parts.append(f"[User]\n{m['content']}")
    return claude_cli_lm("\n\n".join(parts))


dataset = [
    {"input": "What is 2+2?", "answer": "4", "additional_context": {}},
    {"input": "What is the capital of France?", "answer": "Paris", "additional_context": {}},
    {"input": "What color is the sky on a clear day?", "answer": "blue", "additional_context": {}},
]

result = gepa.optimize(
    seed_candidate={"instructions": "Answer the question correctly."},
    trainset=dataset,
    task_lm=claude_cli_chat,
    reflection_lm=claude_cli_lm,
    max_metric_calls=20,
    reflection_minibatch_size=1,
    display_progress_bar=True,
)

print(f"Best instructions: {result.best_candidate}")
print(f"Best score: {result.val_aggregate_scores[result.best_idx]:.3f}")
```

!!! tip "Parallelism"
    Each `claude -p` call is a separate subprocess, so GEPA's parallel evaluation works out of the box if you use `claude_cli_chat` as your task LM. The reflection LM calls are sequential by design (one proposal per iteration).

!!! note "Rate limits and latency"
    `claude -p` is subject to your subscription's rate limits. For large-scale optimization runs, using the API via LiteLLM (the default path with `reflection_lm="anthropic/claude-sonnet-4-20250514"`) will be faster and more reliable. The CLI approach is best for experimentation and moderate-scale runs where you want to avoid API key setup.
