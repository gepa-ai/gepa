# LangChainAdapter

GEPA adapter for LangChain v1. Optimize the prompts (and other text components) of any LangChain pipeline — a single chat model, an agent built with `langchain.agents.create_agent`, a custom LangGraph graph, RAG, etc. — using the GEPA optimizer.

## Install

```bash
pip install "gepa[langchain]"
```

The `langchain` extra installs only `langchain` and `langchain-core`; install a provider package separately for the model you want to use (e.g. `langchain-openai`, `langchain-anthropic`).

## Quickstart

Two things are required:

- **`rollout_fn(candidate, example) -> state`** — runs the candidate prompt against the example and returns a state dict (e.g. `{"messages": [...]}` for chat, or the full agent state from `agent.invoke(...)`).
- **`eval_fn(example, state) -> (score, feedback)`** — scores the rollout. The feedback string is what GEPA's reflection LM reads when proposing improvements.

```python
from gepa import optimize
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from gepa.adapters.langchain_adapter import (
    LangChainAdapter,
    last_message_text,
    make_reflection_lm,
)

task_llm = init_chat_model("openai:gpt-4o-mini")
reflection_llm = init_chat_model("openai:gpt-5-mini")

def rollout(candidate, example):
    messages = [
        SystemMessage(content=candidate["system_prompt"]),
        HumanMessage(content=example["input"]),
    ]
    result = task_llm.invoke(messages)
    if not isinstance(result, AIMessage):
        result = AIMessage(content=str(result.content))
    return {"messages": messages + [result]}

def evaluate(example, state):
    response = last_message_text(state)
    if example["answer"] in response:
        return 1.0, "Correct."
    return 0.0, f"Wrong. Expected {example['answer']}."

adapter = LangChainAdapter(rollout_fn=rollout, eval_fn=evaluate)

result = optimize(
    seed_candidate={"system_prompt": "Answer the question."},
    trainset=train_set,
    valset=val_set,
    adapter=adapter,
    reflection_lm=make_reflection_lm(reflection_llm),
    max_metric_calls=500,
)
print(result.best_candidate["system_prompt"])
```

## Tool-using agent (`create_agent`)

For a tool-using agent, build the agent inside `rollout_fn` and return the agent state directly. The state's `messages` list contains the full trace — system prompt, user message, tool calls, tool results, and final answer — which `last_message_text` and your `eval_fn` can inspect.

```python
from typing import Literal

from gepa import optimize
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from gepa.adapters.langchain_adapter import (
    LangChainAdapter,
    last_message_text,
    make_reflection_lm,
)

task_llm = init_chat_model("openai:gpt-5-nano", reasoning_effort="minimal")
reflection_llm = init_chat_model("openai:gpt-5", reasoning_effort="medium")

@tool
def calculator(a: int, b: int, op: Literal["+", "-", "*"]) -> str:
    """Compute `a op b` for op in {+, -, *}."""
    if op == "+": return str(a + b)
    if op == "-": return str(a - b)
    if op == "*": return str(a * b)
    return f"ERROR: unsupported op {op!r}"

def rollout(candidate, example):
    agent = create_agent(
        model=task_llm,
        tools=[calculator],
        system_prompt=candidate["system_prompt"],
    )
    return agent.invoke({"messages": [HumanMessage(content=example["input"])]})

def evaluate(example, state):
    response = last_message_text(state)
    if str(example["answer"]) in response:
        return 1.0, "Correct."
    return 0.0, f"Wrong. Expected {example['answer']}. Got: {response}"

adapter = LangChainAdapter(rollout_fn=rollout, eval_fn=evaluate)

result = optimize(
    seed_candidate={"system_prompt": "Use the calculator tool to compute the expression."},
    trainset=train_set,
    valset=val_set,
    adapter=adapter,
    reflection_lm=make_reflection_lm(reflection_llm),
    max_metric_calls=200,
)
```

## Examples

See [`examples/langchain_adapter/`](../../../../examples/langchain_adapter/) for runnable scripts:

- `pair_sum_product.py` — single-turn prompt optimization on a synthetic task
- `gsm8k.py` — GSM8K word problems
- `aime.py` — AIME math competition problems
- `big_number_arithmetic.py` — tool-using agent built with `create_agent`
