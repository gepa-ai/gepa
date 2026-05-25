## Install

GEPA's `langchain` extra contains only `langchain` + `langchain-core`. To run the examples with a specific langchain provider:

1. The GEPA langchain extra:

   ```bash
   uv sync --extra dev --extra langchain
   ```

2. A LangChain provider package for the model you want to use, e.g.:

   ```bash
   uv pip install langchain-openai      # for openai:* models
   # uv pip install langchain-anthropic # for anthropic:* models
   # uv pip install langchain-google-genai
   # uv pip install langchain-ollama
   ```

   `init_chat_model("openai:gpt-4o-mini")` will raise an ImportError naming the
   missing package if you skip this step.

3. Export your API key in the shell. The default models are `openai:gpt-4o-mini`
   (task) and `openai:gpt-5-mini` (reflection), so:

   ```bash
   export OPENAI_API_KEY=sk-...
   # export ANTHROPIC_API_KEY=sk-ant-...
   ```

   Override with `--task-model` / `--reflection-model` to use a different
   provider.

## Langchain Adapter Examples

All examples build the task and reflection LLMs via LangChain's standard
`init_chat_model(provider:model, **kwargs)`, **except** `big_number_arithmetic.py`,
which wraps the task model in a tool-using agent via LangChain's `create_agent`.
The reflection model in every example still uses `init_chat_model`.

| Script | Task model builder | Description |
|---|---|---|
| [`pair_sum_product.py`](pair_sum_product.py) | `init_chat_model` | Single-turn LLM prompt optimization on a synthetic problem |
| [`big_number_arithmetic.py`](big_number_arithmetic.py) | `create_agent` (tool-using agent) | Multi-digit arithmetic with a calculator tool |
| [`gsm8k.py`](gsm8k.py) | `init_chat_model` | GSM8K grade-school word problems from HuggingFace `datasets` |
| [`aime.py`](aime.py) | `init_chat_model` | AIME math competition problems (train on older AIME, test on AIME 2025) |

### Running with the default OpenAI provider

Set `OPENAI_API_KEY` and run any example directly:

```bash
uv run python examples/langchain_adapter/pair_sum_product.py
uv run python examples/langchain_adapter/big_number_arithmetic.py
uv run python examples/langchain_adapter/gsm8k.py
uv run python examples/langchain_adapter/aime.py
```

### Running through OpenRouter

Install the OpenRouter provider and set `OPENROUTER_API_KEY`:

```bash
uv pip install langchain-openrouter
export OPENROUTER_API_KEY=...
```

Use the `openrouter:<model-id>` prefix with `init_chat_model`. Reasoning effort
is a top-level `reasoning` kwarg:

```bash
uv run python examples/langchain_adapter/pair_sum_product.py \
  --task-model openrouter:openai/gpt-4o-mini \
  --reflection-model openrouter:openai/gpt-5-mini \
  --reflection-model-kwargs '{"reasoning":{"effort":"medium"}}'
```

The same flags work for `big_number_arithmetic.py` and `gsm8k.py` — just swap
the script path.

## Example Optimized Prompts

### Pair Sum Product
<details>
   <summary>
   Starting Prompt
   </summary>

   ```text
   Add adjacent pairs of numbers, then multiply the results.
   If only one pair numbers, just add the numbers

   Pairs Example: 1,2
   - Add pairs: 1+2=3
   - multiply: Assume 1, 3*1=3
   - Answer: 3

   Example: 3, 5, 2, 4
   - Add pairs: 3+5=8, 2+4=6
   - Multiply: 8*6=48
   - Answer: 48

   Now solve the problem below. Put your final answer in <answer> tags. Be very concise
   ```
</details>
<details>
   <summary>
   Optimized Prompt
   </summary>

   ```text
   You are given inputs in the form "Numbers: a, b, c, ..." (commas and spacing may vary). Your job is to compute a single integer result and output it exactly on one line enclosed in <answer>...</answer> tags and nothing else.

   Precise task and rules:
   - Parse the input in the given order and extract all integers (allow positive, negative, and zero; integers may have optional + or - sign). Ignore any non-numeric text. You may assume at least one integer is present.
   - Form adjacent, non-overlapping pairs from the list in order: (n1,n2), (n3,n4), (n5,n6), ... .
   - For each pair compute pair_sum = ni + n(i+1).
   - If there are multiple pairs, compute the final result as the product of all pair_sums (multiply every pair_sum together).
   - If there is exactly one pair (exactly two numbers), the final result is that pair's sum.
   - If the list has an odd number of integers, treat the final unpaired last integer as a standalone multiplicative factor (multiply the product of pair_sums by that last integer).
   - If the input has exactly one integer, return that integer (it is the final product).
   - Use exact integer arithmetic (arbitrary-precision if necessary). Do not perform any floating-point rounding.
   - Do NOT include thousands separators, commas, spaces, or any extra text in the numeric result.
   - Output must be a single line containing nothing but the final integer enclosed in <answer> and </answer> tags. Example: <answer>12345</answer>

   Common pitfalls to avoid (learned from examples):
   - Always pair sequentially and non-overlapping until you run out of numbers; do not leave two numbers unpaired at the end when they should form a final pair.
   - Do not treat multiple trailing numbers incorrectly—only one trailing number can exist (if the count is odd).
   - Multiply the pair sums together exactly; do not drop, combine, or reorder pair-sums incorrectly.
   - Support negative and zero values correctly (they affect sums and product signs).

   Examples (for your reference only; do not output these in responses):
   - Input: "Numbers: 3,5,2,4" -> pair_sums = [8,6] -> result = 8*6 = 48 -> output: <answer>48</answer>
   - Input: "Numbers: 4,5,6" -> pair_sums = [9], leftover 6 -> result = 9*6 = 54 -> output: <answer>54</answer>
   - Input: "Numbers: 7" -> result = 7 -> output: <answer>7</answer>
   ```

</details>

### GSM8K

<details>
   <summary>
   Starting Prompt
   </summary>

   ```text
   Solve the grade-school math word problem.
   Think step by step, then provide your final answer as a single integer on the last line.
   ```

</details>

<details>
   <summary>
   Optimized Prompt
   </summary>

   ```text
      Solve the grade-school math word problem.
      Think step by step, then provide your final answer as a single integer on the last line.
   ```

</details>

