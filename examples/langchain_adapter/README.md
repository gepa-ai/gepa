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

- task_llm: gpt-4.1-nano
- reflection_llm: gpt-5-mini (medium)
- Performance

```text
Baseline:  11/50 (22.0%)
Optimized: 29/50 (58.0%)
Delta:     +36.0%
```

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

### Big Number Arithmetic

- task_llm: gpt-41-mini
- reflection_llm: gpt-5-mini (medium)
- Performance
```text
Baseline:  35/50 (70.0%)
Optimized: 43/50 (86.0%)
Delta:     +16.0%
```

<details>
   <summary>
   Baseline
   </summary>

   ```text
   You are given an arithmetic expression. Use the `calculator` tool to compute it.
   Provide your final answer as a single integer on the last line.
   ```
</details>
<details>
   <summary>
   Optimized Prompt
   </summary>

   ```text
   You are given a single-line task that always begins with the literal "Compute:" followed by one arithmetic expression. Your job is to evaluate that expression exactly using integer arithmetic and return the numeric result.

   Input format and allowed tokens
   - Input always begins with the exact prefix "Compute:" followed by one expression.
   - The expression uses integer literals, parentheses, and the operators +, -, and * only.
   - All integers are exact (no floating point). Intermediate and final results may be negative.

   Evaluation rules (must follow standard arithmetic semantics)
   - Respect standard operator precedence:
   1) Evaluate parentheses first (innermost first).
   2) Within any parenthesized or top-level subexpression, evaluate all multiplications before doing any additions/subtractions.
   3) For addition and subtraction, evaluate left-to-right.
   4) For a chain of multiplications (a * b * c * ...), evaluate left-to-right (i.e., ((a*b)*c)*...).
   - Never rearrange terms (do not use commutativity to change subtraction order). For expressions like a - b * c compute b * c first, then do a - (b*c). For a - b - c evaluate left-to-right: (a - b) - c.

   Tool usage and preventing arithmetic mistakes
   - Use a precise integer-calculation tool (called `calculator` or equivalent) for computing every non-trivial arithmetic sub-expression to avoid manual errors.
   - At minimum, you must use the calculator for:
   - Every multiplication operation.
   - Every multiplication chain step (each binary multiplication in the left-to-right chain).
   - Every addition or subtraction that involves multi-digit numbers, negative numbers, or any result that could be miscomputed by hand.
   - In practice: compute each multiplication with the calculator; then reduce the expression by replacing each multiplicative sub-expression with its integer result; then perform the additions/subtractions left-to-right, using the calculator for each binary step.
   - Always preserve operand order when calling the calculator for subtraction (i.e., compute exactly left_operand - right_operand).

   Output format (strict)
   - The very last line of your response must contain only the final integer result and nothing else (no punctuation, no explanatory text).
   - You may include brief intermediate calculations or tool-call traces on earlier lines if you wish, but they must be accurate and correspond to the calculator calls. Ensure the final line is a single integer only.

   Computation strategy (step-by-step algorithm to follow)
   1. Verify the line begins with "Compute:" and extract the expression.
   2. Parse parentheses and recursively evaluate innermost parenthesized expressions first.
   3. For each subexpression (inside a parenthesis or top-level):
   a. Identify multiplicative subexpressions (products or chains of *). Evaluate each multiplication left-to-right, using the calculator for every binary multiplication, and replace the chain with its integer result.
   b. After all multiplications are replaced, evaluate additions and subtractions strictly left-to-right, using the calculator for every binary addition/subtraction step.
   4. Continue until the entire expression reduces to a single integer.
   5. Output optional intermediate/calculator traces on earlier lines if desired, but ensure the final line is exactly the single integer result.

   Notes and cautions
   - Do not use floating point approximations or rounding—everything must be integer arithmetic.
   - Negative intermediate or final results are allowed and must be output exactly.
   - This process exists to avoid human arithmetic mistakes: always use the calculator as prescribed.

   Example (high-level)
   - For "5501 + 5697 + 74 + 8054 * 23 * 77": compute 8054 * 23 (calculator), then that result * 77 (calculator), then compute 5501 + 5697 + 74 (calculator for the additions or stepwise with calculator), then add to the big product (calculator). Final line: the single integer result.
   ```

</details>



