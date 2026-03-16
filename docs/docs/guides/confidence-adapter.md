# Confidence-Aware Classification Optimization

!!! info "Install the extra"
    ```bash
    pip install "gepa[confidence]"
    ```

The **ConfidenceAdapter** is a purpose-built adapter for **classification tasks** where the LLM returns a structured JSON output with `enum`-constrained fields.  It uses token-level log-probabilities to detect when the model "guesses correctly" and feeds that signal into both scoring and reflective feedback.

## The Problem: Lucky Guesses

Standard prompt optimization evaluates a candidate prompt with binary scoring: correct → 1.0, wrong → 0.0.  This cannot distinguish between a model that genuinely understands a category and one that merely guesses correctly.

Consider a transaction categorization prompt.  The model answers `"Bills/Electricity"` and that happens to be correct.  But the token-level logprobs reveal the runner-up was `"Bills/Gas & Oil"` at almost the same probability.  The model got lucky.  Under a purely binary metric, GEPA keeps this prompt -- it "works".  The next random seed or slight input variation will cause a misclassification.

## How It Works

The ConfidenceAdapter solves this by:

1. **Sending requests with `logprobs=True`** and a JSON schema `response_format` that constrains the output to an `enum` of allowed categories.

2. **Extracting the joint logprob** (sum of per-token logprobs) for the classification field via [`llm-structured-confidence`](https://github.com/rodolfonobrega/llm-structured-confidence).  This is the most natural confidence measure: `exp(joint_logprob)` gives the probability the model assigns to the entire value.

3. **Blending correctness with confidence** via a pluggable scoring strategy, so that lucky guesses receive lower scores.

4. **Feeding confidence details into the reflective feedback** so the reflection LLM knows *why* the model was uncertain and can propose prompts that resolve specific ambiguities.

### What the Reflection LLM Sees

**With DefaultAdapter** (no confidence):

> *"The generated response is incorrect. The correct answer is 'Bills/Electricity'. Ensure that the correct answer is included in the response exactly as it is."*

**With ConfidenceAdapter**:

> *"Correct but unreliable (-2.31 logprob, 10% probability). Model answered 'Bills/Electricity' but was nearly split with alternatives. Top alternatives: 'Bills/Gas & Oil' (9%). The model cannot reliably distinguish between these options with the current prompt."*

Or:

> *"Incorrect with high confidence (-0.05 logprob, 95% probability). Model confidently answered 'Shopping/Electronics' but the correct answer is 'Shopping/Video Games'. The prompt is misleading the model for this type of input."*

## Prerequisites

- **Structured output with enum constraints**: the adapter works best when the LLM's `response_format` forces a JSON object with one or more `enum`-constrained string fields.  This lets the logprobs reflect the model's distribution over the allowed categories.
- **Logprobs support**: the model must expose token-level logprobs.  OpenAI (`gpt-4.1`, `gpt-4.1-mini`, etc.) and Google Gemini (`gemini-2.5-flash`, etc.) both support this.

## Quick Start

```python
import gepa
from gepa.adapters.confidence_adapter import ConfidenceAdapter, LinearBlendScoring

adapter = ConfidenceAdapter(
    model="openai/gpt-4.1-mini",
    field_path="category_name",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "category_name": {
                        "type": "string",
                        "enum": [
                            "Bills/Electricity",
                            "Bills/Gas & Oil",
                            "Food & Drinks/Restaurants",
                            "Shopping/Electronics",
                            "Shopping/Video Games",
                        ],
                    }
                },
                "required": ["category_name"],
                "additionalProperties": False,
            },
        },
    },
    scoring_strategy=LinearBlendScoring(low_confidence_threshold=0.5),
)

result = gepa.optimize(
    seed_candidate={"system_prompt": "Classify the following transaction."},
    trainset=[
        {"input": "UBER EATS payment", "answer": "Food & Drinks/Restaurants", "additional_context": {}},
        {"input": "LIGHT electricity bill", "answer": "Bills/Electricity", "additional_context": {}},
        {"input": "Steam purchase", "answer": "Shopping/Video Games", "additional_context": {}},
    ],
    adapter=adapter,
    reflection_lm="openai/gpt-4.1",
    max_metric_calls=500,
)
```

## Understanding Joint Logprob

The `joint_logprob` is the **sum of all per-token logprobs** for the value tokens of the target field.

For example, if the model outputs `"Bills/Electricity"` and the tokens are `["Bills", "/", "Elec", "tricity"]` with logprobs `[-0.02, -0.01, -0.10, -0.01]`, the joint logprob is `-0.14`.

| Joint Logprob | Probability | Interpretation |
|:---:|:---:|---|
| `-0.05` | ~95% | Very confident |
| `-0.22` | ~80% | Confident |
| `-0.69` | ~50% | Coin flip |
| `-2.30` | ~10% | Very uncertain |
| `-4.60` | ~1% | Near-random guess |

The closer to 0, the more certain the model is about its answer.

## From Logprob to Score: How Scoring Works

It is important to understand the distinction between the **joint logprob** (the raw confidence metric extracted from the LLM) and the **score** (the `[0, 1]` value that GEPA uses for optimization).  They are **not** the same thing.

1. The adapter extracts the **joint logprob** from the LLM response (always ≤ 0).
2. The scoring strategy converts it to a **probability** via `exp(joint_logprob)` (always in `[0, 1]`).
3. The strategy then combines that probability with **correctness** to produce the final **score** in `[0, 1]`.

```
joint_logprob  ──►  probability = exp(logprob)  ──►  scoring strategy  ──►  GEPA score [0, 1]
    -0.14                    0.87                      (depends on            e.g. 1.0
                                                        strategy + correctness)
```

The `joint_logprob` and `probability` are also sent to the **reflection LLM** as diagnostic feedback, and exposed in `objective_scores` for Pareto-based selection.  But the single **score** is what drives GEPA's evolutionary search.

## Scoring Strategies

All strategies follow the same contract:

- **Incorrect answer** → always `0.0`, regardless of confidence.
- **Correct answer, logprobs unavailable** (`None`) → always `1.0` (graceful degradation to binary scoring).
- **Correct answer, logprobs available** → depends on the strategy (see below).

### LinearBlendScoring (recommended)

Proportionally penalizes low-confidence correct answers.  The probability (derived from `exp(logprob)`) is compared against a threshold:

- **Probability ≥ threshold** → score = `1.0` (confident and correct, full credit)
- **Probability < threshold** → score is linearly interpolated between `min_score_on_correct` and `1.0`

The formula for the interpolated region is:

```
score = min_score + (1.0 - min_score) × (probability / threshold)
```

```python
from gepa.adapters.confidence_adapter import LinearBlendScoring

strategy = LinearBlendScoring(
    low_confidence_threshold=0.5,  # probability above which correct = 1.0
    min_score_on_correct=0.3,      # floor for correct but very uncertain
)
```

**Example scores** (with `threshold=0.5`, `min_score=0.3`):

| Answer | Joint Logprob | Probability | Score | Why |
|---|:---:|:---:|:---:|---|
| Correct, very confident | `-0.11` | ~90% | `1.0` | Above threshold |
| Correct, at threshold | `-0.69` | ~50% | `1.0` | At threshold boundary |
| Correct, uncertain | `-1.20` | ~30% | `0.72` | Interpolated: `0.3 + 0.7 × (0.30/0.5)` |
| Correct, very uncertain | `-3.00` | ~5% | `0.37` | Interpolated: `0.3 + 0.7 × (0.05/0.5)` |
| Correct, near-random | `-10.0` | ~0% | `0.30` | Floor (`min_score_on_correct`) |
| Incorrect (any) | any | any | `0.0` | Always zero |

### ThresholdScoring

Binary gate: `1.0` only if correct **and** `exp(logprob)` ≥ threshold.  Everything else is `0.0`.

This is the strictest strategy — it completely discards correct answers where the model was not confident enough.

```python
from gepa.adapters.confidence_adapter import ThresholdScoring

strategy = ThresholdScoring(threshold=0.7)
```

**Example scores** (with `threshold=0.7`):

| Answer | Joint Logprob | Probability | Score | Why |
|---|:---:|:---:|:---:|---|
| Correct, confident | `-0.11` | ~90% | `1.0` | Above threshold |
| Correct, at threshold | `-0.36` | ~70% | `1.0` | At threshold boundary |
| Correct, below threshold | `-0.51` | ~60% | `0.0` | Below threshold → rejected |
| Correct, uncertain | `-2.30` | ~10% | `0.0` | Below threshold → rejected |
| Incorrect (any) | any | any | `0.0` | Always zero |

### SigmoidScoring

Smooth S-curve that maps probability to a score when correct.  Unlike the linear strategy, there is no hard threshold — the transition from low to high score is gradual.

The formula is:

```
score = sigmoid(steepness × (probability - midpoint))
     = 1 / (1 + exp(-steepness × (probability - midpoint)))
```

When `probability == midpoint`, the score is exactly `0.5`.

```python
from gepa.adapters.confidence_adapter import SigmoidScoring

strategy = SigmoidScoring(midpoint=0.5, steepness=10.0)
```

**Example scores** (with `midpoint=0.5`, `steepness=10.0`):

| Answer | Joint Logprob | Probability | Score | Why |
|---|:---:|:---:|:---:|---|
| Correct, very confident | `-0.05` | ~95% | `0.99` | Far above midpoint |
| Correct, confident | `-0.22` | ~80% | `0.95` | Above midpoint |
| Correct, at midpoint | `-0.69` | ~50% | `0.50` | Exactly at midpoint |
| Correct, uncertain | `-1.20` | ~30% | `0.12` | Below midpoint |
| Correct, very uncertain | `-5.00` | ~1% | `0.01` | Far below midpoint |
| Incorrect (any) | any | any | `0.0` | Always zero |

## Multi-Objective Optimization

The adapter automatically exposes three objectives in `objective_scores`:

- `accuracy`: binary (1.0 if correct, 0.0 if not)
- `logprob`: the joint logprob value
- `probability`: `exp(logprob)`

When using GEPA's Pareto frontier, this enables selection of prompts that balance accuracy and confidence.

## Enum Resolution with `response_schema`

When you pass `response_schema=`, the library can resolve token prefixes back to full enum values.  For example, the token `"Pos"` can be resolved to `"Positive"` when the schema defines an enum with `"Positive"` as one of the allowed values.

```python
adapter = ConfidenceAdapter(
    model="openai/gpt-4.1-mini",
    field_path="category_name",
    response_format=my_json_schema,
    response_schema=my_json_schema["json_schema"]["schema"],
)
```

This improves the quality of the `top_alternatives` reported in reflective feedback.
