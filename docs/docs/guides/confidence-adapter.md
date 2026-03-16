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

## Scoring Strategies

All strategies receive the joint logprob, convert it to probability internally, and map `(is_correct, probability)` to a score in `[0, 1]`.  When logprobs are unavailable (e.g. the model doesn't support them), all strategies degrade gracefully to binary scoring.

### LinearBlendScoring (recommended)

Proportionally penalises low-confidence correct answers.

```python
from gepa.adapters.confidence_adapter import LinearBlendScoring

strategy = LinearBlendScoring(
    low_confidence_threshold=0.5,  # probability above which correct = 1.0
    min_score_on_correct=0.3,      # floor for correct but very uncertain
)
```

| Answer | Probability | Score |
|---|:---:|:---:|
| Correct, confident | 90% | 1.0 |
| Correct, uncertain | 30% | 0.51 |
| Correct, very uncertain | 5% | 0.37 |
| Incorrect (any confidence) | any | 0.0 |

### ThresholdScoring

Binary gate: `1.0` only if correct **and** probability >= threshold.

```python
from gepa.adapters.confidence_adapter import ThresholdScoring

strategy = ThresholdScoring(threshold=0.7)
```

### SigmoidScoring

Smooth S-curve that maps probability to score when correct.

```python
from gepa.adapters.confidence_adapter import SigmoidScoring

strategy = SigmoidScoring(midpoint=0.5, steepness=10.0)
```

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
