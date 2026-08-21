# Confidence-Aware Classification Optimization

!!! info "Install the extra"
    ```bash
    pip install "gepa[confidence]"
    ```

The **ConfidenceAdapter** is a purpose-built adapter for **classification tasks** where the LLM returns a structured JSON output with `enum`-constrained fields.  It uses token-level log-probabilities to detect when the model "guesses correctly" and feeds that signal into both scoring and reflective feedback.

In a [benchmark across three datasets](../blog/posts/2026-03-17-confidence-adapter-benchmark/index.md), ConfidenceAdapter matched or beat DefaultAdapter on all tasks, with accuracy gains of **+2.10pp** on AG News and **+1.80pp** on Emotion.

## The Problem: Lucky Guesses

Standard prompt optimization evaluates a candidate prompt with binary scoring: correct → 1.0, wrong → 0.0.  This cannot distinguish between a model that genuinely understands a category and one that merely guesses correctly.

Consider a transaction categorization prompt.  The model answers `"Bills/Electricity"` and that happens to be correct.  But the token-level logprobs reveal the runner-up was `"Bills/Gas & Oil"` at almost the same probability.  The model got lucky.  Under a purely binary metric, GEPA keeps this prompt — it "works".  The next random seed or slight input variation will cause a misclassification.

## How It Works

The ConfidenceAdapter solves this by:

1. **Sending requests with `logprobs=True`** and a JSON schema `response_format` that constrains the output to an `enum` of allowed categories.

2. **Extracting the joint logprob** (sum of per-token logprobs) for the classification field via [`llm-structured-confidence`](https://github.com/rodolfonobrega/llm-structured-confidence).  This is the most natural confidence measure: `exp(joint_logprob)` gives the probability the model assigns to the entire value.

3. **Blending correctness with confidence** via a pluggable scoring strategy, so that lucky guesses receive lower scores.

4. **Feeding confidence details into the reflective feedback** so the reflection LLM knows *why* the model was uncertain and can propose prompts that resolve specific ambiguities.

### What the Reflection LLM Sees

The feedback varies based on both correctness and confidence level.

**With DefaultAdapter** (no confidence):

> *"The generated response is incorrect. The correct answer is 'Bills/Electricity'. Ensure that the correct answer is included in the response exactly as it is."*

**With ConfidenceAdapter** — correct but uncertain (below `low_confidence_threshold`):

> *"Correct but uncertain (73% probability). Model answered 'Bills/Electricity' but was nearly split with alternatives. Top alternatives: 'Bills/Gas & Oil' (24%). The model cannot reliably distinguish between these categories with the current prompt."*

**With ConfidenceAdapter** — incorrect with high confidence (above `high_confidence_threshold`):

> *"WRONG — model has 99% certainty on 'Shopping/Electronics' but the correct answer is 'Shopping/Video Games'. The model has no doubt about its wrong answer; the prompt is actively misleading it for this type of input. The prompt must add explicit rules to disambiguate 'Shopping/Electronics' vs 'Shopping/Video Games'."*

**With ConfidenceAdapter** — correct and confident:

> *"Correct."*

The strength of the feedback is proportional to the severity of the error. High-conviction mistakes get the strongest language, prompting the reflection LLM to write targeted disambiguation rules. Correct predictions above the confidence threshold get minimal feedback — no need to fix what works.

## Prerequisites

- **Structured output with enum constraints**: the adapter works best when the LLM's `response_format` forces a JSON object with one or more `enum`-constrained string fields.  This lets the logprobs reflect the model's distribution over the allowed categories.
- **Logprobs support**: the model must expose token-level logprobs.  OpenAI (`gpt-4.1`, `gpt-4.1-mini`, etc.) and Google Gemini (`gemini-2.5-flash`, etc.) both support this.

## Quick Start

```python
import gepa
from gepa.adapters.confidence_adapter import ConfidenceAdapter

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

The defaults (`high_confidence_threshold=0.99`, `low_confidence_threshold=0.90`, `LinearBlendScoring` with threshold `0.99`) are tuned for modern models like GPT-4.1-mini that produce high probabilities with structured output. See [Threshold Calibration](#threshold-calibration) if your model behaves differently.

## Adapter Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | *(required)* | LiteLLM model string or callable |
| `field_path` | *(required)* | JSON field name containing the classification (e.g., `"category_name"`) |
| `response_format` | `None` | JSON schema for structured output (passed to LiteLLM) |
| `response_schema` | `None` | Schema dict for enum resolution — maps partial tokens back to full category names |
| `scoring_strategy` | `LinearBlendScoring(0.99)` | How to convert correctness + confidence into a `[0, 1]` score |
| `high_confidence_threshold` | `0.99` | Probability above which a prediction is "high confidence" in feedback. Incorrect predictions above this threshold get the strongest error signal. |
| `low_confidence_threshold` | `0.90` | Probability below which a *correct* prediction is flagged as "unreliable" in feedback, prompting the reflection LLM to address the ambiguity. |
| `top_logprobs` | `5` | Number of alternative tokens to request from the API |
| `failure_score` | `0.0` | Score assigned when the API call fails or the response cannot be parsed |
| `max_litellm_workers` | `10` | Parallelism for `litellm.batch_completion` |

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

!!! warning "Logprobs are confidence scores, not calibrated probabilities"

    Although we write `probability = exp(joint_logprob)` throughout this guide, this value should be treated as a **confidence score** rather than a true calibrated probability.  An LLM that reports 90% confidence is not necessarily correct 90% of the time — models are often **overconfident**, meaning the raw logprobs do not reflect real-world accuracy rates.

    For the purposes of this adapter, this distinction does not affect correctness: what matters is the **relative ranking** — a logprob of `-0.05` reliably indicates more certainty than `-2.30`, and the scoring strategies use this ranking to separate confident answers from lucky guesses.

    However, if your use case requires **true probability estimates** (e.g. "the model is correct 90% of the time when it reports 90% confidence"), you should apply **calibration techniques** on top of the raw scores.  Common approaches include Platt scaling, isotonic regression, or temperature scaling.

## Threshold Calibration

A critical aspect of using ConfidenceAdapter effectively is setting the right thresholds for your model.

**LLM calibration varies significantly across models.** When the response is constrained to an enum, the constrained decoding process concentrates probability mass on the chosen token. How much it concentrates depends on the model — architecture, training data, alignment tuning (RLHF/DPO), and the decoding implementation all influence the resulting distribution.

Some models produce well-spread distributions where a 70% prediction genuinely reflects uncertainty. Others — like GPT-4.1-mini with structured output — tend to produce probabilities between 95–100% even for incorrect predictions. **You should not assume any particular distribution; instead, evaluate your model before choosing thresholds.**

### Why this matters

If your model produces consistently high probabilities and you set thresholds too low:

- The scoring collapses to binary (every correct answer gets 1.0)
- All feedback says "Correct." — no signal about uncertain predictions
- ConfidenceAdapter behaves identically to DefaultAdapter

Conversely, if you set thresholds too high for a model with spread-out distributions:

- Every correct prediction gets penalized
- Every correct prediction is flagged as "unreliable"
- The optimization signal becomes noisy

### How to choose thresholds

1. Run a small sample of predictions (50–100) with your model and structured output enabled
2. Look at the probability distribution of correct and incorrect predictions
3. Set `high_confidence_threshold` where the bulk of correct predictions cluster
4. Set `low_confidence_threshold` slightly below that — predictions below this are genuinely uncertain

**Example for different model families:**

| Model behavior | `high_confidence_threshold` | `low_confidence_threshold` | `LinearBlendScoring` threshold |
|---|---|---|---|
| Very high probabilities (GPT-4.1-mini) | `0.99` | `0.90` | `0.99` |
| Moderately high (GPT-4o, Gemini) | `0.95` | `0.80` | `0.95` |
| Well-spread distributions | `0.85` | `0.60` | `0.85` |

The defaults (`0.99` / `0.90`) work well for GPT-4.1-mini and similar models. Adjust if your model's probability distribution is different.

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
    low_confidence_threshold=0.99,  # probability above which correct = 1.0
    min_score_on_correct=0.3,       # floor for correct but very uncertain
)
```

!!! note
    The default `ConfidenceAdapter()` creates a `LinearBlendScoring` with `low_confidence_threshold` equal to `high_confidence_threshold` (0.99 by default). You only need to pass a `scoring_strategy` explicitly if you want different settings.

**Example scores** (with `threshold=0.99`, `min_score=0.3`):

| Answer | Probability | Score | Why |
|---|:---:|:---:|---|
| Correct, very confident | 99.9% | `1.0` | Above threshold |
| Correct, at threshold | 99.0% | `1.0` | At threshold boundary |
| Correct, moderately confident | 95.0% | `0.972` | Interpolated: `0.3 + 0.7 × (0.95/0.99)` |
| Correct, uncertain | 80.0% | `0.866` | Interpolated: `0.3 + 0.7 × (0.80/0.99)` |
| Correct, very uncertain | 50.0% | `0.654` | Interpolated: `0.3 + 0.7 × (0.50/0.99)` |
| Incorrect (any) | any | `0.0` | Always zero |

### ThresholdScoring

Binary gate: `1.0` only if correct **and** `exp(logprob)` ≥ threshold.  Everything else is `0.0`.

This is the strictest strategy — it completely discards correct answers where the model was not confident enough.

```python
from gepa.adapters.confidence_adapter import ThresholdScoring

strategy = ThresholdScoring(threshold=0.99)
```

**Example scores** (with `threshold=0.99`):

| Answer | Probability | Score | Why |
|---|:---:|:---:|---|
| Correct, confident | 99.5% | `1.0` | Above threshold |
| Correct, at threshold | 99.0% | `1.0` | At threshold boundary |
| Correct, below threshold | 95.0% | `0.0` | Below threshold → rejected |
| Correct, uncertain | 80.0% | `0.0` | Below threshold → rejected |
| Incorrect (any) | any | `0.0` | Always zero |

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

strategy = SigmoidScoring(midpoint=0.95, steepness=50.0)
```

**Example scores** (with `midpoint=0.95`, `steepness=50.0`):

| Answer | Probability | Score | Why |
|---|:---:|:---:|---|
| Correct, very confident | 99.5% | `0.99` | Far above midpoint |
| Correct, confident | 97.0% | `0.73` | Above midpoint |
| Correct, at midpoint | 95.0% | `0.50` | Exactly at midpoint |
| Correct, below midpoint | 93.0% | `0.27` | Below midpoint |
| Correct, uncertain | 80.0% | `0.00` | Far below midpoint |
| Incorrect (any) | any | `0.0` | Always zero |

## Multi-Objective Optimization

The adapter automatically exposes two objectives in `objective_scores`:

- `accuracy`: binary (1.0 if correct, 0.0 if not)
- `probability`: `exp(logprob)` — the joint probability of the predicted value

When using GEPA's Pareto frontier, this enables selection of prompts that balance accuracy and confidence.

## Enum Resolution with `response_schema`

When you pass `response_schema=`, the library can resolve token prefixes back to full enum values.  For example, the token `"sad"` can be resolved to `"sadness"` when the schema defines an enum with `"sadness"` as one of the allowed values.  This improves the quality of the `top_alternatives` reported in reflective feedback.

```python
adapter = ConfidenceAdapter(
    model="openai/gpt-4.1-mini",
    field_path="category_name",
    response_format=my_json_schema,
    response_schema=my_json_schema["json_schema"]["schema"],
)
```

For a detailed explanation of how token resolution works, including multi-token categories and partial-token matching, see the [blog post on structured output and logprobs](../blog/posts/2026-03-17-confidence-adapter-benchmark/index.md#structured-output-and-logprobs).
