#!/usr/bin/env python3
# %% [markdown]
# # ConfidenceAdapter vs DefaultAdapter — Benchmark
#
# Compares two GEPA prompt-optimization adapters for LLM text classification:
#
# - **DefaultAdapter** — binary scoring (correct = 1, wrong = 0)
# - **ConfidenceAdapter** — logprob-based confidence scoring + richer reflection feedback
#
# **Datasets**: AG News (4-class), Emotion (6-class), Rotten Tomatoes (binary)
# **Task model**: openai/gpt-4.1-mini (temperature=0, no reasoning)
# **Reflection model**: anthropic/claude-sonnet-4-6 (thinking enabled, budget_tokens=1024)
#
# ## Required environment variables
#
# By default this script calls the OpenAI and Anthropic APIs directly:
#   OPENAI_API_KEY    — for the task model (gpt-4.1-mini)
#   ANTHROPIC_API_KEY — for the reflection model (claude-sonnet-4-6)
#
# If you use an OpenAI-compatible proxy that routes all models (including
# Anthropic) through a single endpoint, set OPENAI_API_BASE and change
# USE_LITELLM_PROXY to True below.  With a proxy you only need
# OPENAI_API_KEY — the proxy handles Anthropic auth internally.
#
# Run from the repo root:
#   uv run python -m examples.confidence_adapter.main
#
# Or cell-by-cell in VS Code / Jupyter.

# %% Imports and Configuration
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import litellm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from llm_structured_confidence import extract_confidence
from sklearn.metrics import roc_auc_score, roc_curve

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TASK_MODEL = "openai/gpt-4.1-mini"
REFLECTION_MODEL = "anthropic/claude-sonnet-4-6"

# Set to True if you use an OpenAI-compatible proxy (e.g. LiteLLM gateway)
# that routes all models — including Anthropic — through OPENAI_API_BASE.
USE_LITELLM_PROXY = False
PROVIDER_KWARGS = {"custom_llm_provider": "openai"} if USE_LITELLM_PROXY else {}

# Why train > val?  In LLM classification the model doesn't "learn" from training
# examples — they only feed the reflection loop.  Each iteration, GEPA samples a
# minibatch (20/class) and evaluates the prompt to find errors.  120/class gives
# 6 unique minibatch cycles.  The val set (40/class) is evaluated in full every
# iteration and needs statistical power for reliable ranking.
TRAIN_PER_CLASS = 120
VAL_PER_CLASS = 40
MINIBATCH_PER_CLASS = 20
TEST_TOTAL = 2000
BATCH_WORKERS = 15

SEED_PROMPT = "Classify the following text into one of the given categories."

OUTPUT = Path(__file__).resolve().parent / "outputs"
for subdir in ["datasets", "optimization", "evaluation", "charts"]:
    (OUTPUT / subdir).mkdir(parents=True, exist_ok=True)

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
EMO_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
RT_LABELS = ["negative", "positive"]

print(f"Task model:       {TASK_MODEL}")
print(f"Reflection model: {REFLECTION_MODEL}")
print(f"Seed: {SEED}  |  Train/class: {TRAIN_PER_CLASS}  |  Val/class: {VAL_PER_CLASS}")
print(f"Minibatch/class: {MINIBATCH_PER_CLASS}  |  Test total: {TEST_TOTAL}")


# %% Helpers


def make_response_format(categories):
    """JSON schema forcing the LLM to output {"category": "<one of categories>"}."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"category": {"type": "string", "enum": categories}},
                "required": ["category"],
                "additionalProperties": False,
            },
        },
    }


def make_evaluator():
    """Binary evaluator for DefaultAdapter: correct = 1.0, wrong = 0.0."""
    from gepa.adapters.default_adapter.default_adapter import EvaluationResult

    def evaluator(data, response_text):
        try:
            predicted = json.loads(response_text).get("category", "")
        except (json.JSONDecodeError, TypeError):
            return EvaluationResult(score=0.0, feedback="Failed to parse JSON.")
        if predicted.strip().lower() == data["answer"].strip().lower():
            return EvaluationResult(score=1.0, feedback=f"Correct: '{predicted}'.")
        return EvaluationResult(
            score=0.0,
            feedback=f"Incorrect. Expected '{data['answer']}', got '{predicted}'.",
        )

    return evaluator


# %% Reflection LM
# Claude Sonnet 4.6 with minimal thinking (budget_tokens=1024).
# GEPA uses this to analyze errors and propose improved prompts.

_THINKING_PAYLOADS = [
    ("no_thinking", None),
    ("thinking_disabled", {"thinking": {"type": "disabled"}}),
    ("thinking_enabled", {"thinking": {"type": "enabled", "budget_tokens": 1024}}),
]

# Discover which thinking payload works with this model/gateway.
print("\n--- Failfast: testing reflection model ---")
_WORKING_THINKING_PAYLOAD = None
for _name, _extra in _THINKING_PAYLOADS:
    _kwargs = {"model": REFLECTION_MODEL, "messages": [{"role": "user", "content": "Say OK."}],
               "max_tokens": 16, **PROVIDER_KWARGS}
    if _extra is not None:
        _kwargs["extra_body"] = _extra
    try:
        _r = litellm.completion(**_kwargs)
        print(f"  Reflection model OK ({_name}): {_r.choices[0].message.content[:40]}")
        _WORKING_THINKING_PAYLOAD = _extra
        break
    except Exception as _e:
        print(f"  {_name} failed: {str(_e).splitlines()[0][:100]}")
else:
    raise RuntimeError("Reflection model unreachable — fix API keys / proxy config before continuing.")


def reflection_lm(prompt):
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    kwargs = {"model": REFLECTION_MODEL, "messages": messages, "max_tokens": 4096, **PROVIDER_KWARGS}
    if _WORKING_THINKING_PAYLOAD is not None:
        kwargs["extra_body"] = _WORKING_THINKING_PAYLOAD
    for attempt in range(3):
        try:
            return litellm.completion(**kwargs).choices[0].message.content
        except Exception:
            if attempt >= 2:
                raise
            time.sleep(1.5 * (attempt + 1))


# %% Verify: LLM call + confidence extraction
# Quick sanity check: one structured-output call with logprobs, then confidence
# extraction.  Confirms the full pipeline works before the long experiment runs.

print("\n--- Failfast: testing task model ---")
_test_rf = make_response_format(AG_LABELS)
_test_schema = _test_rf["json_schema"]["schema"]
_test_resp = litellm.completion(
    model=TASK_MODEL,
    messages=[
        {"role": "system", "content": SEED_PROMPT},
        {"role": "user", "content": "Oil prices surged to a 6-month high as OPEC cuts output."},
    ],
    response_format=_test_rf,
    logprobs=True,
    top_logprobs=5,
    seed=SEED,
    temperature=0,
    **PROVIDER_KWARGS,
)
_test_conf = extract_confidence(_test_resp, field_path="category", response_schema=_test_schema)

print("Input:       'Oil prices surged to a 6-month high as OPEC cuts output.'")
print(f"Prediction:  {_test_resp.choices[0].message.content}")
print(f"Probability: {_test_conf.get('joint_probability', 'N/A')}")
print(
    f"Top alt:     {_test_conf.get('top_alternative_resolved', 'N/A')} "
    f"({_test_conf.get('top_alternative_probability', 'N/A')})"
)
print("--- Verification OK ---\n")


# %% Load datasets


def load_and_split(dataset_name, label_names):
    """Load a HuggingFace dataset and produce stratified train/val/test splits."""
    rng = random.Random(SEED)
    num_classes = len(label_names)

    hf_names = {"ag_news": "ag_news", "emotion": "dair-ai/emotion", "rotten_tomatoes": "rotten_tomatoes"}
    src_train = load_dataset(hf_names[dataset_name], split="train")
    src_test = load_dataset(hf_names[dataset_name], split="test")

    by_class = defaultdict(list)
    for item in src_train:
        by_class[item["label"]].append(item)
    for v in by_class.values():
        rng.shuffle(v)

    train_data, val_data = [], []
    for idx in sorted(by_class):
        pool = by_class[idx]
        for item in pool[:TRAIN_PER_CLASS]:
            train_data.append({"input": item["text"], "answer": label_names[item["label"]], "additional_context": {}})
        for item in pool[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS]:
            val_data.append({"input": item["text"], "answer": label_names[item["label"]], "additional_context": {}})
    rng.shuffle(train_data)
    rng.shuffle(val_data)

    by_class_test = defaultdict(list)
    for item in src_test:
        by_class_test[item["label"]].append(item)
    for v in by_class_test.values():
        rng.shuffle(v)

    per_class_test = TEST_TOTAL // num_classes
    test_data = []
    for idx in sorted(by_class_test):
        pool = by_class_test[idx]
        for item in pool[: min(per_class_test, len(pool))]:
            test_data.append({"text": item["text"], "expected": label_names[item["label"]]})
    rng.shuffle(test_data)

    print(f"  {dataset_name}: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


print("Loading datasets...")
ag_train, ag_val, ag_test = load_and_split("ag_news", AG_LABELS)
emo_train, emo_val, emo_test = load_and_split("emotion", EMO_LABELS)
rt_train, rt_val, rt_test = load_and_split("rotten_tomatoes", RT_LABELS)

for name, train, val, test in [
    ("ag_news", ag_train, ag_val, ag_test),
    ("emotion", emo_train, emo_val, emo_test),
    ("rotten_tomatoes", rt_train, rt_val, rt_test),
]:
    pd.DataFrame(train).to_csv(OUTPUT / "datasets" / f"{name}_train.csv", index=False)
    pd.DataFrame(val).to_csv(OUTPUT / "datasets" / f"{name}_val.csv", index=False)
    pd.DataFrame(test).to_csv(OUTPUT / "datasets" / f"{name}_test.csv", index=False)
print("Saved all splits to output/datasets/")


# %% Evaluate a prompt on a test set


def evaluate_on_test(prompt, test_data, categories, dataset_name, condition):
    """Evaluate a prompt on a test set using parallel LLM calls with logprobs."""
    rf = make_response_format(categories)
    schema = rf["json_schema"]["schema"]

    messages_batch = [
        [{"role": "system", "content": prompt}, {"role": "user", "content": item["text"]}] for item in test_data
    ]

    all_responses = []
    chunk_size = 100
    for start in range(0, len(test_data), chunk_size):
        chunk = messages_batch[start : start + chunk_size]
        for attempt in range(1, 4):
            try:
                resps = litellm.batch_completion(
                    model=TASK_MODEL,
                    messages=chunk,
                    max_workers=BATCH_WORKERS,
                    response_format=rf,
                    logprobs=True,
                    top_logprobs=5,
                    seed=SEED,
                    temperature=0,
                    **PROVIDER_KWARGS,
                )
                all_responses.extend(resps)
                break
            except Exception as e:
                if attempt >= 3:
                    raise
                print(f"    Retry {attempt}: {e}")
                time.sleep(3 * attempt)
        done = min(start + chunk_size, len(test_data))
        if done % 500 == 0 or done == len(test_data):
            print(f"    [{done}/{len(test_data)}]")

    results = []
    for item, resp in zip(test_data, all_responses, strict=False):
        if isinstance(resp, Exception):
            results.append(
                {
                    "text": item["text"][:300],
                    "expected": item["expected"],
                    "predicted": "",
                    "is_correct": False,
                    "score": 0.0,
                    "top_alternative": "",
                    "top_alternative_score": 0.0,
                }
            )
            continue

        content = resp.choices[0].message.content
        try:
            predicted = json.loads(content).get("category", "")
        except (json.JSONDecodeError, TypeError):
            predicted = ""

        conf = extract_confidence(resp, field_path="category", response_schema=schema)
        score = conf.get("joint_probability") or conf.get("mean_nonzero_probability") or 0.0
        top_alt = conf.get("top_alternative_resolved") or conf.get("top_alternative") or ""
        top_alt_score = conf.get("top_alternative_probability") or 0.0
        is_correct = predicted.strip().lower() == item["expected"].strip().lower()

        results.append(
            {
                "text": item["text"][:300],
                "expected": item["expected"],
                "predicted": predicted,
                "is_correct": is_correct,
                "score": round(score, 6),
                "top_alternative": top_alt,
                "top_alternative_score": round(top_alt_score, 6),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / "evaluation" / f"{dataset_name}_{condition}.csv", index=False)

    acc = df["is_correct"].mean()
    ms = df["score"].mean()
    msc = df.loc[df["is_correct"], "score"].mean() if df["is_correct"].any() else 0
    msw = df.loc[~df["is_correct"], "score"].mean() if (~df["is_correct"]).any() else 0
    print(f"  {dataset_name}/{condition}: acc={acc:.4f}, score={ms:.4f} (correct={msc:.4f}, wrong={msw:.4f})")

    return df, {
        "dataset": dataset_name,
        "condition": condition,
        "accuracy": round(acc, 4),
        "mean_score": round(ms, 4),
        "mean_score_correct": round(msc, 4),
        "mean_score_wrong": round(msw, 4),
        "total": len(df),
        "correct": int(df["is_correct"].sum()),
        "prompt": prompt,
    }


# %% Run GEPA optimization

import gepa
from gepa.adapters.confidence_adapter import ConfidenceAdapter
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter


def run_optimization(adapter, trainset, valset, dataset_name, adapter_name, categories):
    """Run GEPA optimization and save results JSON."""
    num_classes = len(categories)
    minibatch = MINIBATCH_PER_CLASS * num_classes
    budget = 10 * (minibatch + len(valset)) + len(valset)

    print(
        f"\n  Optimizing {dataset_name}/{adapter_name} "
        f"(train={len(trainset)}, val={len(valset)}, minibatch={minibatch})..."
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": SEED_PROMPT},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=minibatch,
        max_metric_calls=budget,
        seed=SEED,
        display_progress_bar=True,
        raise_on_exception=False,
    )

    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"  Best val_score={best_score:.4f}, candidates={len(result.candidates)}")

    info = {
        "dataset": dataset_name,
        "adapter": adapter_name,
        "best_val_score": best_score,
        "best_prompt": result.best_candidate["system_prompt"],
        "num_candidates": len(result.candidates),
        "val_scores_history": result.val_aggregate_scores,
    }
    with open(OUTPUT / "optimization" / f"{dataset_name}_{adapter_name}.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return result


# %% Step 1: Baseline evaluation
# Evaluate the simple seed prompt on all test sets — no optimization.

print("\n" + "=" * 60)
print("STEP 1: BASELINE (seed prompt, no optimization)")
print("=" * 60)

baselines = {}
for name, test, labels in [
    ("ag_news", ag_test, AG_LABELS),
    ("emotion", emo_test, EMO_LABELS),
    ("rotten_tomatoes", rt_test, RT_LABELS),
]:
    df, summary = evaluate_on_test(SEED_PROMPT, test, labels, name, "baseline")
    baselines[name] = {"df": df, "summary": summary}


# %% Step 2: GEPA Optimization
# For each dataset, optimize with DefaultAdapter (binary scoring) then
# ConfidenceAdapter (logprob-based scoring).  Both get identical budget and data.

print("\n" + "=" * 60)
print("STEP 2: GEPA OPTIMIZATION")
print("=" * 60)

opt_results = {}

for name, train, val, labels in [
    ("ag_news", ag_train, ag_val, AG_LABELS),
    ("emotion", emo_train, emo_val, EMO_LABELS),
    ("rotten_tomatoes", rt_train, rt_val, RT_LABELS),
]:
    rf = make_response_format(labels)
    schema = rf["json_schema"]["schema"]

    # --- DefaultAdapter: binary correct/incorrect scoring ---
    default_adapter = DefaultAdapter(
        model=TASK_MODEL,
        evaluator=make_evaluator(),
        litellm_batch_completion_kwargs={
            "response_format": rf,
            "seed": SEED,
            "temperature": 0,
            **PROVIDER_KWARGS,
        },
    )
    opt_results[f"{name}_default"] = run_optimization(
        default_adapter,
        train,
        val,
        name,
        "default",
        labels,
    )

    # --- ConfidenceAdapter: logprob confidence scoring ---
    # Thresholds are calibrated for GPT-4.1-mini with enum constraints:
    #   high_confidence_threshold=0.99 — only truly certain answers get full score
    #   low_confidence_threshold=0.90 — correct answers below 90% flagged as unreliable
    # LinearBlendScoring uses 0.99 as its threshold (auto-aligned from high_confidence_threshold)
    confidence_adapter = ConfidenceAdapter(
        model=TASK_MODEL,
        field_path="category",
        response_format=rf,
        response_schema=schema,
        high_confidence_threshold=0.99,
        low_confidence_threshold=0.90,
        litellm_batch_completion_kwargs={
            "seed": SEED,
            "temperature": 0,
            **PROVIDER_KWARGS,
        },
        max_litellm_workers=BATCH_WORKERS,
    )
    opt_results[f"{name}_confidence"] = run_optimization(
        confidence_adapter,
        train,
        val,
        name,
        "confidence",
        labels,
    )


# %% Step 3: Test-set evaluation
# Evaluate the best prompt from each adapter on the held-out test sets.

print("\n" + "=" * 60)
print("STEP 3: TEST-SET EVALUATION (optimized prompts)")
print("=" * 60)

test_results = {}
for name, test, labels in [
    ("ag_news", ag_test, AG_LABELS),
    ("emotion", emo_test, EMO_LABELS),
    ("rotten_tomatoes", rt_test, RT_LABELS),
]:
    for adapter_name in ["default", "confidence"]:
        key = f"{name}_{adapter_name}"
        prompt = opt_results[key].best_candidate["system_prompt"]
        df, summary = evaluate_on_test(prompt, test, labels, name, adapter_name)
        test_results[key] = {"df": df, "summary": summary}


# %% Step 4: Build combined comparison CSVs
# Side-by-side: baseline vs default vs confidence predictions for every test example.

print("\n" + "=" * 60)
print("STEP 4: COMBINED COMPARISON CSVs")
print("=" * 60)

combined = {}
for name in ["ag_news", "emotion", "rotten_tomatoes"]:
    bl = baselines[name]["df"]
    dd = test_results[f"{name}_default"]["df"]
    dc = test_results[f"{name}_confidence"]["df"]

    c = pd.DataFrame(
        {
            "text": bl["text"],
            "expected": bl["expected"],
            "baseline_predicted": bl["predicted"],
            "baseline_correct": bl["is_correct"],
            "baseline_score": bl["score"],
            "default_predicted": dd["predicted"],
            "default_correct": dd["is_correct"],
            "default_score": dd["score"],
            "default_top_alt": dd["top_alternative"],
            "default_top_alt_score": dd["top_alternative_score"],
            "confidence_predicted": dc["predicted"],
            "confidence_correct": dc["is_correct"],
            "confidence_score": dc["score"],
            "confidence_top_alt": dc["top_alternative"],
            "confidence_top_alt_score": dc["top_alternative_score"],
        }
    )
    c.to_csv(OUTPUT / "evaluation" / f"{name}_combined.csv", index=False)
    combined[name] = c
    print(f"  Saved {name}_combined.csv")


# %% Step 5: Generate charts

print("\n" + "=" * 60)
print("STEP 5: GENERATING CHARTS")
print("=" * 60)

CHARTS = OUTPUT / "charts"
DS_NAMES = ["ag_news", "emotion", "rotten_tomatoes"]
DS_DISPLAY = {
    "ag_news": "AG News\n(4 classes)",
    "emotion": "Emotion\n(6 classes)",
    "rotten_tomatoes": "Rotten Tomatoes\n(binary)",
}
COLORS = {"baseline": "#9CA3AF", "default": "#3B82F6", "confidence": "#10B981"}

# --- 5a. Accuracy comparison (grouped bar) ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(DS_NAMES))
w = 0.25
for i, (cond, label) in enumerate(
    [("baseline", "Baseline"), ("default", "DefaultAdapter"), ("confidence", "ConfidenceAdapter")]
):
    accs = []
    for name in DS_NAMES:
        if cond == "baseline":
            accs.append(baselines[name]["summary"]["accuracy"])
        else:
            accs.append(test_results[f"{name}_{cond}"]["summary"]["accuracy"])
    bars = ax.bar(x + (i - 1) * w, accs, w, label=label, color=COLORS[cond])
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
ax.set_ylabel("Accuracy")
ax.set_title("Test-Set Accuracy: Baseline vs DefaultAdapter vs ConfidenceAdapter", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([DS_DISPLAY[n] for n in DS_NAMES])
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "accuracy_comparison.png", dpi=150)
plt.close()
print("  Saved accuracy_comparison.png")


# --- 5b. Convergence (val score over iterations) ---
for name in DS_NAMES:
    fig, ax = plt.subplots(figsize=(9, 6))
    for adapter, label, color in [
        ("default", "DefaultAdapter", COLORS["default"]),
        ("confidence", "ConfidenceAdapter", COLORS["confidence"]),
    ]:
        with open(OUTPUT / "optimization" / f"{name}_{adapter}.json") as f:
            hist = json.load(f)["val_scores_history"]
        xs = list(range(1, len(hist) + 1))
        best_idx = int(np.argmax(hist))
        best_val, best_x = hist[best_idx], best_idx + 1
        ax.plot(xs, hist, "o-", color=color, linewidth=1.5, markersize=5, alpha=0.4, label=f"_{label}")
        ax.plot(
            best_x,
            best_val,
            marker="*",
            markersize=20,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
            label=f"{label}  \u2605 best = {best_val:.4f} (candidate {best_x})",
        )
    ax.set_xlabel("Candidate (iteration)", fontsize=11)
    ax.set_ylabel("Mean Candidate Score (val set)", fontsize=11)
    ax.set_title(f"Optimization Convergence — {name.replace('_', ' ').title()}", fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(alpha=0.3)
    ax.text(
        0.5,
        -0.22,
        "Each point is one candidate prompt evaluated on the val set. GEPA selects the best (\u2605), not the last.\n"
        "DefaultAdapter score = accuracy  |  ConfidenceAdapter score = confidence-weighted",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f0f0f0", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(CHARTS / f"convergence_{name}.png", dpi=150)
    plt.close()
    print(f"  Saved convergence_{name}.png")


# --- 5c. Per-class precision, recall (multiclass datasets) ---
from sklearn.metrics import precision_recall_fscore_support

for name, labels in [("ag_news", AG_LABELS), ("emotion", EMO_LABELS)]:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    x = np.arange(len(labels))
    w = 0.25
    for i, (cond, clabel, color) in enumerate(
        [
            ("baseline", "Baseline", COLORS["baseline"]),
            ("default", "DefaultAdapter", COLORS["default"]),
            ("confidence", "ConfidenceAdapter", COLORS["confidence"]),
        ]
    ):
        df_src = baselines[name]["df"] if cond == "baseline" else test_results[f"{name}_{cond}"]["df"]
        prec_per, _, _, _ = precision_recall_fscore_support(
            df_src["expected"],
            df_src["predicted"],
            labels=labels,
            average=None,
            zero_division=0,
        )
        bars = ax.bar(x + (i - 1) * w, prec_per, w, label=clabel, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel("Precision")
    ax.set_title(f"Per-Class Precision — {name.replace('_', ' ').title()}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / f"per_class_precision_{name}.png", dpi=150)
    plt.close()
    print(f"  Saved per_class_precision_{name}.png")

for name, labels in [("ag_news", AG_LABELS), ("emotion", EMO_LABELS)]:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    x = np.arange(len(labels))
    w = 0.25
    for i, (cond, clabel, color) in enumerate(
        [
            ("baseline", "Baseline", COLORS["baseline"]),
            ("default", "DefaultAdapter", COLORS["default"]),
            ("confidence", "ConfidenceAdapter", COLORS["confidence"]),
        ]
    ):
        df_src = baselines[name]["df"] if cond == "baseline" else test_results[f"{name}_{cond}"]["df"]
        _, recall_per, _, _ = precision_recall_fscore_support(
            df_src["expected"],
            df_src["predicted"],
            labels=labels,
            average=None,
            zero_division=0,
        )
        bars = ax.bar(x + (i - 1) * w, recall_per, w, label=clabel, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel("Recall")
    ax.set_title(f"Per-Class Recall — {name.replace('_', ' ').title()}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / f"per_class_recall_{name}.png", dpi=150)
    plt.close()
    print(f"  Saved per_class_recall_{name}.png")


# --- 5d. ROC curve (Rotten Tomatoes binary) ---
fig, ax = plt.subplots(figsize=(8, 7))
for cond, label, color, ls in [
    ("baseline", "Baseline", COLORS["baseline"], "--"),
    ("default", "DefaultAdapter", COLORS["default"], "-"),
    ("confidence", "ConfidenceAdapter", COLORS["confidence"], "-"),
]:
    df_roc = baselines["rotten_tomatoes"]["df"] if cond == "baseline" else test_results[f"rotten_tomatoes_{cond}"]["df"]
    y_true = (df_roc["expected"] == "positive").astype(int).values
    y_score = np.where(
        df_roc["predicted"].str.strip().str.lower() == "positive", df_roc["score"].values, 1.0 - df_roc["score"].values
    )
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})", color=color, linestyle=ls, linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Rotten Tomatoes (Binary Sentiment)", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "roc_rotten_tomatoes.png", dpi=150)
plt.close()
print("  Saved roc_rotten_tomatoes.png")


# --- 5e. Confidence distribution (ECDF: DefaultAdapter vs ConfidenceAdapter) ---
C_CORRECT, C_INCORRECT = "#10B981", "#EF4444"
fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharey=True, sharex=True)
ds_titles = {
    "ag_news": "AG News (4 classes)",
    "emotion": "Emotion (6 classes)",
    "rotten_tomatoes": "Rotten Tomatoes (binary)",
}
adapter_rows = [
    ("default", "DefaultAdapter"),
    ("confidence", "ConfidenceAdapter"),
]
for col, name in enumerate(DS_NAMES):
    for row, (adapter_key, adapter_label) in enumerate(adapter_rows):
        ax = axes[row, col]
        df = test_results[f"{name}_{adapter_key}"]["df"]
        for outcome, color, lbl in [(True, C_CORRECT, "Correct"), (False, C_INCORRECT, "Incorrect")]:
            mask = df["is_correct"] if outcome else ~df["is_correct"]
            scores = np.sort(df.loc[mask, "score"].values)
            n = len(scores)
            if n == 0:
                continue
            ecdf_y = np.arange(1, n + 1) / n
            ax.step(scores, ecdf_y, where="post", color=color, linewidth=2.5, label=f"{lbl} (n={n})")
            ax.fill_between(scores, ecdf_y, step="post", alpha=0.08, color=color)
        ax.axvline(x=0.99, color="#888888", linestyle="--", alpha=0.6, linewidth=1.2)
        cor_scores = df.loc[df["is_correct"], "score"].values
        inc_scores = df.loc[~df["is_correct"], "score"].values
        pct_c = (cor_scores >= 0.99).mean() * 100
        pct_i = (inc_scores >= 0.99).mean() * 100 if len(inc_scores) > 0 else 0
        ax.text(
            0.38,
            0.55,
            f"\u226599% confidence:\n  Correct: {pct_c:.0f}%\n  Incorrect: {pct_i:.0f}%",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
        )
        ax.set_xlim(0.30, 1.005)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")
        if row == 0:
            ax.set_title(ds_titles[name], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{adapter_label}\nCumulative fraction", fontsize=11)
        if row == 1:
            ax.set_xlabel("Confidence (probability)", fontsize=11)
fig.suptitle(
    "Confidence Distribution: DefaultAdapter vs ConfidenceAdapter",
    fontsize=14,
    fontweight="bold",
)
fig.text(
    0.5,
    0.01,
    "ECDF of probability scores split by correct (green) vs incorrect (red) predictions.\n"
    "Both adapters receive logprob scores from the same model \u2014 the difference comes from the optimized prompt.",
    ha="center",
    fontsize=9,
    color="#555555",
    style="italic",
)
fig.tight_layout()
fig.subplots_adjust(bottom=0.10, top=0.92, hspace=0.15)
fig.savefig(CHARTS / "confidence_distribution.png", dpi=150)
plt.close()
print("  Saved confidence_distribution.png")


# --- 5f. Per-class F1 (multiclass datasets) ---
for name, labels in [("ag_news", AG_LABELS), ("emotion", EMO_LABELS)]:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.2), 6))
    x = np.arange(len(labels))
    w = 0.25
    for i, (cond, clabel, color) in enumerate(
        [
            ("baseline", "Baseline", COLORS["baseline"]),
            ("default", "DefaultAdapter", COLORS["default"]),
            ("confidence", "ConfidenceAdapter", COLORS["confidence"]),
        ]
    ):
        df_src = baselines[name]["df"] if cond == "baseline" else test_results[f"{name}_{cond}"]["df"]
        _, _, f1_c, _ = precision_recall_fscore_support(
            df_src["expected"],
            df_src["predicted"],
            labels=labels,
            average=None,
            zero_division=0,
        )
        bars = ax.bar(x + (i - 1) * w, f1_c, w, label=clabel, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Class F1 Score — {name.replace('_', ' ').title()}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / f"per_class_f1_{name}.png", dpi=150)
    plt.close()
    print(f"  Saved per_class_f1_{name}.png")


# --- 5g. Precision / Recall / F1 overview ---
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharey=True)
ds_titles = {
    "ag_news": "AG News (4 classes)",
    "emotion": "Emotion (6 classes)",
    "rotten_tomatoes": "Rotten Tomatoes (binary)",
}
for ax, name in zip(axes, DS_NAMES, strict=True):
    metric_names = ["Precision", "Recall", "F1"]
    x = np.arange(len(metric_names))
    w = 0.25
    for i, (cond, clabel, color) in enumerate(
        [
            ("baseline", "Baseline", COLORS["baseline"]),
            ("default", "DefaultAdapter", COLORS["default"]),
            ("confidence", "ConfidenceAdapter", COLORS["confidence"]),
        ]
    ):
        df_src = baselines[name]["df"] if cond == "baseline" else test_results[f"{name}_{cond}"]["df"]
        labels_sorted = sorted(df_src["expected"].unique())
        p, r, f1, _ = precision_recall_fscore_support(
            df_src["expected"],
            df_src["predicted"],
            labels=labels_sorted,
            average="weighted",
            zero_division=0,
        )
        bars = ax.bar(x + (i - 1) * w, [p, r, f1], w, label=clabel, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    ax.set_title(ds_titles[name], fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Weighted average", fontsize=12)
    if ax == axes[0]:
        ax.legend(fontsize=11, loc="lower right")
fig.suptitle("Precision, Recall, and F1 (Weighted Average)", fontsize=16, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(CHARTS / "metrics_comparison.png", dpi=150)
plt.close()
print("  Saved metrics_comparison.png")


# %% Summary

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE — RESULTS")
print("=" * 60)

all_summaries = {}
for name in DS_NAMES:
    all_summaries[name] = {
        "baseline": baselines[name]["summary"],
        "default": test_results[f"{name}_default"]["summary"],
        "confidence": test_results[f"{name}_confidence"]["summary"],
    }

    bl = baselines[name]["summary"]["accuracy"]
    da = test_results[f"{name}_default"]["summary"]["accuracy"]
    ca = test_results[f"{name}_confidence"]["summary"]["accuracy"]
    print(f"\n  {name}:")
    print(f"    Baseline:          {bl:.2%}")
    print(f"    DefaultAdapter:    {da:.2%} ({da - bl:+.2%})")
    print(f"    ConfidenceAdapter: {ca:.2%} ({ca - bl:+.2%})")

    c = combined[name]
    conf_wins = int((c["confidence_correct"] & ~c["default_correct"]).sum())
    def_wins = int((c["default_correct"] & ~c["confidence_correct"]).sum())
    if conf_wins + def_wins > 0:
        print(f"    Disagreements: Confidence wins {conf_wins}, Default wins {def_wins}")

with open(OUTPUT / "summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2, default=str)

print(f"\nAll outputs saved to {OUTPUT}/")
for p in sorted(OUTPUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUTPUT)}")
