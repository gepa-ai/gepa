#!/usr/bin/env python3
"""SAIR Equational Theories — optimize a cheatsheet via GEPA.

Optimizes a cheatsheet for the SAIR Mathematics Distillation Challenge:
Equational Theories Stage 1. The cheatsheet is inserted into a fixed
prompt template alongside the equation placeholders, then sent to three
evaluation models via OpenRouter.

The task: given two equations over magmas, determine whether Equation 1
implies Equation 2 (TRUE/FALSE). Three models are evaluated with equal
weight.

The candidate is the cheatsheet text only. The evaluator assembles the
full prompt from the fixed template + cheatsheet + per-problem equations,
calls OpenRouter, extracts a TRUE/FALSE verdict, and scores correctness.

Usage:
    # 1. Prepare data (downloads from HuggingFace, creates train/val splits)
    uv run python -m examples.sair_competition.prepare_data

    # 2. Run optimization
    export OPENROUTER_API_KEY="sk-or-..."
    uv run python -m examples.sair_competition.main

Requires:
    pip install "gepa[full]" httpx datasets
    OPENROUTER_API_KEY environment variable
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent

# Fixed prompt template — the cheatsheet is inserted between the task
# description and the output format instructions.
PROMPT_TEMPLATE = """\
You are a mathematician specializing in equational theories of magmas.
Your task is to determine whether Equation 1 ({equation1}) implies Equation 2 ({equation2}) over all magmas.
{cheatsheet_section}
Output format (use exact headers without any additional text or formatting):
VERDICT: must be exactly TRUE or FALSE (in the same line).
REASONING: must be non-empty.
PROOF: required if VERDICT is TRUE, empty otherwise.
COUNTEREXAMPLE: required if VERDICT is FALSE, empty otherwise."""

# Model configs matching evaluation_models.json from the official judge repo
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "provider": "deepinfra/bf16",
        "max_tokens": 8192,
        "seed": 0,
        "reasoning_effort": "low",
        "temperature": 0.0,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "provider": "deepinfra/fp8",
        "max_tokens": 8192,
        "seed": 0,
        "temperature": 0.0,
    },
    "google/gemma-4-31b-it": {
        "provider": "novita/bf16",
        "max_tokens": 8192,
        "seed": 0,
        "temperature": 0.0,
    },
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 2
RETRY_DELAY = 2.0

# ---------------------------------------------------------------------------
# Verdict extraction (mirrors the official judge.py logic)
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(
    r"(?:VERDICT|ANSWER|RESULT)\s*[:=]\s*(TRUE|FALSE)",
    re.IGNORECASE,
)
_BOXED_RE = re.compile(r"\\boxed\{(TRUE|FALSE)\}", re.IGNORECASE)


def extract_verdict(response_text: str) -> bool | None:
    """Extract TRUE/FALSE verdict from model response.

    Priority: boxed > labeled > bare first/last line.
    Within same type: last occurrence wins.
    """
    if not response_text or not response_text.strip():
        return None

    # 1. Boxed (highest priority)
    boxed = _BOXED_RE.findall(response_text)
    if boxed:
        return boxed[-1].upper() == "TRUE"

    # 2. Labeled (VERDICT: / ANSWER: / RESULT:)
    # Filter out instruction patterns like "VERDICT: TRUE or FALSE"
    filtered = []
    for m in _VERDICT_RE.finditer(response_text):
        after = response_text[m.end() : m.end() + 20]
        if not re.match(r"\s+or\s+", after, re.IGNORECASE):
            filtered.append(m.group(1))
    if filtered:
        return filtered[-1].upper() == "TRUE"

    # 3. Bare first/last line
    lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
    if lines:
        for line in [lines[-1], lines[0]]:
            if line.upper() in ("TRUE", "FALSE"):
                return line.upper() == "TRUE"

    return None


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def render_prompt(cheatsheet: str, equation1: str, equation2: str) -> str:
    """Assemble the full prompt from template + cheatsheet + equations."""
    if cheatsheet and cheatsheet.strip():
        cheatsheet_section = cheatsheet
    else:
        cheatsheet_section = ""

    return PROMPT_TEMPLATE.format(
        equation1=equation1,
        equation2=equation2,
        cheatsheet_section=cheatsheet_section,
    )


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

def call_openrouter(
    prompt_text: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    """Call OpenRouter with the exact competition configuration."""
    config = MODEL_CONFIGS.get(model, {})

    provider_parts = config.get("provider", "").split("/")
    provider_block: dict[str, Any] = {"allow_fallbacks": False}
    if len(provider_parts) == 2:
        provider_block["order"] = [provider_parts[0]]
        provider_block["quantizations"] = [provider_parts[1]]

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": config.get("max_tokens", 8192),
        "temperature": config.get("temperature", 0.0),
        "provider": provider_block,
    }
    if config.get("seed") is not None:
        body["seed"] = config["seed"]
    if config.get("reasoning_effort"):
        body["reasoning"] = {"effort": config["reasoning_effort"]}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=240.0) as client:
                resp = client.post(OPENROUTER_URL, json=body, headers=headers)

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return {
                    "text": "", "verdict": None, "tokens_in": 0, "tokens_out": 0,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                }

            resp.raise_for_status()
            data = resp.json()

            text = ""
            if data.get("choices"):
                msg = data["choices"][0].get("message", {})
                text = msg.get("content", "") or ""

            usage = data.get("usage", {})
            return {
                "text": text,
                "verdict": extract_verdict(text),
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "error": None,
            }
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return {"text": "", "verdict": None, "tokens_in": 0, "tokens_out": 0, "error": str(e)}

    return {"text": "", "verdict": None, "tokens_in": 0, "tokens_out": 0, "error": "max retries exceeded"}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(candidate: str, example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Evaluate a cheatsheet on one (problem, model) pair.

    Args:
        candidate: The cheatsheet text (may be empty string).
        example: Dict with equation1, equation2, answer (bool), model (str).

    Returns:
        (score, side_info) where score is 1.0 (correct) or 0.0 (wrong/parse failure).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    equation1 = example["equation1"]
    equation2 = example["equation2"]
    expected = example["answer"]
    model = example["model"]
    model_alias = example.get("model_alias", model)

    # Assemble the full prompt
    prompt_text = render_prompt(candidate, equation1, equation2)

    # Check size constraint (10KB limit for complete prompt)
    prompt_bytes = len(prompt_text.encode("utf-8"))
    if prompt_bytes > 10240:
        oa.log(f"WARNING: Complete prompt is {prompt_bytes} bytes, exceeds 10KB limit")

    # Call the model
    result = call_openrouter(prompt_text, model, api_key)

    verdict = result["verdict"]
    correct = verdict == expected if verdict is not None else False
    score = 1.0 if correct else 0.0

    # Build side_info
    side_info: dict[str, Any] = {
        "model": model_alias,
        "equation1": equation1,
        "equation2": equation2,
        "expected": expected,
        "verdict": verdict,
        "correct": correct,
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "prompt_bytes": prompt_bytes,
    }

    if result["error"]:
        side_info["error"] = result["error"]
        oa.log(f"[{model_alias}] ERROR: {result['error']}")
    elif verdict is None:
        oa.log(f"[{model_alias}] PARSE FAIL on {example['id']}: could not extract verdict")
        oa.log(f"  Response (first 300 chars): {result['text'][:300]}")
    elif not correct:
        oa.log(f"[{model_alias}] WRONG on {example['id']}: expected={expected} got={verdict}")
        oa.log(f"  Eq1: {equation1}")
        oa.log(f"  Eq2: {equation2}")
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=\n(?:PROOF|COUNTEREXAMPLE|$))",
            result["text"],
            re.DOTALL,
        )
        if reasoning_match:
            oa.log(f"  Reasoning: {reasoning_match.group(1).strip()[:500]}")

    return score, side_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OBJECTIVE = (
    "Optimize a cheatsheet that helps LLMs determine whether one equation "
    "implies another over all magmas (universal algebra).\n\n"
    "The cheatsheet is inserted into a fixed prompt template between the task "
    "description and the output format instructions. It must help three "
    "different LLMs (GPT-OSS-120B, Llama-3.3-70B, Gemma-4-31B) maximize "
    "the fraction of correct TRUE/FALSE verdicts.\n\n"
    "The cheatsheet should teach the models what they need to know about "
    "equational implication over magmas to answer correctly. It does NOT "
    "need to include the task description or output format — those are "
    "already in the template.\n\n"
    "The complete prompt (template + cheatsheet) must be under 10KB.\n\n"
    "The score is binary per problem: 1.0 if the extracted verdict matches "
    "ground truth, 0.0 otherwise. The total score is the average across all "
    "problems and all three models with equal weight."
)

BACKGROUND = (
    "A magma is a set M with a single binary operation * (no axioms "
    "required — not necessarily associative, commutative, or anything else). "
    "An equational implication asks: if Equation 1 holds for all elements "
    "in every magma, does Equation 2 necessarily also hold?\n\n"
    "Example: 'x = x * y' (Eq1) implies 'x = x * x' (Eq2), because "
    "substituting y = x in Eq1 gives x = x * x. But 'x * y = y * x' "
    "(commutativity) does NOT imply 'x * (y * z) = (x * y) * z' "
    "(associativity) — there exist non-associative commutative magmas.\n\n"
    "The cheatsheet should include:\n"
    "- A clear explanation of what a magma is and what equational "
    "implication means (many models don't know this).\n"
    "- For TRUE implications: guide toward substitution arguments "
    "(replacing variables with expressions) and structural reasoning "
    "(deriving Eq2 step-by-step from Eq1).\n"
    "- For FALSE implications: guide toward constructing small finite "
    "counterexample magmas (2-3 element operation tables) where Eq1 "
    "holds but Eq2 doesn't.\n"
    "- Common patterns: equations like 'x = ...' are strong (they fix x "
    "for all inputs). Equations with more distinct variables are often "
    "weaker. Very short equations tend to be stronger.\n"
    "- Important: the cheatsheet must work across three very different "
    "model architectures. Avoid model-specific tricks. Focus on clear "
    "mathematical reasoning strategies.\n"
    "- The cheatsheet replaces the empty middle section of the prompt "
    "template. The template already has task instructions and output "
    "format — the cheatsheet should provide mathematical knowledge "
    "and problem-solving strategies, not repeat the instructions."
)


def _load_data(path: Path) -> list[dict[str, Any]]:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    train_path = DATA_DIR / "sair_train.jsonl"
    val_path = DATA_DIR / "sair_val.jsonl"

    if not train_path.exists() or not val_path.exists():
        print("Data files not found. Run prepare_data.py first:")
        print("  uv run python -m examples.sair_competition.prepare_data")
        return

    trainset = _load_data(train_path)
    valset = _load_data(val_path)

    print(f"Train: {len(trainset)} examples ({len(trainset) // 3} problems x 3 models)")
    print(f"Val:   {len(valset)} examples ({len(valset) // 3} problems x 3 models)")
    print("Seed candidate: empty cheatsheet")
    print()

    result = optimize_anything(
        seed_candidate="",
        evaluator=evaluate,
        dataset=trainset,
        valset=valset,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=GEPAConfig(
            reflection=ReflectionConfig(
                reflection_lm="openai/gpt-4.1-mini",
                reflection_minibatch_size=5,
            ),
            engine=EngineConfig(
                max_metric_calls=1500,
                seed=42,
            ),
        ),
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Best score: {result.best_score:.4f}")

    best_cheatsheet = result.best_candidate
    print(f"Best cheatsheet ({len(best_cheatsheet)} bytes):")
    print(best_cheatsheet)

    # Save best cheatsheet
    out_path = DATA_DIR / "best_cheatsheet.txt"
    with open(out_path, "w") as f:
        f.write(best_cheatsheet)
    print(f"\nSaved to {out_path}")

    # Also save the complete prompt for submission
    full_prompt = render_prompt(best_cheatsheet, "{{equation1}}", "{{equation2}}")
    submission_path = DATA_DIR / "submission_prompt.txt"
    with open(submission_path, "w") as f:
        f.write(full_prompt)
    prompt_bytes = len(full_prompt.encode("utf-8"))
    print(f"Full submission prompt saved to {submission_path} ({prompt_bytes} bytes)")
    if prompt_bytes > 10240:
        print("WARNING: Exceeds 10KB limit!")


if __name__ == "__main__":
    main()
