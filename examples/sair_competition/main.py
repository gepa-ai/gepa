#!/usr/bin/env python3
"""SAIR Equational Theories — optimize a complete prompt via GEPA.

Optimizes a complete prompt template (instructions + cheatsheet + placeholders)
for the SAIR Mathematics Distillation Challenge: Equational Theories Stage 1.

The task: given two equations over magmas, determine whether Equation 1
implies Equation 2 (TRUE/FALSE). Three models are evaluated with equal weight.

The candidate is the complete prompt text, containing {{equation1}} and
{{equation2}} placeholders that get substituted per problem. The evaluator
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

# Budget constraint from competition: recommend ≤ $0.01 per problem
MAX_RETRIES = 2
RETRY_DELAY = 2.0

SEED_PROMPT = """\
You are a mathematician specializing in equational theories of magmas.
Your task is to determine whether Equation 1 ({{equation1}}) implies Equation 2 ({{equation2}}) over all magmas.

Output format (use exact headers without any additional text or formatting):
VERDICT: must be exactly TRUE or FALSE (in the same line).
REASONING: must be non-empty.
PROOF: required if VERDICT is TRUE, empty otherwise.
COUNTEREXAMPLE: required if VERDICT is FALSE, empty otherwise.\
"""

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
    labeled = _VERDICT_RE.findall(response_text)
    # Filter out instruction patterns like "VERDICT: TRUE or FALSE"
    filtered = []
    for match in labeled:
        # Find the full match context to check for "or"
        for m in _VERDICT_RE.finditer(response_text):
            if m.group(1).upper() == match.upper():
                after = response_text[m.end(): m.end() + 20]
                if not re.match(r"\s+or\s+", after, re.IGNORECASE):
                    filtered.append(match)
    if filtered:
        return filtered[-1].upper() == "TRUE"

    # 3. Bare first/last line
    lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
    if lines:
        for line in [lines[-1], lines[0]]:
            upper = line.upper()
            if upper in ("TRUE", "FALSE"):
                return upper == "TRUE"

    return None


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

def _render_prompt(template: str, equation1: str, equation2: str) -> str:
    """Substitute equation placeholders (matching official prompt.py)."""
    result = template
    result = result.replace("{{equation1}}", equation1)
    result = result.replace("{{ equation1 }}", equation1)
    result = result.replace("{{equation2}}", equation2)
    result = result.replace("{{ equation2 }}", equation2)
    return result


def call_openrouter(
    prompt_text: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    """Call OpenRouter with the exact competition configuration.

    Returns dict with keys: text, verdict, tokens_in, tokens_out, error.
    """
    config = MODEL_CONFIGS.get(model, {})

    # Build provider routing
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
                return {"text": "", "verdict": None, "tokens_in": 0, "tokens_out": 0,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

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
            return {"text": "", "verdict": None, "tokens_in": 0, "tokens_out": 0,
                    "error": str(e)}

    return {"text": "", "verdict": None, "tokens_in": 0, "tokens_out": 0, "error": "max retries exceeded"}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(candidate: str, example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Evaluate a complete prompt template on one (problem, model) pair.

    Args:
        candidate: The complete prompt template with {{equation1}}/{{equation2}}.
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

    # Render the prompt
    prompt_text = _render_prompt(candidate, equation1, equation2)

    # Validate placeholders were substituted
    if "{{equation1}}" in prompt_text or "{{ equation1 }}" in prompt_text:
        oa.log("WARNING: {{equation1}} placeholder not substituted — check template format")
        return 0.0, {"error": "equation1 placeholder not substituted"}

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
        # Include reasoning excerpt for reflection
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?=\n(?:PROOF|COUNTEREXAMPLE|$))", result["text"], re.DOTALL)
        if reasoning_match:
            oa.log(f"  Reasoning: {reasoning_match.group(1).strip()[:500]}")

    return score, side_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OBJECTIVE = (
    "Optimize a complete prompt template for determining whether one equation "
    "implies another over all magmas (universal algebra). The prompt is sent "
    "to three different LLMs (GPT-OSS-120B, Llama-3.3-70B, Gemma-4-31B) "
    "and must maximize the fraction of correct TRUE/FALSE verdicts.\n\n"
    "The prompt template must contain {{equation1}} and {{equation2}} "
    "placeholders that get substituted with the actual equations per problem.\n\n"
    "The evaluation extracts a VERDICT (TRUE or FALSE) from the model's "
    "response. Boxed answers (\\boxed{TRUE}) have highest priority, then "
    "labeled answers (VERDICT: TRUE), then bare first/last line answers.\n\n"
    "The score is binary: 1.0 if the extracted verdict matches ground truth, "
    "0.0 otherwise (including parse failures). The total score is the average "
    "across all problems and all three models with equal weight."
)

BACKGROUND = (
    "This is the SAIR Mathematics Distillation Challenge: Equational Theories.\n\n"
    "A magma is a set M with a single binary operation *. An equational "
    "implication asks: if Equation 1 holds for all elements in every magma, "
    "does Equation 2 necessarily also hold?\n\n"
    "For example: 'x = x * y' implies 'x = x * x' (substitute y = x). But "
    "'x * y = y * x' (commutativity) does NOT imply 'x * (y * z) = (x * y) * z' "
    "(associativity).\n\n"
    "Key strategies for the cheatsheet:\n"
    "- Teach the model what a magma is and what equational implication means.\n"
    "- Provide heuristics: for TRUE implications, guide toward substitution "
    "arguments or structural reasoning. For FALSE implications, guide toward "
    "constructing small finite counterexample magmas (2-3 elements).\n"
    "- Common patterns: equations with 'x = ...' are often stronger (they "
    "constrain x). Longer equations with more variables are often weaker.\n"
    "- The cheatsheet must be under 10KB total (including all instructions).\n"
    "- The prompt must produce output parseable as TRUE or FALSE. The safest "
    "format is 'VERDICT: TRUE' or 'VERDICT: FALSE' on its own line.\n"
    "- Three very different models will run the same prompt. The prompt must "
    "work well across model architectures — avoid model-specific tricks.\n"
    "- For FALSE cases, a concrete counterexample magma (operation table) is "
    "the strongest argument. For TRUE cases, step-by-step equational "
    "derivation from Equation 1 to Equation 2 is most reliable.\n\n"
    "The candidate must be a valid prompt template string containing "
    "{{equation1}} and {{equation2}} placeholders."
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

    print(f"Train: {len(trainset)} examples")
    print(f"Val:   {len(valset)} examples")
    print(f"Seed prompt size: {len(SEED_PROMPT)} bytes")
    print()

    result = optimize_anything(
        seed_candidate=SEED_PROMPT,
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
    print(f"Best prompt ({len(result.best_candidate)} bytes):")
    print(result.best_candidate)

    # Save best prompt
    out_path = DATA_DIR / "best_prompt.txt"
    with open(out_path, "w") as f:
        f.write(result.best_candidate)
    print(f"\nSaved to {out_path}")
    if len(result.best_candidate) > 10240:
        print(f"WARNING: Prompt is {len(result.best_candidate)} bytes, exceeds 10KB limit!")


if __name__ == "__main__":
    main()
