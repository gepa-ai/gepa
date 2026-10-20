# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Optimize a prompt with GEPA using a model you serve yourself.

Start any OpenAI-compatible server first (vLLM, SGLang, Ollama, llama.cpp, ...; see
this directory's README), then point the script at it::

    uv run python examples/local_models/main.py \
        --model Qwen/Qwen3-8B \
        --api-base http://localhost:8000/v1

GEPA only talks to the server over HTTP, so nothing engine-specific is installed here.
"""

from __future__ import annotations

import argparse

import gepa
from gepa.lm import LM

# A tiny toy task for the built-in DefaultAdapter. Each example is a dict with
# `input`, `additional_context`, and `answer` (substring-matched).
TRAINSET = [
    {"input": "What is 2 + 2?", "additional_context": {}, "answer": "4"},
    {"input": "What is the capital of France?", "additional_context": {}, "answer": "Paris"},
    {"input": "What color do you get by mixing blue and yellow?", "additional_context": {}, "answer": "green"},
    {"input": "How many days are in a week?", "additional_context": {}, "answer": "7"},
]

SEED_PROMPT = {"system_prompt": "You are a helpful assistant. Answer the question."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model name as the server lists it at /v1/models.")
    parser.add_argument(
        "--api-base", required=True, help="The server's OpenAI-compatible base URL, including the /v1 suffix."
    )
    parser.add_argument("--api-key", default="EMPTY", help="Ignored by most local servers; required by the client.")
    parser.add_argument(
        "--reflection-model",
        default=None,
        help="Hosted reflection model for a mixed setup (e.g. 'openai/gpt-6.1-sol'). Defaults to the local model.",
    )
    parser.add_argument("--max-metric-calls", type=int, default=20, help="Optimization budget.")
    args = parser.parse_args()

    # "openai/" tells LiteLLM to speak the OpenAI-compatible protocol to api_base.
    local_lm = LM(f"openai/{args.model}", api_base=args.api_base, api_key=args.api_key)
    reflection_lm = args.reflection_model or local_lm

    print(f"Task model:       {local_lm!r}")
    print(f"Reflection model: {reflection_lm!r}")

    result = gepa.optimize(
        seed_candidate=SEED_PROMPT,
        trainset=TRAINSET,
        task_lm=local_lm,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
    )

    print("\n=== Result ===")
    best = result.best_candidate
    print("Best prompt:", best["system_prompt"] if isinstance(best, dict) else best)
    print("Best score: ", result.val_aggregate_scores[result.best_idx])


if __name__ == "__main__":
    main()
