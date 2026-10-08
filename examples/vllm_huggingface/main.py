# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Optimize a prompt with GEPA using a local Hugging Face model served by vLLM.

Start a vLLM OpenAI-compatible server first (see this directory's README), then::

    uv run python examples/vllm_huggingface/main.py \
        --served-model-name qwen2.5-7b-instruct \
        --api-base http://localhost:8000/v1

Core GEPA is dependency-free: this script only needs a reachable
OpenAI-compatible endpoint. ``vllm`` itself runs in the serving process, not here.
"""

from __future__ import annotations

import argparse

import gepa
from gepa.lm import make_vllm_lm

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--served-model-name", default="qwen2.5-7b-instruct",
                        help="Must match vLLM's --served-model-name.")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="OpenAI-compatible base URL, including the /v1 suffix.")
    parser.add_argument("--api-key", default="EMPTY", help="Ignored by vLLM; required by the client.")
    parser.add_argument("--reflection-model", default=None,
                        help="Override reflection model (e.g. 'openai/gpt-4o' for a mixed setup). "
                             "Defaults to the same local vLLM model.")
    parser.add_argument("--max-metric-calls", type=int, default=20, help="Optimization budget.")
    args = parser.parse_args()

    task_lm = make_vllm_lm(args.served_model_name, api_base=args.api_base, api_key=args.api_key)
    # Reflection can be the same local model, or a hosted string for a mixed setup.
    reflection_lm = args.reflection_model or make_vllm_lm(
        args.served_model_name, api_base=args.api_base, api_key=args.api_key
    )

    print(f"Task model:       {task_lm!r}")
    print(f"Reflection model: {reflection_lm!r}")

    result = gepa.optimize(
        seed_candidate=SEED_PROMPT,
        trainset=TRAINSET,
        # An LM is a plain callable, so it slots into both roles. (Pyright sees a
        # nominal Protocol param-name nuance here; the call is runtime-safe.)
        task_lm=task_lm,  # type: ignore[arg-type]
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
    )

    print("\n=== Result ===")
    best = result.best_candidate
    print("Best prompt:", best["system_prompt"] if isinstance(best, dict) else best)
    print("Best score: ", result.val_aggregate_scores[result.best_idx])


if __name__ == "__main__":
    main()
