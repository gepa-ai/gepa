# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Optimize a prompt with GEPA using a model you serve yourself.

Start any OpenAI-compatible server first (vLLM, SGLang, Ollama, llama.cpp, ...; see
this directory's README), then point the script at it::

    uv run python examples/local_models/main.py \
        --model Qwen/Qwen3-8B \
        --api-base http://localhost:8000/v1

The same local model works with both GEPA entry points; pick one with ``--api``:

- ``optimize`` (default): ``gepa.optimize`` with the built-in DefaultAdapter, which
  calls the task model for you.
- ``optimize_anything``: ``gepa.optimize_anything.optimize_anything`` with your own
  evaluator, which calls the model and returns a score plus feedback.

GEPA only talks to the server over HTTP, so nothing engine-specific is installed here.
"""

from __future__ import annotations

import argparse
from typing import Any

import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult
from gepa.core.result import GEPAResult
from gepa.lm import LM
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.proposer.reflective_mutation.base import LanguageModel

# A toy QA task with a house style the seed prompt never mentions: reply with only the
# answer, lowercase, no punctuation. Chatty replies ("The capital of France is Paris.")
# score 0, so GEPA has to infer the format from the feedback.
TRAINSET = [
    {"input": "What is 2 + 2?", "additional_context": {}, "answer": "4"},
    {"input": "What is the capital of France?", "additional_context": {}, "answer": "paris"},
    {"input": "What color do you get by mixing blue and yellow?", "additional_context": {}, "answer": "green"},
    {"input": "Which planet is known as the Red Planet?", "additional_context": {}, "answer": "mars"},
    {"input": "How many days are in a week?", "additional_context": {}, "answer": "7"},
]

SEED_PROMPT = {"system_prompt": "You are a helpful assistant. Answer the question."}


def score(output: str, answer: str) -> tuple[float, str]:
    """Exact match, with feedback that tells the reflection model what was expected."""
    if output.strip() == answer:
        return 1.0, "Correct."
    return 0.0, f"Expected exactly {answer!r}, got {output.strip()!r}."


def run_gepa_optimize(local_lm: LM, reflection_lm: LanguageModel | str, max_metric_calls: int) -> GEPAResult:
    """gepa.optimize: the DefaultAdapter sends each question to ``task_lm`` and scores the answer."""
    return gepa.optimize(
        seed_candidate=SEED_PROMPT,
        trainset=TRAINSET,
        task_lm=local_lm,
        evaluator=lambda data, response: EvaluationResult(*score(response, data["answer"])),
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
    )


def run_optimize_anything(local_lm: LM, reflection_lm: LanguageModel | str, max_metric_calls: int) -> GEPAResult:
    """optimize_anything: you own the evaluator, so you call the local model yourself."""

    def evaluate(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        output = local_lm(
            [
                {"role": "system", "content": candidate["system_prompt"]},
                {"role": "user", "content": example["input"]},
            ]
        )
        value, feedback = score(output, example["answer"])
        # Everything in the info dict is shown to the reflection model as feedback.
        return value, {"Input": example["input"], "Output": output, "Feedback": feedback}

    return optimize_anything(
        seed_candidate=SEED_PROMPT,
        evaluator=evaluate,
        dataset=TRAINSET,
        objective="Write a system prompt that makes the assistant answer each question correctly and concisely.",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=max_metric_calls),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
        ),
    )


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
    parser.add_argument(
        "--api",
        choices=["optimize", "optimize_anything"],
        default="optimize",
        help="GEPA entry point to run (default: optimize).",
    )
    parser.add_argument("--max-metric-calls", type=int, default=20, help="Optimization budget.")
    args = parser.parse_args()

    # "openai/" tells LiteLLM to speak the OpenAI-compatible protocol to api_base.
    local_lm = LM(f"openai/{args.model}", api_base=args.api_base, api_key=args.api_key)
    reflection_lm = args.reflection_model or local_lm

    print(f"Entry point:      {args.api}")
    print(f"Task model:       {local_lm!r}")
    print(f"Reflection model: {reflection_lm!r}")

    run = run_gepa_optimize if args.api == "optimize" else run_optimize_anything
    result = run(local_lm, reflection_lm, args.max_metric_calls)

    print("\n=== Result ===")
    best = result.best_candidate
    print(f"Score: {result.val_aggregate_scores[0]:.2f} (seed) -> {result.val_aggregate_scores[result.best_idx]:.2f}")
    print("Best prompt:", best["system_prompt"] if isinstance(best, dict) else best)


if __name__ == "__main__":
    main()
