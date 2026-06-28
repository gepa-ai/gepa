#!/usr/bin/env python3
"""Runnable template: optimize a system prompt with GEPA via optimize_anything.

Uses the OptimizeAnythingConfig + engine_config API. Fill in the three TODOs
(model call, grading, data) and run:

    pip install gepa
    python optimize_prompt.py

Set credentials for the reflection LM first (Anthropic: ANTHROPIC_API_KEY; or for Bedrock:
AWS creds / AWS_BEARER_TOKEN_BEDROCK).
"""
from __future__ import annotations

import json
import os

from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything

# --- reflection LM (LiteLLM id). Bedrock inference-profile ARN or e.g. "anthropic/claude-sonnet-4-6"
REFLECTION_LM = os.environ.get(
    "GEPA_REFLECTION_LM",
    "bedrock/arn:aws:bedrock:us-east-1:000000000000:inference-profile/us.anthropic.claude-sonnet-4-6",
)

SEED_PROMPT = "You are an expert assistant. Solve the task. Output only the final answer."

N_SAMPLES = 4  # >1 to fight N=1 selection variance for stochastic models (costs N x evals)


# ---------------------------------------------------------------- your task plumbing
def run_model(system_prompt: str, example: dict) -> str:
    """TODO: call your model with `system_prompt` on `example` and return its output text."""
    raise NotImplementedError


def grade(output: str, example: dict) -> dict:
    """TODO: return a dict with at least {'score': float, ...feedback...}.
    For a gated/multi-objective score, also return 'scores': {'objA': .., 'objB': ..}."""
    raise NotImplementedError


def load_examples() -> tuple[list[dict], list[dict], list[dict]]:
    """TODO: return (dataset, valset, test_set). Keep val and test disjoint from dataset."""
    raise NotImplementedError


# ---------------------------------------------------------------- evaluator
_eval_log = open("evaluate_calls.jsonl", "a")  # your own flat per-eval record


def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
    outs, scores, infos = [], [], []
    for _ in range(N_SAMPLES):
        out = run_model(candidate, example)
        g = grade(out, example)
        outs.append(out)
        scores.append(float(g["score"]))
        infos.append(g)
    score = sum(scores) / len(scores)  # mean ~ pass@1 estimate
    info: dict = {
        "score": score,
        "n": N_SAMPLES,
        "samples": [{"output": o, **g} for o, g in zip(outs, infos)],
    }
    # multi-objective: average per-objective metrics if grade() returned 'scores'
    if "scores" in infos[0]:
        keys = infos[0]["scores"].keys()
        info["scores"] = {k: sum(g["scores"][k] for g in infos) / len(infos) for k in keys}
    _eval_log.write(json.dumps({"key": example.get("id"), "score": score}, default=str) + "\n")
    _eval_log.flush()
    return score, info


# ---------------------------------------------------------------- run
def main() -> None:
    dataset, valset, test_set = load_examples()
    result = optimize_anything(
        seed_candidate=SEED_PROMPT,
        evaluator=evaluate,
        dataset=dataset,
        valset=valset,
        test_set=test_set,  # reporting-only: seed + final candidate scored here at the end
        objective="Produce a system prompt that maximizes task quality.",
        background="State the domain rules, constraints, and required output format here.",
        config=OptimizeAnythingConfig(
            engine="gepa",
            name="prompt_opt",
            max_evals=300,
            max_concurrency=16,
            run_dir="runs/prompt_opt",
            output_dir="outputs/prompt_opt",
            engine_config={
                "reflection": {
                    "reflection_lm": REFLECTION_LM,
                    "reflection_minibatch_size": 5,
                },
                "engine": {"max_workers": 32, "seed": 0, "frontier_type": "hybrid"},
            },
        ),
    )

    print("best_score (selection set):", result.best_score)
    print("test_score (held-out avg): ", result.metadata.get("test_score"))          # avg over test_set
    print("test_scores (per-example): ", result.metadata.get("test_scores"))         # dict per example
    print("seed test_score:           ", result.metadata.get("baseline_test_score"))
    with open("best_system_prompt.txt", "w") as f:
        f.write(result.best_candidate)
    # full GEPAResult (all candidates + per-instance scores) for post-hoc analysis
    with open("gepa_result_metadata.json", "w") as f:
        json.dump({k: str(v) for k, v in result.metadata.items()}, f, indent=2)
    print("wrote best_system_prompt.txt + gepa_result_metadata.json")


if __name__ == "__main__":
    main()
