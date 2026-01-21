import os
import random
from pathlib import Path

import pytest
from datasets import load_dataset

RECORDER_DIR = Path(__file__).parent


def init_pupa_dataset():
    raw_ds = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")["train"]

    def _to_inst(item):
        return {
            "input": item["user_query"],
            "additional_context": {
                "predicted_category": str(item.get("predicted_category", "")),
                "pii_units": str(item.get("pii_units", "")),
                "target_response": str(item.get("target_response", "")),
                "redacted_query": str(item.get("redacted_query", "")),
            },
            "answer": str(item["redacted_query"]),
        }

    data = [_to_inst(item) for item in raw_ds]
    rng = random.Random(0)
    rng.shuffle(data)

    mid = len(data) // 2
    trainset = data[:mid]
    valset = data[mid:]
    testset = data[: min(20, len(data))]

    return trainset, valset, testset


@pytest.fixture(scope="module")
def recorder_dir() -> Path:
    """Provides the path to the recording directory and ensures it exists."""
    RECORDER_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDER_DIR


@pytest.mark.parametrize("frontier_type", ["objective", "hybrid", "instance"])
def test_pareto_frontier_type(mocked_lms, recorder_dir, frontier_type):
    """
    End-to-end test of GEPA optimization on the PUPA dataset with different frontier tracking modes.
    """
    import gepa
    from gepa.adapters.default_adapter.default_adapter import (
        DefaultAdapter,
        DefaultDataInst,
        EvaluationResult,
    )

    task_lm, reflection_lm = mocked_lms

    # Custom evaluator that returns EvaluationResult with objective scores

    def evaluator(data: DefaultDataInst, response: str) -> EvaluationResult:
        judge_prompt = (
            "You are a strict grader. Compare the assistant response to the gold redaction.\n"
            f"GOLD:\n{data['answer'].strip()}\n\nRESPONSE:\n{response.strip()}\n\n"
            "Return only a number between 0 and 1."
        )
        quality_str = reflection_lm(judge_prompt)  # cached in llm_cache.json
        try:
            quality = float(quality_str.strip())
        except ValueError:
            quality = 0.0

        pii_units = data["additional_context"].get("pii_units", "")
        pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]
        leaked = sum(1 for pii in pii_list if pii and pii in response)
        leakage_frac = leaked / len(pii_list) if pii_list else 0.0
        leakage_score = 1.0 - leakage_frac

        total_score = (quality + leakage_score) / 2

        if total_score > 0.0:
            feedback = f"The generated response is correct. The response include the correct answer '{data['answer']}'"
        else:
            additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
            feedback = f"The generated response is incorrect. The correct answer is '{data['answer']}'. Ensure that the correct answer is included in the response exactly as it is. Here is some additional context that might be helpful:\n{additional_context_str}"
        return EvaluationResult(
            score=total_score,
            feedback=feedback,
            objective_scores={"quality": quality, "leakage": leakage_score},
        )

    adapter = DefaultAdapter(model=task_lm, evaluator=evaluator)

    trainset, valset, _ = init_pupa_dataset()
    trainset = trainset[:20]
    valset = valset[:12]

    seed_prompt = {"system_prompt": "You are a helpful assistant."}

    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        frontier_type=frontier_type,
        max_metric_calls=32,
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )
    assert gepa_result.total_metric_calls in [36, 48]

    best_score = gepa_result.val_aggregate_scores[gepa_result.best_idx]
    print(f"\n[{frontier_type}] Best score: {best_score}")
    if gepa_result.val_aggregate_subscores:
        best_subscores = gepa_result.val_aggregate_subscores[gepa_result.best_idx]
        print(f"[{frontier_type}] Objective scores: {best_subscores}")

    optimized_prompt_file = recorder_dir / f"optimized_prompt_{frontier_type}.txt"
    best_prompt = gepa_result.best_candidate["system_prompt"]

    if os.environ.get("RECORD_TESTS", "false").lower() == "true":
        with open(optimized_prompt_file, "w") as f:
            f.write(best_prompt)
        assert isinstance(best_prompt, str) and len(best_prompt) > 0
    else:
        with open(optimized_prompt_file) as f:
            expected_prompt = f.read()
        assert best_prompt == expected_prompt
