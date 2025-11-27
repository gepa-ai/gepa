import os
from pathlib import Path

import pytest

RECORDER_DIR = Path(__file__).parent


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
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    task_lm, reflection_lm = mocked_lms

    # scoring_fn returns (primary_score, objective_scores)

    def scoring_fn(data, response: str) -> tuple[float, dict[str, float]]:
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

        return quality, {"quality": quality, "leakage": leakage_score}

    adapter = DefaultAdapter(model=task_lm, scoring_fn=scoring_fn)

    trainset, valset, _ = gepa.examples.pupa.init_dataset()
    trainset = trainset[:60]
    valset = valset[:12]

    seed_prompt = {"system_prompt": "You are a helpful assistant."}

    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        frontier_type=frontier_type,
        max_metric_calls=50,
        reflection_minibatch_size=5,
        display_progress_bar=False,
    )

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
