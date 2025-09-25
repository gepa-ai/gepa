import json
import os
from pathlib import Path
import pytest

RECORDER_DIR = Path(__file__).parent

@pytest.fixture(scope="module")
def recorder_dir() -> Path:
    """Provides the path to the recording directory and ensures it exists."""
    RECORDER_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDER_DIR


# --- The Test Function ---

def test_aime_prompt_optimize(mocked_lms, recorder_dir):
    """
    Tests the GEPA optimization process using recorded/replayed LLM calls.
    """
    # Imports for the specific test logic
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    # 1. Setup: Unpack fixtures and load data
    task_lm, reflection_lm = mocked_lms
    adapter = DefaultAdapter(model=task_lm)

    print("Initializing AIME dataset...")
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[:10]  # [3:8]

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    # 2. Execution: Run the core optimization logic
    print("Running GEPA optimization process...")
    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
        display_progress_bar=True,
    )

    # 3. Assertion: Verify the result against the golden file
    optimized_prompt_file = recorder_dir / "optimized_prompt.txt"
    best_prompt = gepa_result.best_candidate["system_prompt"]

    # In record mode, we save the "golden" result
    if os.environ.get("RECORD_TESTS", "false").lower() == "true":
        print(f"--- Saving optimized prompt to {optimized_prompt_file} ---")
        with open(optimized_prompt_file, "w") as f:
            f.write(best_prompt)
        # Add a basic sanity check to ensure the process produced a valid output
        assert isinstance(best_prompt, str) and len(best_prompt) > 0

    # In replay mode, we assert against the golden result
    else:
        print(f"--- Asserting against golden file: {optimized_prompt_file} ---")
        with open(optimized_prompt_file) as f:
            expected_prompt = f.read()
        assert best_prompt == expected_prompt
