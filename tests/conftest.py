import os
import json
from pathlib import Path
import pytest

# --- Pytest Fixtures ---
@pytest.fixture(scope="module")
def mocked_lms(recorder_dir):
    """
    A pytest fixture to handle record/replay logic for LLM calls.

    In 'record' mode, it calls the actual LLM API and saves the results.
    In 'replay' mode (default), it loads results from a cached JSON file.

    Takes as input `recorder_dir`, a fixture returning the path to store the llm cache in.
    """
    should_record = os.environ.get("RECORD_TESTS", "false").lower() == "true"
    cache_file = recorder_dir / "llm_cache.json"
    cache = {}

    def get_task_key(messages):
        """Creates a deterministic key from a list of message dicts."""
        # json.dumps with sort_keys=True provides a canonical representation.
        # The tuple prefix distinguishes it from reflection_lm calls.
        return str(("task_lm", json.dumps(messages, sort_keys=True)))

    def get_reflection_key(prompt):
        """Creates a deterministic key for the reflection prompt string."""
        return str(("reflection_lm", prompt))

    # --- Record Mode ---
    if should_record:
        # Lazy import litellm only when needed to avoid dependency in replay mode.
        import litellm

        print("\n--- Running in RECORD mode. Making live API calls. ---")

        def task_lm(messages):
            key = get_task_key(messages)
            if key not in cache:
                response = litellm.completion(model="openai/gpt-4.1-nano", messages=messages)
                cache[key] = response.choices[0].message.content.strip()
            return cache[key]

        def reflection_lm(prompt):
            key = get_reflection_key(prompt)
            if key not in cache:
                response = litellm.completion(model="openai/gpt-4.1", messages=[{"role": "user", "content": prompt}])
                cache[key] = response.choices[0].message.content.strip()
            return cache[key]

        # Yield the live functions to the test, then save the cache on teardown.
        yield task_lm, reflection_lm

        print(f"--- Saving cache to {cache_file} ---")
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    # --- Replay Mode ---
    else:
        print("\n--- Running in REPLAY mode. Using cached API calls. ---")
        try:
            with open(cache_file) as f:
                cache = json.load(f)
        except FileNotFoundError:
            pytest.fail(f"Cache file not found: {cache_file}. Run with 'RECORD_TESTS=true pytest' to generate it.")

        def task_lm(messages):
            key = get_task_key(messages)
            if key not in cache:
                pytest.fail(f"Unseen input for task_lm in replay mode. Key: {key}")
            return cache[key]

        def reflection_lm(prompt):
            key = get_reflection_key(prompt)
            if key not in cache:
                pytest.fail(f"Unseen input for reflection_lm in replay mode. Key: {key}")
            return cache[key]

        yield task_lm, reflection_lm
