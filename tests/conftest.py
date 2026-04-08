import json
import os
import random
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def create_mocked_lms_context(cache_dir: Path):
    """
    Generator for mocked LLM functions that handle record/replay logic.

    In 'record' mode, it calls the actual LLM API and saves the results.
    In 'replay' mode (default), it loads results from a cached JSON file.

    Args:
        cache_dir: Path to directory containing llm_cache.json

    Yields:
        tuple: (task_lm, reflection_lm) callable functions
    """
    should_record = os.environ.get("RECORD_TESTS", "false").lower() == "true"
    cache_file = cache_dir / "llm_cache.json"
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


@pytest.fixture(scope="module")
def mocked_lms(recorder_dir):
    """
    A pytest fixture to handle record/replay logic for LLM calls.

    Takes as input `recorder_dir`, a fixture returning the path to store the llm cache in.
    """
    yield from create_mocked_lms_context(recorder_dir)


@pytest.fixture
def rng():
    return random.Random(42)


def _make_litellm_mock_response(content: str):
    """Create a mock litellm.completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = "stop"
    return resp


@pytest.fixture(autouse=True)
def _mock_litellm_for_llm_call_tests(request):
    """Auto-mock litellm.completion for tests marked with @pytest.mark.llm_call.

    Returns a JSON dict with randomized values that satisfies both the reflection
    proposer (expects fenced code block) and the refiner (expects JSON dict).
    """
    marker = request.node.get_closest_marker("llm_call")
    if marker is None:
        yield
        return

    call_rng = random.Random(42)

    def fake_completion(*_args, **kwargs):
        messages = kwargs.get("messages", [])
        last_content = messages[-1]["content"] if messages else ""

        # Check for multi-param candidates
        param_a_match = re.search(r'"param_a"\s*:\s*"(\d+)"', last_content)
        param_b_match = re.search(r'"param_b"\s*:\s*"(\d+)"', last_content)
        if param_a_match and param_b_match:
            a = int(param_a_match.group(1)) + call_rng.randint(-5, 5)
            b = 100 - a + call_rng.randint(-3, 3)
            response_text = json.dumps({"param_a": str(a), "param_b": str(b)})
        else:
            # Each call returns a unique candidate so the cache can't stall budget
            number_match = re.search(r'"number"\s*:\s*"(\d+)"', last_content)
            if number_match:
                old = int(number_match.group(1))
                new_number = old + call_rng.randint(-10, 10)
            else:
                new_number = 42 + call_rng.randint(-10, 10)
            response_text = json.dumps({"number": str(new_number)})

        # Wrap in fenced block for reflection proposer compatibility
        content = f"```\n{response_text}\n```"
        return _make_litellm_mock_response(content)

    with patch("litellm.completion", side_effect=fake_completion):
        yield
