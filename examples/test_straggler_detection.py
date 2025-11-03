"""
Test to verify straggler detection is working correctly.
Creates artificial slow tasks to trigger straggler cancellation.
"""

import asyncio
import random
import time
from turbo_gepa.config import Config
from turbo_gepa.adapters.default_adapter import DefaultAdapter


# Create a dataset with mock examples
def create_mock_dataset(size=20):
    return [
        {
            "input": f"Problem {i}",
            "expected_answer": f"Answer {i}",
        }
        for i in range(size)
    ]


async def mock_task_lm(prompt: str, example: dict) -> dict:
    """Mock LLM that simulates variable latency with some intentional stragglers."""
    # 70% of tasks complete quickly (0.3-1.0s)
    # 30% are stragglers (4-8s) - MORE aggressive to ensure detection
    if random.random() < 0.7:
        delay = random.uniform(0.3, 1.0)
        print(f"  Fast task for {example['input']}: {delay:.1f}s")
    else:
        delay = random.uniform(4.0, 8.0)
        print(f"  STRAGGLER task for {example['input']}: {delay:.1f}s")

    await asyncio.sleep(delay)

    # Return mock metrics
    return {
        "quality": random.uniform(0.5, 0.9),
        "neg_cost": -0.001,
        "output": f"Mock response for {example['input']}",
        "input": example["input"],
        "expected_answer": example.get("expected_answer", ""),
    }


async def mock_reflect_lm(parent: str, traces: list) -> list[str]:
    """Mock reflection that returns empty list (we don't need mutations for this test)."""
    return []


if __name__ == "__main__":
    print("=" * 80)
    print("STRAGGLER DETECTION TEST")
    print("=" * 80)
    print()
    print("This test creates artificial stragglers (30% of tasks take 4-8s, 70% take 0.3-1.0s)")
    print("With mean+1σ threshold, stragglers should be detected and cancelled.")
    print()

    dataset = create_mock_dataset(size=50)  # Increased from 20 to 50

    config = Config(
        eval_concurrency=20,  # Increased from 10 to 20 concurrent evals
        batch_size=50,
        max_mutations_per_round=0,  # No mutations - just eval seed
        shards=(1.0,),  # Single rung
        log_level="INFO",
    )

    print(f"Configuration:")
    print(f"  eval_concurrency: {config.eval_concurrency}")
    print(f"  dataset_size: {len(dataset)}")
    print(f"  Expected behavior: Fast tasks complete first, stragglers get cancelled")
    print()
    print("Expected stragglers: ~15 tasks (30% of 50)")
    print("With 20 concurrent evals, first ~35 fast tasks complete quickly,")
    print("then remaining ~15 stragglers should be detected and cancelled.")
    print()

    # Monkey-patch litellm to use our mock
    import litellm

    original_acompletion = litellm.acompletion

    async def mock_acompletion(**kwargs):
        # Extract the user message to determine which example this is
        messages = kwargs.get("messages", [])
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Create a fake example dict from the message
        example = {"input": user_content[:50], "expected_answer": "mock"}

        # Run our mock task
        result = await mock_task_lm("", example)

        # Return in litellm format
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})]
                self.usage = type('obj', (object,), {'total_tokens': 100})()

        return MockResponse(result["output"])

    litellm.acompletion = mock_acompletion

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="gpt-3.5-turbo",  # Doesn't matter, we monkey-patched
        reflection_lm="gpt-3.5-turbo",
        config=config,
        auto_config=False,
    )

    seed = "You are a helpful assistant."

    print("🚀 Starting test...")
    print()

    start_time = time.time()

    result = adapter.optimize(
        seeds=[seed],
        max_evaluations=20,
        display_progress=True,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Evaluations: {len(dataset)}")
    print(f"Throughput: {len(dataset)/elapsed:.2f} evals/sec")
    print()
    print("Expected observations in log:")
    print("  - 🔄 Starting eval messages showing tasks launching")
    print("  - ✅ Completed messages showing fast tasks finishing (0.3-1.0s)")
    print("  - 🔍 Straggler check messages showing threshold calculation")
    print("  - ⚡ Early stop messages showing stragglers being cancelled")
    print("  - 📊 Straggler stats showing cancellation summary")
    print()

    # Calculate expected timing
    # Best case: ~35 fast tasks (70%) complete in ~1s, no stragglers run
    # Worst case: Some stragglers start before detection kicks in
    expected_best = 2  # ~1s for fast tasks + overhead
    expected_worst = 5  # Some stragglers may start before cancellation

    if elapsed < expected_worst:
        print(f"✅ PASS: Completed in {elapsed:.1f}s (< {expected_worst}s expected with straggler cancellation)")
        print(f"   Expected range: {expected_best}-{expected_worst}s with aggressive straggler detection")
    elif elapsed < 10:
        print(f"⚠️  MARGINAL: Completed in {elapsed:.1f}s (slower than expected, but faster than no cancellation)")
        print(f"   Some stragglers may have run longer before being cancelled")
    else:
        print(f"❌ FAIL: Took {elapsed:.1f}s (>10s suggests stragglers were not cancelled)")
        print(f"   Expected: {expected_best}-{expected_worst}s with cancellation, ~10-15s without")
    print("=" * 80)
