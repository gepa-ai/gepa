"""
Test if concurrent calls are actually running in parallel.
"""
import asyncio
import time
import os

os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultDataInst
import litellm


async def test_concurrency(n_calls=10, concurrency=10):
    """Test if concurrent calls actually run in parallel."""
    print("=" * 80)
    print(f"CONCURRENCY TEST: {n_calls} calls with concurrency={concurrency}")
    print("=" * 80)

    trainset, _, _ = gepa.examples.aime.init_dataset()
    examples = [
        DefaultDataInst(
            input=trainset[i]["input"],
            answer=trainset[i]["answer"],
            id=f"aime_{i}",
            additional_context=trainset[i].get("additional_context"),
        )
        for i in range(n_calls)
    ]

    seed_text = "You are a helpful assistant. Answer the math question and provide your final answer in the format '### <answer>'"
    task_lm = "openrouter/openai/gpt-oss-20b"

    print(f"\nModel: {task_lm}")
    print(f"Total calls: {n_calls}")
    print(f"Max concurrency: {concurrency}")
    print(f"\nExpected behavior:")
    print(f"  - If running in parallel: total time ≈ (n_calls / concurrency) * avg_call_time")
    print(f"  - If running sequentially: total time ≈ n_calls * avg_call_time")
    print(f"\nLaunching all calls at once...\n")

    # Track start time for each call
    call_times = {}

    semaphore = asyncio.Semaphore(concurrency)

    async def call_with_timing(example, idx):
        async with semaphore:
            start = time.time()
            call_times[idx] = {"start": start, "acquired_sem": start}

            try:
                messages = [
                    {"role": "system", "content": seed_text},
                    {"role": "user", "content": example.input}
                ]

                api_start = time.time()
                response = await litellm.acompletion(model=task_lm, messages=messages, temperature=1.0)
                api_end = time.time()

                end = time.time()
                call_times[idx]["api_start"] = api_start
                call_times[idx]["api_end"] = api_end
                call_times[idx]["end"] = end
                call_times[idx]["duration"] = end - start
                call_times[idx]["api_duration"] = api_end - api_start
                call_times[idx]["success"] = True

                output = response.choices[0].message.content if response.choices else ""
                return {"success": True, "output": output}
            except Exception as e:
                end = time.time()
                call_times[idx]["end"] = end
                call_times[idx]["duration"] = end - start
                call_times[idx]["success"] = False
                call_times[idx]["error"] = str(e)[:50]
                return {"success": False, "error": str(e)[:50]}

    # Launch ALL calls at once
    overall_start = time.time()
    tasks = [call_with_timing(ex, i) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    overall_end = time.time()

    total_time = overall_end - overall_start

    # Analyze timing
    print("\n" + "=" * 80)
    print("TIMING ANALYSIS")
    print("=" * 80)

    # Show when each call started and ended
    print("\nCall Timeline (relative to start):")
    print(f"{'Call':<6} {'Start':<8} {'End':<8} {'Duration':<10} {'API Time':<10} {'Status'}")
    print("-" * 80)

    for idx in sorted(call_times.keys()):
        ct = call_times[idx]
        rel_start = ct["start"] - overall_start
        rel_end = ct["end"] - overall_start
        duration = ct["duration"]
        api_duration = ct.get("api_duration", 0.0)
        status = "✅ OK" if ct["success"] else f"❌ {ct.get('error', 'ERR')}"
        print(f"#{idx:<5} {rel_start:<8.2f} {rel_end:<8.2f} {duration:<10.2f} {api_duration:<10.2f} {status}")

    # Calculate metrics
    successful = sum(1 for ct in call_times.values() if ct["success"])
    failed = len(call_times) - successful

    if successful > 0:
        durations = [ct["duration"] for ct in call_times.values() if ct["success"]]
        api_durations = [ct.get("api_duration", 0) for ct in call_times.values() if ct["success"]]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        avg_api_duration = sum(api_durations) / len(api_durations) if api_durations else 0

        # Check for parallelism
        # If calls are parallel, we should see overlapping start/end times
        start_times = [ct["start"] - overall_start for ct in call_times.values()]
        end_times = [ct["end"] - overall_start for ct in call_times.values()]

        # Count how many calls were running simultaneously at various points
        max_concurrent = 0
        for t in range(int(total_time) + 1):
            concurrent_at_t = sum(1 for idx in call_times
                                 if call_times[idx]["start"] - overall_start <= t <= call_times[idx]["end"] - overall_start)
            max_concurrent = max(max_concurrent, concurrent_at_t)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total wall-clock time: {total_time:.2f}s")
        print(f"Successful calls: {successful}/{n_calls}")
        print(f"Failed calls: {failed}/{n_calls}")
        print(f"\nPer-call timings:")
        print(f"  Average: {avg_duration:.2f}s")
        print(f"  Min: {min_duration:.2f}s")
        print(f"  Max: {max_duration:.2f}s")
        print(f"  Avg API time: {avg_api_duration:.2f}s")
        print(f"\nConcurrency metrics:")
        print(f"  Max concurrent observed: {max_concurrent}")
        print(f"  Expected max concurrent: {concurrency}")
        print(f"  Throughput: {successful / total_time:.2f} calls/sec")

        # Calculate speedup
        sequential_time = avg_duration * n_calls
        speedup = sequential_time / total_time
        ideal_speedup = min(concurrency, n_calls)

        print(f"\nSpeedup analysis:")
        print(f"  Sequential time estimate: {sequential_time:.2f}s ({n_calls} × {avg_duration:.2f}s)")
        print(f"  Actual time: {total_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Ideal speedup: {ideal_speedup:.2f}x")
        print(f"  Efficiency: {speedup / ideal_speedup * 100:.1f}%")

        if speedup < 1.5:
            print(f"\n⚠️  WARNING: Very low speedup! Calls may not be running in parallel.")
            print(f"    Check if semaphore is working correctly or if API has rate limits.")
        elif max_concurrent < concurrency * 0.5:
            print(f"\n⚠️  WARNING: Max concurrent ({max_concurrent}) much lower than expected ({concurrency})")
            print(f"    Possible bottleneck in task scheduling or API rate limiting.")
        else:
            print(f"\n✅ Concurrency is working! Observed {max_concurrent} concurrent calls.")

    return total_time, successful, failed


async def main():
    print("\n🔬 Concurrency Test\n")

    # Test 1: 10 calls with concurrency=10 (all should run at once)
    print("\n" + "=" * 80)
    print("TEST 1: 10 calls, concurrency=10 (should all run together)")
    print("=" * 80)
    await test_concurrency(n_calls=10, concurrency=10)

    # Test 2: 10 calls with concurrency=5 (should run in 2 batches)
    print("\n\n" + "=" * 80)
    print("TEST 2: 10 calls, concurrency=5 (should run in 2 batches)")
    print("=" * 80)
    await test_concurrency(n_calls=10, concurrency=5)

    # Test 3: 10 calls with concurrency=1 (should run sequentially)
    print("\n\n" + "=" * 80)
    print("TEST 3: 10 calls, concurrency=1 (should run sequentially)")
    print("=" * 80)
    await test_concurrency(n_calls=10, concurrency=1)


if __name__ == "__main__":
    asyncio.run(main())
