"""
Quick test of 10 concurrent calls without :nitro suffix.
"""
import asyncio
import time
import os

os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultDataInst
import litellm


async def quick_test():
    """Test 10 concurrent calls."""
    print("=" * 80)
    print("QUICK CONCURRENCY TEST: 10 calls with concurrency=10")
    print("=" * 80)

    trainset, _, _ = gepa.examples.aime.init_dataset()
    examples = [
        DefaultDataInst(
            input=trainset[i]["input"],
            answer=trainset[i]["answer"],
            id=f"aime_{i}",
            additional_context=trainset[i].get("additional_context"),
        )
        for i in range(10)
    ]

    seed_text = "You are a helpful assistant. Answer the math question and provide your final answer in the format '### <answer>'"
    task_lm = "openrouter/openai/gpt-oss-120b:nitro"

    print(f"\nModel: {task_lm}")
    print(f"Total calls: 10")
    print(f"Max concurrency: 10")
    print(f"\nLaunching all 10 calls at once...\n")

    semaphore = asyncio.Semaphore(10)
    overall_start = time.time()

    async def call_with_timing(example, idx):
        async with semaphore:
            start = time.time()
            print(f"  🚀 Starting call {idx+1}/10...", flush=True)
            try:
                messages = [
                    {"role": "system", "content": seed_text},
                    {"role": "user", "content": example.input}
                ]
                response = await litellm.acompletion(model=task_lm, messages=messages, temperature=1.0)
                end = time.time()
                duration = end - start
                output = response.choices[0].message.content if response.choices else ""
                print(f"  ✅ Call {idx+1}/10 completed in {duration:.2f}s", flush=True)
                return {"success": True, "duration": duration}
            except Exception as e:
                end = time.time()
                duration = end - start
                print(f"  ❌ Call {idx+1}/10 failed in {duration:.2f}s: {str(e)[:50]}", flush=True)
                return {"success": False, "duration": duration, "error": str(e)[:50]}

    # Launch all calls
    tasks = [call_with_timing(ex, i) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    overall_end = time.time()

    total_time = overall_end - overall_start

    # Calculate stats
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failed = len(results) - successful

    if successful > 0:
        durations = [r["duration"] for r in results if isinstance(r, dict) and r.get("success")]
        avg_duration = sum(durations) / len(durations)

        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful: {successful}/10")
        print(f"Failed: {failed}/10")
        print(f"Average call time: {avg_duration:.2f}s")
        print(f"Throughput: {successful / total_time:.2f} calls/sec")

        # Speedup calculation
        sequential_time = avg_duration * 10
        speedup = sequential_time / total_time
        print(f"\nSpeedup: {speedup:.2f}x (vs sequential)")
        print(f"Efficiency: {speedup / 10 * 100:.1f}%")

        if speedup >= 5:
            print(f"\n✅ Good concurrency! Calls are running in parallel.")
        else:
            print(f"\n⚠️  Low concurrency - possible bottleneck")


if __name__ == "__main__":
    asyncio.run(quick_test())
