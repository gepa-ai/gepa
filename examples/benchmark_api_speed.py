"""
Benchmark raw API speed to identify bottlenecks.
"""
import asyncio
import time
import os

os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultDataInst
import litellm


async def benchmark_single_call():
    """Test a single API call."""
    print("=" * 80)
    print("BENCHMARK: Single API Call")
    print("=" * 80)

    # Load one example
    trainset, _, _ = gepa.examples.aime.init_dataset()
    example = DefaultDataInst(
        input=trainset[0]["input"],
        answer=trainset[0]["answer"],
        id="aime_0",
        additional_context=trainset[0].get("additional_context"),
    )

    seed_text = "You are a helpful assistant. Answer the math question and provide your final answer in the format '### <answer>'"

    task_lm = "openrouter/openai/gpt-oss-20b:nitro"

    print(f"\nModel: {task_lm}")
    print(f"Example: {example.id}")
    print(f"\nMaking single call...\n")

    messages = [
        {"role": "system", "content": seed_text},
        {"role": "user", "content": example.input}
    ]

    start = time.time()
    response = await litellm.acompletion(
        model=task_lm,
        messages=messages,
        temperature=1.0,
    )
    elapsed = time.time() - start

    output = response.choices[0].message.content if response.choices else ""

    print(f"✅ Call completed in {elapsed:.2f}s")
    print(f"   Output length: {len(output)} chars")
    print(f"   Output preview: {output[:100]}...")

    return elapsed


async def benchmark_sequential_calls(n_calls=10):
    """Test N sequential API calls."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {n_calls} Sequential API Calls")
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
    task_lm = "openrouter/openai/gpt-oss-20b:nitro"

    print(f"\nModel: {task_lm}")
    print(f"Examples: {n_calls}")
    print(f"\nMaking {n_calls} sequential calls...\n")

    start = time.time()
    results = []
    for i, example in enumerate(examples):
        call_start = time.time()
        messages = [
            {"role": "system", "content": seed_text},
            {"role": "user", "content": example.input}
        ]
        response = await litellm.acompletion(model=task_lm, messages=messages, temperature=1.0)
        call_elapsed = time.time() - call_start
        output = response.choices[0].message.content if response.choices else ""
        results.append((output, call_elapsed))
        print(f"  Call {i+1}/{n_calls}: {call_elapsed:.2f}s")

    total_elapsed = time.time() - start
    avg_time = total_elapsed / n_calls

    print(f"\n✅ {n_calls} sequential calls completed in {total_elapsed:.2f}s")
    print(f"   Average time per call: {avg_time:.2f}s")
    print(f"   Throughput: {n_calls / total_elapsed:.2f} calls/sec")

    return avg_time, total_elapsed


async def benchmark_concurrent_calls(n_calls=10, concurrency=5):
    """Test N concurrent API calls."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {n_calls} Concurrent API Calls (concurrency={concurrency})")
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
    task_lm = "openrouter/openai/gpt-oss-20b:nitro"

    print(f"\nModel: {task_lm}")
    print(f"Examples: {n_calls}")
    print(f"Concurrency: {concurrency}")
    print(f"\nMaking {n_calls} concurrent calls...\n")

    start = time.time()

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def call_with_semaphore(example, idx):
        async with semaphore:
            call_start = time.time()
            messages = [
                {"role": "system", "content": seed_text},
                {"role": "user", "content": example.input}
            ]
            response = await litellm.acompletion(model=task_lm, messages=messages, temperature=1.0)
            call_elapsed = time.time() - call_start
            output = response.choices[0].message.content if response.choices else ""
            print(f"  Call {idx+1}/{n_calls}: {call_elapsed:.2f}s")
            return output, call_elapsed

    # Launch all calls concurrently
    tasks = [call_with_semaphore(ex, i) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks)

    total_elapsed = time.time() - start
    avg_time = total_elapsed / n_calls

    print(f"\n✅ {n_calls} concurrent calls completed in {total_elapsed:.2f}s")
    print(f"   Average time per call: {avg_time:.2f}s")
    print(f"   Throughput: {n_calls / total_elapsed:.2f} calls/sec")
    print(f"   Speedup vs sequential: {(n_calls * results[0][1]) / total_elapsed:.2f}x")

    return avg_time, total_elapsed


async def benchmark_full_seed_eval(n_examples=45, concurrency=32):
    """Benchmark evaluating seed on all examples."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: Full Seed Evaluation ({n_examples} examples, concurrency={concurrency})")
    print("=" * 80)

    trainset, _, _ = gepa.examples.aime.init_dataset()
    n_examples = min(n_examples, len(trainset))
    examples = [
        DefaultDataInst(
            input=trainset[i]["input"],
            answer=trainset[i]["answer"],
            id=f"aime_{i}",
            additional_context=trainset[i].get("additional_context"),
        )
        for i in range(n_examples)
    ]

    seed_text = "You are a helpful assistant. Answer the math question and provide your final answer in the format '### <answer>'"
    task_lm = "openrouter/openai/gpt-oss-20b:nitro"

    print(f"\nModel: {task_lm}")
    print(f"Examples: {n_examples}")
    print(f"Concurrency: {concurrency}")
    print(f"\nEvaluating seed on all examples...\n")

    start = time.time()

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    successful = 0
    failed = 0

    async def call_with_semaphore(example, idx):
        nonlocal successful, failed
        async with semaphore:
            try:
                messages = [
                    {"role": "system", "content": seed_text},
                    {"role": "user", "content": example.input}
                ]
                response = await litellm.acompletion(model=task_lm, messages=messages, temperature=1.0)
                output = response.choices[0].message.content if response.choices else ""
                # Check if it has the answer format
                has_answer = "###" in output
                successful += 1
                return {"output": output, "has_answer": has_answer}
            except Exception as e:
                failed += 1
                print(f"  ❌ Call {idx+1} failed: {str(e)[:50]}")
                return None

    # Launch all calls concurrently
    tasks = [call_with_semaphore(ex, i) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = time.time() - start

    # Calculate stats
    answered = [r.get('has_answer', False) for r in results if r and isinstance(r, dict)]
    answer_rate = sum(answered) / len(answered) if answered else 0.0

    print(f"\n✅ Evaluation completed in {total_elapsed:.2f}s")
    print(f"   Successful: {successful}/{n_examples}")
    print(f"   Failed: {failed}/{n_examples}")
    print(f"   Answer rate: {answer_rate:.1%}")
    print(f"   Throughput: {successful / total_elapsed:.2f} evals/sec")
    print(f"   Time per eval: {total_elapsed / successful:.2f}s" if successful > 0 else "")

    return answer_rate, total_elapsed


async def main():
    print("\n🔬 API Speed Benchmark\n")

    # Test 1: Single call
    single_time = await benchmark_single_call()

    # Test 2: Sequential calls
    avg_seq, total_seq = await benchmark_sequential_calls(n_calls=5)

    # Test 3: Concurrent calls
    avg_conc, total_conc = await benchmark_concurrent_calls(n_calls=10, concurrency=5)

    # Test 4: Full seed evaluation
    quality, elapsed = await benchmark_full_seed_eval(n_examples=45, concurrency=32)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Single call time: {single_time:.2f}s")
    print(f"Sequential throughput (5 calls): {5/total_seq:.2f} calls/sec")
    print(f"Concurrent throughput (10 calls, conc=5): {10/total_conc:.2f} calls/sec")
    print(f"Full evaluation (45 examples, conc=32): {45/elapsed:.2f} evals/sec")
    print(f"\nExpected time for full seed eval: {elapsed:.2f}s")
    print(f"Observed in diagnostic: ~129s for 38 evals = {38/129:.2f} evals/sec")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
