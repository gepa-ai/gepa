#!/usr/bin/env python3
"""Compare baseline measurements: sequential (1 GPU) vs parallel (4 GPUs)."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from examples.kernelbench.eval import (
    execute_baseline, load_dataset, init_gpu_manager
)


def main():
    # Load a subset of problems for quick comparison
    dataset = load_dataset(levels=["level1"])[:6]  # First 6 problems
    print(f"Testing {len(dataset)} problems\n")

    # Test 1: Sequential on GPU 4
    print("=" * 60)
    print("TEST 1: Sequential (1 GPU, device=4)")
    print("=" * 60)
    sequential_times = {}
    for ex in dataset:
        pid = ex.problem_id
        print(f"  {pid}...", end=" ", flush=True)
        start = time.time()
        result = execute_baseline(ex.ref_arch, timeout=120, device=4)
        elapsed = time.time() - start
        runtime = result.get("PerformanceStatsMean")
        if runtime:
            sequential_times[pid] = runtime
            print(f"{runtime:.3f} ms (took {elapsed:.1f}s)")
        else:
            print(f"FAILED: {result.get('ErrorType')}")

    # Test 2: Parallel on 4 GPUs
    print()
    print("=" * 60)
    print("TEST 2: Parallel (4 GPUs: 4,5,6,7)")
    print("=" * 60)

    # Initialize GPU manager
    init_gpu_manager(device_list=[4, 5, 6, 7], lock_dir="/tmp/gpu_locks_test")

    def measure_one(ex):
        pid = ex.problem_id
        start = time.time()
        result = execute_baseline(ex.ref_arch, timeout=120, device=None)  # Let GPUManager pick
        elapsed = time.time() - start
        runtime = result.get("PerformanceStatsMean")
        return pid, runtime, elapsed, result.get("ErrorType")

    parallel_times = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(measure_one, ex): ex for ex in dataset}
        for future in as_completed(futures):
            pid, runtime, elapsed, error = future.result()
            if runtime:
                parallel_times[pid] = runtime
                print(f"  {pid}: {runtime:.3f} ms (took {elapsed:.1f}s)")
            else:
                print(f"  {pid}: FAILED: {error}")

    # Compare results
    print()
    print("=" * 60)
    print("COMPARISON: Sequential vs Parallel")
    print("=" * 60)
    print(f"{'Problem':<50} {'Seq (ms)':>10} {'Par (ms)':>10} {'Diff %':>10}")
    print("-" * 80)

    diffs = []
    for pid in sequential_times:
        seq = sequential_times.get(pid)
        par = parallel_times.get(pid)
        if seq and par:
            diff_pct = (par - seq) / seq * 100
            diffs.append(diff_pct)
            print(f"{pid:<50} {seq:>10.3f} {par:>10.3f} {diff_pct:>+10.1f}%")

    if diffs:
        print("-" * 80)
        print(f"{'Average diff:':<50} {'':<10} {'':<10} {sum(diffs)/len(diffs):>+10.1f}%")
        print(f"{'Max diff:':<50} {'':<10} {'':<10} {max(diffs):>+10.1f}%")
        print(f"{'Min diff:':<50} {'':<10} {'':<10} {min(diffs):>+10.1f}%")


if __name__ == "__main__":
    main()
