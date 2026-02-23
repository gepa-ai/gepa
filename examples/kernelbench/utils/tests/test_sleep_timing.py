#!/usr/bin/env python3
"""
Experiment: Does sleep between measurements affect timing?

Measures the same baseline code 3 times:
- Without sleep (back-to-back)
- With 10 second sleep between measurements

Run: uv run python -m experiments.kernelbench.utils.tests.test_sleep_timing
"""

import time
import sys


REF_ARCH = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(4096, 4096).cuda()]

def get_init_inputs():
    return []
"""


def main():
    print("=" * 70)
    print("Sleep vs No-Sleep Timing Experiment")
    print("=" * 70)

    from experiments.kernelbench.utils.eval import acquire_gpu, execute_baseline, get_free_gpus

    free = get_free_gpus()
    print(f"Free GPUs: {free}")
    if not free:
        print("No free GPUs available. Exiting.")
        return 1

    gpu = acquire_gpu()
    print(f"Using GPU {gpu}")

    # Experiment 1: No sleep (back-to-back)
    print("\n" + "-" * 70)
    print("Experiment 1: Back-to-back measurements (no sleep)")
    print("-" * 70)

    times_no_sleep = []
    for i in range(10):
        print(f"  Run {i+1}/10...", end=" ", flush=True)
        result = execute_baseline(REF_ARCH, timeout=120, device=gpu)
        runtime = result.get("PerformanceStatsMean")
        times_no_sleep.append(runtime)
        print(f"{runtime:.4f} ms")

    # Experiment 2: With 30s sleep
    print("\n" + "-" * 70)
    print("Experiment 2: With 30s sleep between measurements")
    print("-" * 70)

    times_with_sleep = []
    for i in range(10):
        if i > 0:
            print(f"  Sleeping 30s...")
            time.sleep(30)
        print(f"  Run {i+1}/10...", end=" ", flush=True)
        result = execute_baseline(REF_ARCH, timeout=120, device=gpu)
        runtime = result.get("PerformanceStatsMean")
        times_with_sleep.append(runtime)
        print(f"{runtime:.4f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    avg_no_sleep = sum(times_no_sleep) / len(times_no_sleep)
    avg_with_sleep = sum(times_with_sleep) / len(times_with_sleep)

    print(f"\n  No sleep:    {times_no_sleep} -> avg {avg_no_sleep:.4f} ms")
    print(f"  With sleep:  {times_with_sleep} -> avg {avg_with_sleep:.4f} ms")

    diff_pct = abs(avg_no_sleep - avg_with_sleep) / avg_no_sleep * 100
    print(f"\n  Difference: {diff_pct:.2f}%")

    if diff_pct < 5:
        print("  -> Sleep has minimal impact (<5% difference)")
    else:
        print(f"  -> Sleep has noticeable impact ({diff_pct:.1f}% difference)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
