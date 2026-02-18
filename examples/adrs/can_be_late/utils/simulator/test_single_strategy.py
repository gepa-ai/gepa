#!/usr/bin/env python3
"""Test single region strategy directly."""

import sys
import subprocess

# Test the simple greedy strategy
cmd = [
    "python", "main.py",
    "--strategy", "rc_cr_threshold",  # First test with known working strategy
    "--env", "trace",
    "--trace-file", "data/real/ping_based/random_start_time/us-west-2a_k80_1/0.json",
    "--task-duration-hours", "48",
    "--deadline-hours", "52",
    "--restart-overhead-hours", "0.2",
    "--silent"
]

print("Testing rc_cr_threshold baseline...")
result = subprocess.run(cmd, capture_output=True, text=True)
print(f"Return code: {result.returncode}")
if result.returncode != 0:
    print(f"STDERR: {result.stderr}")
else:
    # Look for cost in output
    if "Final cost:" in result.stdout:
        for line in result.stdout.split('\n'):
            if "Final cost:" in line:
                print(f"Found: {line}")
    else:
        print("No 'Final cost:' found in output")
        print("Output preview:")
        print(result.stdout[:500])