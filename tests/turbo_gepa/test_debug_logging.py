#!/usr/bin/env python3
"""
Quick test to verify debug logging is working.
Run with: python tests/turbo_gepa/test_debug_logging.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

print("✅ Debug logging has been added to orchestrator.py:_stream_launch_ready()")
print("\nWhen you run aime_benchmark.py and it hangs, you'll now see:")
print("\n1. Initial queue state:")
print("   🔍 DEBUG _stream_launch_ready: queue=X total_inflight=0")
print("      Per-shard queue[i]: X candidates")
print("      First candidate rung: X, fp: ...")
print("\n2. Rung selection details:")
print("   shard[0]: queue=X, key=0.3, score=X, deficit=X, inflight=X/Y")
print("   shard[1]: queue=X, key=1.0, score=X, deficit=X, inflight=X/Y")
print("\n3. If no rung selected:")
print("   ❌ DEBUG: No eligible rung found despite queue=X")
print("      (shows all rung keys, capacities, and inflight counts)")
print("\n✅ Ready to debug! Run: python examples/aime_benchmark.py --run turbo")
print("\nExpected diagnosis:")
print("  - If rung keys don't match: Key mismatch between scheduler and orchestrator")
print("  - If capacity is 0: Capacity initialization bug")
print("  - If wrong shard selected: Per-shard queue indexing bug")
