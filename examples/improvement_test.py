#!/usr/bin/env python3
"""
Test that proves optimization actually improves the prompt.
Uses a simple task: capitalizing words.
"""
import os
import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "gepa"))

# Clear cache before running
cache_dir = Path(__file__).parent / "gepa" / ".turbo_gepa" / "cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("✓ Cleared cache\n")

import warnings
os.environ["LITELLM_LOG"] = "ERROR"
warnings.filterwarnings("ignore")

print("✓ Imports starting...")

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

print("✓ Imports successful\n")

# Simple capitalization task - very easy to improve
dataset = [
    DefaultDataInst(
        input="hello world",
        answer="HELLO WORLD",
        id="cap_1",
    ),
    DefaultDataInst(
        input="python code",
        answer="PYTHON CODE",
        id="cap_2",
    ),
    DefaultDataInst(
        input="test case",
        answer="TEST CASE",
        id="cap_3",
    ),
]

print(f"📊 Dataset: {len(dataset)} simple capitalization problems")
print("   Task: Convert text to uppercase")
print("   Example: 'hello world' → 'HELLO WORLD'\n")

# Minimal config
config = Config(
    shards=(1.0,),  # No ASHA
    batch_size=3,
    max_mutations_per_round=3,
    eval_concurrency=4,
    max_total_inflight=4,
    n_islands=1,
    reflection_batch_size=2,
    target_quality=1.0,  # Stop at 100%
)

print(f"🔧 Config: {len(dataset)} problems, {config.max_mutations_per_round} mutations per round\n")

# Create adapter
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",
    reflection_lm="openrouter/openai/gpt-oss-120b:nitro",
    auto_config=False,
)
adapter.config = config

print("🚀 Starting optimization...")
print("=" * 70)
start = time.time()

try:
    result = adapter.optimize(
        seeds=["You are a helpful assistant."],  # Bad seed - doesn't mention uppercase
        max_rounds=3,
        max_evaluations=15,
        enable_auto_stop=False,
        display_progress=True,
    )
    elapsed = time.time() - start

    print("=" * 70)
    print(f"\n✅ SUCCESS! Optimization completed in {elapsed:.1f} seconds\n")

    # Show improvement
    if "pareto_entries" in result and result["pareto_entries"]:
        print("📊 QUALITY IMPROVEMENT:")

        # Find seed
        seed_entry = None
        for entry in result["pareto_entries"]:
            if "You are a helpful assistant" in entry.candidate.text:
                seed_entry = entry
                break

        if seed_entry:
            seed_quality = seed_entry.result.objectives.get("quality", 0)
            print(f"   Seed:  {seed_quality:.0%}")

        # Find best
        best_entry = max(result["pareto_entries"], key=lambda e: e.result.objectives.get("quality", 0))
        best_quality = best_entry.result.objectives.get("quality", 0)
        print(f"   Best:  {best_quality:.0%}")

        if seed_entry:
            improvement = best_quality - seed_quality
            print(f"   Gain: +{improvement:.0%}")

        print(f"\n📝 Best prompt found:")
        print(f"   {best_entry.candidate.text[:150]}...")

        if best_quality > (seed_entry.result.objectives.get("quality", 0) if seed_entry else 0):
            print(f"\n🎉 OPTIMIZATION WORKED! Quality improved from {seed_quality:.0%} to {best_quality:.0%}")
        else:
            print(f"\n⚠️  No improvement detected")

    sys.exit(0)

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")
    sys.exit(1)
except Exception as e:
    elapsed = time.time() - start
    print(f"\n❌ FAILED after {elapsed:.1f} seconds")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
