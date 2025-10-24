#!/usr/bin/env python3
"""
Ultra-simple benchmark - minimal test to show TurboGEPA works.

Just runs 2 rounds with 2 mutations each on 2 AIME problems.
Should complete in under 30 seconds.
"""

import os
import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Clear cache before running to ensure fresh results
cache_dir = Path(__file__).parent.parent / ".turbo_gepa" / "cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("✓ Cleared cache\n")

import warnings
os.environ["LITELLM_LOG"] = "ERROR"
warnings.filterwarnings("ignore")

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Check API key
if not os.environ.get("OPENROUTER_API_KEY"):
    print("❌ Error: OPENROUTER_API_KEY not set")
    sys.exit(1)

print("\n🏁 Ultra-Simple TurboGEPA Benchmark")
print("   2 AIME problems, 2 rounds, ~30 seconds\n")

# Load just 2 AIME problems
print("📊 Loading dataset...")
trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
    )
    for i, ex in enumerate(trainset[:2])
]
print(f"✓ Loaded {len(dataset)} problems\n")

# Ultra-minimal config WITH HIGH CONCURRENCY for speed
config = Config(
    shards=(1.0,),  # No ASHA - full eval always
    batch_size=4,  # Small batch
    max_mutations_per_round=4,  # Only 4 mutations
    eval_concurrency=64,  # HIGH concurrency
    max_total_inflight=64,
    n_islands=1,
    reflection_batch_size=2,
    target_quality=0.5,  # Stop at 50% (1 of 2 correct)
)

print(f"🔧 Config: batch_size={config.batch_size}, mutations={config.max_mutations_per_round}, concurrency={config.eval_concurrency}\n")

# Create adapter
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",
    reflection_lm="openrouter/openai/gpt-oss-120b:nitro",
    auto_config=False,
)
adapter.config = config

print("🚀 Starting optimization...\n")
start = time.time()

result = adapter.optimize(
    seeds=["You are a helpful assistant."],
    max_rounds=2,  # Just 2 rounds
    max_evaluations=20,  # Stop after 20 evals
    enable_auto_stop=False,  # Don't stop early
    display_progress=True,
)

elapsed = time.time() - start

# Get best result
if "pareto" in result and result["pareto"]:
    best = max(result["pareto"], key=lambda c: c.meta.get("quality", 0))
    best_quality = best.meta.get("quality", 0)
else:
    best_quality = 0.0

print(f"\n" + "="*70)
print(f"  RESULTS")
print(f"="*70)
print(f"\n📊 Performance:")
print(f"   Time: {elapsed:.1f} seconds")
print(f"   Best quality: {best_quality:.0%}")
print(f"   Target was: 50%")

if best_quality > 0:
    print(f"\n✅ SUCCESS! Optimization improved quality to {best_quality:.0%}")
else:
    print(f"\n⚠️  Quality did not improve (still at 0%)")
    print(f"   This can happen with hard problems - try running longer")

print(f"\n✨ Benchmark complete!\n")
