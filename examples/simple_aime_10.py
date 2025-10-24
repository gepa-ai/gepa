#!/usr/bin/env python3
"""
Simple AIME benchmark - 10 problems with OSS model.
The simplest possible script.
"""

import os
import sys
import shutil
from pathlib import Path

# Clear cache for fresh results
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

print("📊 Loading 10 AIME problems...")
trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset[:5])
]
print(f"✓ Loaded {len(dataset)} problems\n")

# Simple config
config = Config(
    shards=(1.0,),
    batch_size=10,
    max_mutations_per_round=5,
    eval_concurrency=64,
    max_total_inflight=64,
    n_islands=1,
    reflection_batch_size=3,
    target_quality=0.3,  # Stop at 30% (3/10 correct)
)

print(
    f"🔧 Config: {len(dataset)} problems, {config.max_mutations_per_round} mutations/round\n"
)

# Create adapter
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",
    reflection_lm="openrouter/openai/gpt-oss-120b:nitro",
    auto_config=False,
)
adapter.config = config

print("🚀 Starting optimization...\n")

# Try to enable dashboard with plotext
try:
    from turbo_gepa.dashboard import TerminalDashboard

    dashboard = TerminalDashboard()
    metrics_callback = dashboard.update
    print("📈 Dashboard enabled\n")
except ImportError:
    metrics_callback = None
    print("📊 Dashboard not available (plotext not installed)\n")

import time

start = time.time()
SEED_PROMPT = (
    "You are a helpful assistant. You are given a question and you need to answer it. "
    "The answer should be given at the end of your response in exactly the format '### <final answer>'"
)
try:
    result = adapter.optimize(
        seeds=[SEED_PROMPT],
        max_rounds=3,
        max_evaluations=50,
        enable_auto_stop=False,
        display_progress=True,
        metrics_callback=metrics_callback,
    )
except KeyboardInterrupt:
    print("\n\n⚠️  Optimization interrupted by user.")
    sys.exit(1)

elapsed = time.time() - start

# Get best result
if "pareto" in result and result["pareto"]:
    best = max(result["pareto"], key=lambda c: c.meta.get("quality", 0))
    best_quality = best.meta.get("quality", 0)
else:
    best_quality = 0.0

print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}")
print(f"\n⏱️  Time: {elapsed:.1f} seconds")
print(f"📊 Best quality: {best_quality:.0%}")
print(f"🎯 Target was: 30%\n")

if best_quality > 0:
    print(f"✅ SUCCESS! Achieved {best_quality:.0%} quality")
else:
    print(f"⚠️  No improvement from baseline")

print()
