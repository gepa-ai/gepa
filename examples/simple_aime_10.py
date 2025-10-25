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
    for i, ex in enumerate(trainset[:10])
]
print(f"✓ Loaded {len(dataset)} problems\n")

# Simple config knobs
MAX_ROUNDS = 1
BATCH_SIZE = 10
MAX_MUTATIONS_PER_ROUND = 5
MAX_EXAMPLES_INFLIGHT = 64

config = Config(
    shards=(1.0,),
    batch_size=BATCH_SIZE,
    max_mutations_per_round=MAX_MUTATIONS_PER_ROUND,
    eval_concurrency=MAX_EXAMPLES_INFLIGHT,
    max_total_inflight=MAX_EXAMPLES_INFLIGHT,
    n_islands=1,
    reflection_batch_size=3,
    target_quality=0.3,  # Stop at 30% (3/10 correct)
)

print(
    f"🔧 Config: {len(dataset)} problems | batch_size={BATCH_SIZE} | "
    f"mutations/round={MAX_MUTATIONS_PER_ROUND} | max_examples_inflight={MAX_EXAMPLES_INFLIGHT}\n"
)

# Create adapter
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
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
        max_rounds=MAX_ROUNDS,
        enable_auto_stop=False,
        display_progress=True,
        optimize_temperature_after_convergence=False,
        metrics_callback=metrics_callback,
    )
except KeyboardInterrupt:
    print("\n\n⚠️  Optimization interrupted by user.")
    sys.exit(1)

elapsed = time.time() - start

mutation_trials = 0
if hasattr(adapter, "mutator"):
    try:
        mutation_trials = sum(
            stats.get("trials", 0) for stats in adapter.mutator._operator_stats.values()
        )
    except AttributeError:
        mutation_trials = 0

# Get best result
if "pareto" in result and result["pareto"]:
    best = max(result["pareto"], key=lambda c: c.meta.get("quality", 0))
    best_quality = best.meta.get("quality", 0)
    mutations_on_pareto = [
        c for c in result["pareto"] if c.meta.get("source") != "seed"
    ]
else:
    best_quality = 0.0
    best = None
    mutations_on_pareto = []

print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}")
print(f"\n⏱️  Time: {elapsed:.1f} seconds")
print(f"📊 Best quality: {best_quality:.0%}")
print(f"🎯 Target was: 30%\n")
print(f"🧪 Mutations evaluated: {mutation_trials}")
print(f"🧬 Mutations on Pareto: {len(mutations_on_pareto)}")
if best is not None:
    print("\n" + "=" * 70)
    print("BEST PROMPT (FULL TEXT):")
    print("=" * 70)
    print(best.text)
    print("=" * 70)
    print(f"\nPrompt length: {len(best.text)} characters")

if best_quality > 0:
    print(f"✅ SUCCESS! Achieved {best_quality:.0%} quality")
else:
    print(f"⚠️  No improvement from baseline")

print()
